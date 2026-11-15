"""Generate and verify counterfactual minimal pairs with a local LLM (Qwen3.8-27B-FP8, vLLM).

    PATH=$S/env_vllm/bin:$PATH $S/env_vllm/bin/python counterfactual/generate.py --split train
    PATH=$S/env_vllm/bin:$PATH $S/env_vllm/bin/python counterfactual/generate.py --split test

For every SUPPORT / CONTRADICT claim (train: the "original" groups; test: the held-out claims) the
LLM writes three minimal edits:

  claim_negation   the claim is rewritten to assert the opposite; abstracts unchanged.
                   Label flips SUPPORT <-> CONTRADICT, evidence sentences unchanged.
  evidence_flip    the gold evidence sentences are rewritten so that they report the opposite
                   finding; claim unchanged. Label flips. Other sentences (title included) may be
                   minimally edited too when they restate the finding; every edited sentence and
                   the gold evidence are the new evidence.
  evidence_weaken  the same, but the finding no longer bears on the claim (other population,
                   outcome, comparison...). Label NEI, no evidence.

Each edit is then verified: the same LLM, as a fresh zero-shot judge (the B5 prompt), must give the
original input its gold label and the edited input the intended label. Edits that change nothing,
or rewrite most of the text, are rejected before judging. Results go to counterfactual/data/<split>.jsonl,
one record per (claim, edit type) with the judge's verdicts and a `verified` flag.
"""

import argparse
import difflib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "baselines")]

from common import TEST_DIR, TRAIN_DIR, load_samples  # noqa: E402
from llm_zero_shot import MODEL, build_prompt, label_distribution, schema  # noqa: E402

OUT_DIR = ROOT / "counterfactual" / "data"
FLIP = {"SUPPORT": "CONTRADICT", "CONTRADICT": "SUPPORT"}
EDIT_TYPES = ("claim_negation", "evidence_flip", "evidence_weaken")
SYSTEM = "You are an expert scientific editor who writes minimal counterfactual edits."

CLAIM_NEGATION = """Rewrite the scientific claim below so that it asserts the OPPOSITE of what it \
currently asserts, changing as few words as possible. Prefer replacing a word with its antonym \
(increases -> decreases, higher -> lower, associated -> unrelated, effective -> ineffective) over \
simply inserting "not". Keep the topic, entities, numbers and style identical. The result must be \
a fluent claim that someone could plausibly make.

Claim: {claim}

Answer with JSON only: {{"claim": "<rewritten claim>"}}"""

EVIDENCE_EDIT = """Below is a claim and the abstract(s) it cites. The sentences marked >> are the \
evidence; together with the rest of the abstract they currently {relation} the claim.

Claim: {claim}

{abstracts}

Rewrite the marked sentences, changing as few words as possible, so that {goal} Keep every \
other detail of each sentence (study design, populations, numbers that are unaffected, wording, \
length). If an unmarked sentence or the title states the same finding, edit it minimally as well \
so that the whole abstract is coherent with the change; leave all other sentences out of the \
answer. Do not mention the claim.

Answer with JSON only: {{"sentences": {{"<sentence id>": "<rewritten sentence>", ...}}, \
"titles": {{"<article id>": "<rewritten title>", ...}}}} with an entry for every marked sentence \
and only for the unmarked sentences and titles you changed."""

GOALS = {
    "evidence_flip": "the abstract instead {flipped} the claim: the finding is reversed (e.g. the effect \
goes in the other direction, a significant result becomes non-significant, presence becomes absence).",
    "evidence_weaken": "the abstract NEITHER supports NOR contradicts the claim any more: the finding \
must stay a real result but no longer bear on the claim, e.g. it concerns a different population, \
outcome, intervention or comparison than the claim, or it lacks the specific measurement the claim \
needs. It must not report the opposite of the claim either.",
}
RELATION = {"SUPPORT": "support", "CONTRADICT": "contradict"}


def evidence_ids(sample):
    return [(a.doc_id, i) for a in sample.articles for i, t in enumerate(a.evidence_targets) if t]


def abstracts_text(sample, marked):
    parts = []
    for a in sample.articles:
        parts.append(f"[Article {a.doc_id}] {a.title}")
        parts += [f"{'>>' if (a.doc_id, i) in marked else '  '} [{a.doc_id}-{i}] {s.strip()}"
                  for i, s in enumerate(a.sentences) if s]
        parts.append("")
    return "\n".join(parts)


def edit_prompt(sample, kind):
    if kind == "claim_negation":
        text = CLAIM_NEGATION.format(claim=sample.claim)
        out_schema = {"type": "object", "properties": {"claim": {"type": "string"}},
                      "required": ["claim"], "additionalProperties": False}
    else:
        ids = evidence_ids(sample)
        goal = GOALS[kind].format(flipped=RELATION[FLIP[sample.label]] + "s")
        text = EVIDENCE_EDIT.format(relation=RELATION[sample.label], claim=sample.claim,
                                    abstracts=abstracts_text(sample, set(ids)), goal=goal)
        keys = [f"{d}-{i}" for d, i in ids]
        every = [f"{a.doc_id}-{i}" for a in sample.articles for i, t in enumerate(a.sentences) if t]
        out_schema = {"type": "object", "properties": {
            "sentences": {"type": "object", "properties": {k: {"type": "string"} for k in every},
                          "required": keys, "additionalProperties": False},
            "titles": {"type": "object", "properties": {str(a.doc_id): {"type": "string"} for a in sample.articles},
                       "additionalProperties": False}},
            "required": ["sentences", "titles"], "additionalProperties": False}
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], out_schema


def similar(old, new):
    """A minimal edit: most of the text survives."""
    return difflib.SequenceMatcher(None, old.strip().lower(), new.strip().lower()).ratio() >= 0.5


def apply_edit(sample, kind, parsed):
    """The edited Sample (same claim_id; variant = edit type), or None if the edit is unusable."""
    from dataclasses import replace

    if kind == "claim_negation":
        claim = parsed["claim"].strip()
        if not claim or claim.lower() == sample.claim.strip().lower():
            return None
        if not similar(sample.claim, claim):
            return None
        return replace(sample, claim=claim, label=FLIP[sample.label], variant=kind)
    edits, titles = parsed["sentences"], parsed.get("titles", {})
    articles = []
    for a in sample.articles:
        sentences, targets = list(a.sentences), list(a.evidence_targets)
        for i, old in enumerate(a.sentences):
            new = edits.get(f"{a.doc_id}-{i}", "").strip()
            if a.evidence_targets[i] and not new:
                return None
            if not new or not old or new == old.strip():
                continue
            if not similar(old, new):
                return None
            sentences[i] = new
            targets[i] = 1  # gold evidence and every sentence restating the edited finding
        if kind == "evidence_weaken":
            targets = [0] * len(targets)
        title = titles.get(str(a.doc_id), "").strip() or a.title
        if not similar(a.title, title):
            return None
        articles.append(replace(a, title=title, sentences=sentences, evidence_targets=targets))
    if not any(x.sentences[i] != y.sentences[i] for x, y in zip(articles, sample.articles)
               for i, t in enumerate(y.evidence_targets) if t):
        return None
    label = FLIP[sample.label] if kind == "evidence_flip" else "NEI"
    return replace(sample, articles=articles, label=label, variant=kind)


def to_record(sample, kind, edited, judge_original, judge_edited):
    changed, titles, evidence = {}, {}, []
    if edited is not None and kind != "claim_negation":
        for a, o in zip(edited.articles, sample.articles):
            changed |= {f"{a.doc_id}-{i}": s for i, (s, old) in enumerate(zip(a.sentences, o.sentences)) if s != old}
            if a.title != o.title:
                titles[str(a.doc_id)] = a.title
            evidence += [f"{a.doc_id}-{i}" for i, t in enumerate(a.evidence_targets) if t]
    return {
        "claim_id": sample.claim_id, "edit": kind, "original_label": sample.label,
        "original_claim": sample.claim,
        "claim": edited.claim if edited else None,
        "label": edited.label if edited else None,
        "edited_sentences": changed, "edited_titles": titles, "evidence": evidence,
        "judge_original": judge_original, "judge_edited": judge_edited,
        "verified": bool(edited) and judge_original == sample.label and judge_edited == edited.label,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, help="first N claims only (smoke test)")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    samples = load_samples(TRAIN_DIR if args.split == "train" else TEST_DIR)
    samples = [s for s in samples if s.variant == "original" and s.label != "NEI"][:args.limit]
    llm = LLM(model=args.model, max_model_len=16384, gpu_memory_utilization=0.90, seed=0,
              max_num_seqs=128, limit_mm_per_prompt={"image": 0, "video": 0})
    chat = {"chat_template_kwargs": {"enable_thinking": False}}

    jobs = [(s, kind) for s in samples for kind in EDIT_TYPES]
    prompts, params = [], []
    for s, kind in jobs:
        prompt, out_schema = edit_prompt(s, kind)
        prompts.append(prompt)
        params.append(SamplingParams(temperature=0.0, max_tokens=2048,
                                     structured_outputs=StructuredOutputsParams(json=out_schema)))
    outputs = llm.chat(prompts, params, **chat)
    edited = []
    for (s, kind), output in zip(jobs, outputs):
        try:
            edited.append(apply_edit(s, kind, json.loads(output.outputs[0].text)))
        except (json.JSONDecodeError, KeyError):
            edited.append(None)

    # Judge: the B5 zero-shot verifier on every original and every usable edit.
    to_judge = samples + [e for e in edited if e is not None]
    judge_params = [SamplingParams(temperature=0.0, max_tokens=512, logprobs=20,
                                   structured_outputs=StructuredOutputsParams(json=schema(x))) for x in to_judge]
    verdicts = llm.chat([build_prompt(x) for x in to_judge], judge_params, **chat)
    labels = [json.loads(v.outputs[0].text)["label"] for v in verdicts]
    original_verdict = dict(zip((s.claim_id for s in samples), labels[:len(samples)]))
    edited_verdicts = iter(labels[len(samples):])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.split}{'.limit' if args.limit else ''}.jsonl"
    counts = {}
    with out.open("w") as handle:
        for (s, kind), e in zip(jobs, edited):
            record = to_record(s, kind, e, original_verdict[s.claim_id], next(edited_verdicts) if e else None)
            counts.setdefault(kind, [0, 0, 0])
            counts[kind][0] += 1
            counts[kind][1] += e is not None
            counts[kind][2] += record["verified"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    for kind, (n, usable, verified) in counts.items():
        print(f"{kind:16s} claims={n} usable={usable} verified={verified}")


if __name__ == "__main__":
    main()
