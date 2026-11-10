"""Gradio demo: evidence-sensitive scientific claim verification with d1-3B.

    CUDA_VISIBLE_DEVICES=4 python demo/app.py \
        --adapter "Original-data SFT=/scratch/.../d1-original-sft-seed42/final" \
        --adapter "Augmented SFT=/path/to/augmented/final" \
        [--llm-url http://localhost:8000/v1] [--share]

Tabs
  Verify a claim  pick a curated or any test claim, or paste your own; see the label distribution,
                  sentence evidence scores, gold annotations; remove sentences and re-verify.
  Batch mode      ~20 preloaded claims, per-claim latency, optionally against a generative LLM.
  About           baseline results, intended use, limitations.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import random
import re
import statistics
import time

import gradio as gr

from engine import ROOT, ZERO_SHOT, D1Verifier, split_sentences

from LiquidAI.config import LABELS
from LiquidAI.data import load_samples

RESULTS = ROOT / "baselines" / "results"
EXAMPLES_FILE = Path(__file__).with_name("examples.json")
LABEL_COLORS = {"SUPPORT": "34,160,90", "CONTRADICT": "214,60,60", "NEI": "130,130,130"}
LABEL_TEXT = {
    "SUPPORT": "The cited evidence **supports** the claim.",
    "CONTRADICT": "The cited evidence **contradicts** the claim.",
    "NEI": "**Not enough information**: the provided text does not establish or refute the claim.",
}
DISCLAIMER = ("This is a research prototype and a reading aid, not a substitute for reading the abstract or for "
              "expert judgement. Probabilities are model scores and can be confidently wrong.")
HEADER = re.compile(r"^##\s*(?:\[doc\s+(\d+)\])?\s*(.*)$")


# --------------------------------------------------------------------------- data

def load_test():
    return {s.claim_id: s for s in load_samples(ROOT / "data" / "test")}


def articles_to_text(articles) -> str:
    blocks = []
    for a in articles:
        lines = [f"## [doc {a['doc_id']}] {a['title']}"]
        lines += [s.strip() for s in a["sentences"] if s.strip()]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def text_to_articles(text: str) -> list[dict]:
    """Parse the editable abstract box: '## [doc ID] title' headers, then one sentence per line."""
    articles, current = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = HEADER.match(line)
        if match:
            current = {"doc_id": int(match.group(1) or len(articles)), "title": match.group(2).strip(), "lines": []}
            articles.append(current)
            continue
        if current is None:
            current = {"doc_id": 0, "title": "", "lines": []}
            articles.append(current)
        current["lines"].append(line)
    out = []
    for a in articles:
        sentences = [s for line in a["lines"] for s in (split_sentences(line) if len(line) > 400 else [line])]
        if sentences:
            out.append({"doc_id": a["doc_id"], "title": a["title"], "sentences": sentences})
    return out


def sample_articles(sample):
    return [{"doc_id": a.doc_id, "title": a.title, "sentences": [s.strip() for s in a.sentences]}
            for a in sample.articles]


def gold_of(sample):
    return sample.label, {(a.doc_id, i) for a in sample.articles for i, t in enumerate(a.evidence_targets) if t}


# --------------------------------------------------------------------------- rendering

def render_sentences(articles, verdict, removed, gold_evidence) -> str:
    rgb = LABEL_COLORS[verdict.label]
    parts = ['<div style="line-height:1.55;font-size:15px">']
    for a in articles:
        parts.append(f'<div style="margin:10px 0 4px;font-weight:600">[doc {a["doc_id"]}] {html.escape(a["title"])}</div>')
        for i, sentence in enumerate(a["sentences"]):
            key = (a["doc_id"], i)
            star = ' <span title="gold evidence" style="color:#d4a017">&#9733;</span>' if key in gold_evidence else ""
            if key in removed:
                parts.append(f'<div style="opacity:.45;text-decoration:line-through;padding:2px 6px">'
                             f'<code>{i}</code> {html.escape(sentence)} <i>(removed)</i>{star}</div>')
                continue
            p = verdict.sentence_probabilities.get(key, 0.0)
            selected = p >= 0.5
            alpha = p * (0.55 if verdict.label != "NEI" else 0.25)
            border = f"border-left:4px solid rgb({rgb});" if selected and verdict.label != "NEI" else "border-left:4px solid transparent;"
            weight = "font-weight:600;" if selected and verdict.label != "NEI" else ""
            parts.append(f'<div style="background:rgba({rgb},{alpha:.3f});{border}{weight}padding:2px 6px;margin:1px 0;border-radius:3px">'
                         f'<code>{i}</code> {html.escape(sentence)} '
                         f'<span style="font-size:12px;opacity:.75">[p={p:.2f}]</span>{star}</div>')
    parts.append("</div>")
    return "".join(parts)


def verdict_markdown(verdict, model, gold_label, edited, n_removed) -> str:
    lines = [f"### {verdict.label}  ({verdict.label_probabilities[verdict.label]:.0%})", LABEL_TEXT[verdict.label]]
    if verdict.label == "NEI":
        lines.append("_Sentence scores are shown faintly for transparency; no sentence is claimed as evidence._")
    if gold_label and not edited:
        mark = "correct" if gold_label == verdict.label else "**incorrect**"
        lines.append(f"Gold label: **{gold_label}** ({mark}). &#9733; = annotated gold evidence sentence.")
    elif gold_label:
        lines.append("_Input was edited or sentences were removed, so the gold label no longer applies._")
    lines.append(f"Model: {model} &middot; {n_removed} sentence(s) removed &middot; "
                 f"{verdict.seconds * 1000:.0f} ms &middot; {verdict.input_tokens} tokens read")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- app

def build_app(engine: D1Verifier, test, examples, batch_ids, llm_url):
    example_choices = [f"[{e['tag']}] #{e['claim_id']}: {test[e['claim_id']].claim}" for e in examples]
    example_ids = dict(zip(example_choices, (e["claim_id"] for e in examples)))
    example_notes = {e["claim_id"]: e.get("note", "") for e in examples}
    all_choices = [f"#{cid} [{s.label}] {s.claim}" for cid, s in sorted(test.items())]

    def sentence_choices(articles):
        return [f"{a['doc_id']}-{i}: {s[:80]}" for a in articles for i, s in enumerate(a["sentences"])]

    def load_claim(claim_id):
        sample = test[claim_id]
        articles = sample_articles(sample)
        text = articles_to_text(articles)
        note = example_notes.get(claim_id, "")
        info = f"Loaded test claim #{claim_id} (gold: **{sample.label}**)." + (f" _{note}_" if note else "")
        return (sample.claim, text, gr.update(choices=sentence_choices(articles), value=[]),
                {"claim_id": claim_id, "claim": sample.claim, "text": text}, info)

    def on_example(choice):
        return load_claim(example_ids[choice]) if choice else (gr.skip(),) * 5

    def on_any(choice):
        return load_claim(int(choice.split()[0][1:])) if choice else (gr.skip(),) * 5

    def on_random():
        return load_claim(random.choice(list(test)))

    def on_text_change(text):
        return gr.update(choices=sentence_choices(text_to_articles(text)), value=[])

    def removed_keys(selection):
        return {(int(item.split(":")[0].split("-")[0]), int(item.split(":")[0].split("-")[1])) for item in selection or []}

    def run(claim, text, model, selection, loaded):
        claim = (claim or "").strip()
        articles = text_to_articles(text or "")
        if not claim or not articles:
            raise gr.Error("Enter a claim and at least one abstract sentence.")
        removed = removed_keys(selection)
        masked = [{**a, "sentences": [("" if (a["doc_id"], i) in removed else s) for i, s in enumerate(a["sentences"])]}
                  for a in articles]
        verdict = engine.verify(claim, masked, model)
        gold_label, gold_evidence = None, set()
        edited = True
        if loaded and loaded.get("claim_id") in test:
            gold_label, gold_evidence = gold_of(test[loaded["claim_id"]])
            edited = claim != loaded["claim"] or text != loaded["text"] or bool(removed)
        probs = {label: verdict.label_probabilities[label] for label in LABELS}
        return (probs, verdict_markdown(verdict, model, gold_label, edited, len(removed)),
                render_sentences(articles, verdict, removed, gold_evidence if not (claim != (loaded or {}).get("claim") or text != (loaded or {}).get("text")) else set()),
                {"evidence": [f"{d}-{i}" for (d, i), p in verdict.sentence_probabilities.items() if p >= 0.5]})

    def remove_evidence(claim, text, model, selection, loaded, last):
        """Remove every sentence the model currently selects as evidence, then re-verify."""
        articles = text_to_articles(text or "")
        lookup = {f"{a['doc_id']}-{i}": f"{a['doc_id']}-{i}: {s[:80]}" for a in articles for i, s in enumerate(a["sentences"])}
        selection = sorted(set(selection or []) | {lookup[k] for k in (last or {}).get("evidence", []) if k in lookup})
        return (gr.update(value=selection), *run(claim, text, model, selection, loaded))

    def compare(claim, text, selection):
        claim = (claim or "").strip()
        articles = text_to_articles(text or "")
        if not claim or not articles:
            raise gr.Error("Enter a claim and at least one abstract sentence.")
        removed = removed_keys(selection)
        masked = [{**a, "sentences": [("" if (a["doc_id"], i) in removed else s) for i, s in enumerate(a["sentences"])]}
                  for a in articles]
        rows = []
        for name in engine.model_names:
            v = engine.verify(claim, masked, name)
            evidence = [f"{d}-{i}" for (d, i), p in v.sentence_probabilities.items() if p >= 0.5]
            rows.append([name, v.label, *(round(v.label_probabilities[l], 3) for l in LABELS),
                         ", ".join(evidence) or "-", round(v.seconds * 1000)])
        return rows

    def run_batch(model, use_llm):
        rows, d1_times, llm_times = [], [], []
        correct = llm_correct = 0
        headers = ["claim", "gold", "d1 label", "d1 conf", "d1 ms"]
        recorded = None
        if use_llm and not llm_url:
            recorded = load_recorded_llm()
        if use_llm:
            headers += ["LLM label", "LLM ms"]
        for n, cid in enumerate(batch_ids, start=1):
            sample = test[cid]
            v = engine.verify(sample.claim, sample_articles(sample), model)
            d1_times.append(v.seconds)
            correct += v.label == sample.label
            row = [f"#{cid} {sample.claim[:70]}", sample.label, v.label,
                   round(v.label_probabilities[v.label], 2), round(v.seconds * 1000)]
            if use_llm and llm_url:
                label, seconds = call_llm(llm_url, sample)
                llm_times.append(seconds)
                llm_correct += label == sample.label
                row += [label, round(seconds * 1000)]
            elif use_llm and recorded:
                label = recorded["labels"].get(cid, "?")
                llm_correct += label == sample.label
                row += [label, "recorded"]
            rows.append(row)
            summary = (f"**{model}**: {correct}/{n} correct, total {sum(d1_times):.2f} s, "
                       f"median {statistics.median(d1_times) * 1000:.0f} ms/claim")
            if use_llm and llm_times:
                summary += (f"  \n**Qwen3.8-27B (live)**: {llm_correct}/{n} correct, total {sum(llm_times):.2f} s, "
                            f"median {statistics.median(llm_times) * 1000:.0f} ms/claim")
            elif use_llm and recorded:
                summary += (f"  \n**Qwen3.8-27B (recorded run, same GPU type)**: {llm_correct}/{n} correct, "
                            f"median {recorded['median_seconds'] * 1000:.0f} ms/claim at batch size 1")
            yield gr.update(value=rows, headers=headers), summary

    with gr.Blocks(title="Evidence-Sensitive Claim Verification") as app:
        gr.Markdown("# Evidence-sensitive scientific claim verification\n"
                    "Given a claim and cited abstract(s), d1-3B predicts **SUPPORT / CONTRADICT / NOT ENOUGH INFO** "
                    "in one forward pass and scores every sentence as evidence.  \n"
                    f"_{DISCLAIMER}_")
        loaded = gr.State(None)
        last = gr.State(None)
        with gr.Tab("Verify a claim"):
            with gr.Row():
                example = gr.Dropdown(example_choices, label="Curated examples (representative / challenging / failure)", scale=3)
                any_claim = gr.Dropdown(all_choices, label="Any test claim", scale=3)
                random_button = gr.Button("Random test claim", scale=1)
            info = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    claim = gr.Textbox(label="Claim", lines=2)
                    text = gr.Textbox(label="Cited abstract(s): '## [doc ID] title' header, then one sentence per line",
                                      lines=14)
                    model = gr.Radio(engine.model_names, value=engine.model_names[-1], label="Model")
                    removed = gr.CheckboxGroup([], label="Remove sentences (they become unavailable text, as in our augmentation)")
                    with gr.Row():
                        verify = gr.Button("Verify", variant="primary")
                        strip = gr.Button("Remove predicted evidence & re-verify")
                        compare_button = gr.Button("Compare all models")
                with gr.Column(scale=1):
                    probs = gr.Label(label="Label probabilities", num_top_classes=3)
                    summary = gr.Markdown()
                    sentences = gr.HTML()
                    comparison = gr.Dataframe(headers=["model", "label", *(f"P({l})" for l in LABELS), "evidence (p>=0.5)", "ms"],
                                              label="All models on the current input", interactive=False)
        with gr.Tab("Batch mode"):
            gr.Markdown(f"Verify {len(batch_ids)} preloaded test claims one at a time and compare per-claim latency.")
            with gr.Row():
                batch_model = gr.Radio(engine.model_names, value=engine.model_names[-1], label="d1 model")
                use_llm = gr.Checkbox(value=True, label="Compare with generative LLM (Qwen3.8-27B" +
                                      (", live)" if llm_url else ", recorded run)"))
                batch_button = gr.Button("Run batch", variant="primary")
            batch_summary = gr.Markdown()
            batch_table = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("About"):
            gr.Markdown(about_markdown())

        outputs = [claim, text, removed, loaded, info]
        example.change(on_example, example, outputs)
        any_claim.change(on_any, any_claim, outputs)
        random_button.click(on_random, None, outputs)
        text.input(on_text_change, text, removed)
        verify.click(run, [claim, text, model, removed, loaded], [probs, summary, sentences, last])
        strip.click(remove_evidence, [claim, text, model, removed, loaded, last], [removed, probs, summary, sentences, last])
        compare_button.click(compare, [claim, text, removed], comparison)
        batch_button.click(run_batch, [batch_model, use_llm], [batch_table, batch_summary])
    return app


# --------------------------------------------------------------------------- LLM comparison

def call_llm(url, sample):
    """One claim through a vLLM OpenAI-compatible server, same prompt as baselines/llm_zero_shot.py."""
    import sys
    import urllib.request

    sys.path.insert(0, str(ROOT / "baselines"))
    from llm_zero_shot import MODEL, build_prompt, schema

    body = {"model": MODEL, "messages": build_prompt(sample), "temperature": 0.0, "max_tokens": 512,
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": schema(sample)}},
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        content = json.load(response)["choices"][0]["message"]["content"]
    seconds = time.perf_counter() - started
    try:
        return json.loads(content)["label"], seconds
    except (ValueError, KeyError):
        return "?", seconds


def load_recorded_llm():
    folder = RESULTS / "b5_llm_qwen3.8_27b"
    if not (folder / "metrics.json").exists():
        return None
    metrics = json.loads((folder / "metrics.json").read_text())
    labels = {}
    for line in (folder / "predictions.jsonl").read_text().splitlines():
        record = json.loads(line)
        labels[record["claim_id"]] = record["predicted_label"]
    return {"labels": labels, "median_seconds": metrics["latency"]["median_seconds_per_claim"]}


# --------------------------------------------------------------------------- about tab

def about_markdown() -> str:
    rows = []
    for path in sorted(RESULTS.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        rows.append(f"| {path.parent.name} | {m['label_macro_f1']:.3f} | {m['label_accuracy']:.3f} | "
                    f"{m['sentence_level_evidence_f1']:.3f} | {m['label_ece']:.3f} | "
                    f"{m['latency']['median_seconds_per_claim'] * 1000:.0f} |")
    table = ("| system | label macro-F1 | accuracy | evidence F1 | ECE | median ms/claim |\n|---|---|---|---|---|---|\n"
             + "\n".join(rows)) if rows else "_No baseline results found._"
    return f"""## What the system does
Input: a claim and one or more cited abstracts, split into sentences. Output: one of SUPPORT, CONTRADICT,
NOT ENOUGH INFO with calibrated probabilities, and P(evidence) for every sentence (selected at p >= 0.5).
d1-3B (LiquidAI) answers one 3-way question and one yes/no question per sentence, without generating text.

## Models
* **d1-3B zero-shot**: no fine-tuning.
* **Original-data SFT**: LoRA on the unmodified training claims.
* **Augmented SFT**: LoRA on the training claims plus evidence-removed copies (label becomes NEI when all
  evidence groups are removed), our proposed method.

## Test-set results (333 held-out claims)
{table}

## Intended use and limitations
{DISCLAIMER}
* Trained and evaluated on SciFact biomedical abstracts only; other domains are untested.
* Gold rationales can miss valid evidence, so a removed-evidence input is not guaranteed to be truly NEI.
* The demo reads the abstract once for all questions (d1's tree pass). Probabilities can differ from the
  evaluation run by bf16 rounding (max 0.03 label / 0.06 evidence on 30 checked claims, no label flips).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", action="append", default=[], help="NAME=PATH of a LoRA adapter directory")
    parser.add_argument("--llm-url", help="vLLM OpenAI-compatible base URL for live LLM comparison")
    parser.add_argument("--batch-size", type=int, default=20, help="claims in batch mode")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    adapters = {}
    for item in args.adapter:
        name, path = item.split("=", 1)
        if (Path(path) / "adapter_config.json").is_file():
            adapters[name] = path
        else:
            print(f"skipping adapter {name!r}: no adapter_config.json in {path}")
    engine = D1Verifier(adapters)
    test = load_test()
    examples = json.loads(EXAMPLES_FILE.read_text()) if EXAMPLES_FILE.exists() else [
        {"claim_id": cid, "tag": "test"} for cid in sorted(test)[:10]]
    rng = random.Random(0)
    by_label = {label: sorted(c for c, s in test.items() if s.label == label) for label in LABELS}
    quota = {"SUPPORT": 8, "NEI": 8, "CONTRADICT": 4}
    scale = args.batch_size / sum(quota.values())
    batch_ids = sorted(c for label, n in quota.items() for c in rng.sample(by_label[label], max(1, round(n * scale))))
    engine.verify("warm-up", [{"doc_id": 0, "title": "", "sentences": ["warm-up sentence."]}])
    build_app(engine, test, examples, batch_ids, args.llm_url).launch(
        server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
