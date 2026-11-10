"""B5: zero-shot generative LLM (Qwen3.8-27B-FP8, non-thinking mode) via vLLM.

The prompt shows the claim and every cited abstract with sentence IDs "<doc_id>-<original_index>".
Output is constrained to JSON {"label": SUPPORT|CONTRADICT|NEI, "evidence": [sentence IDs]} by a
per-claim JSON schema whose evidence items are an enum of that claim's valid IDs. Decoding is
greedy. Label probabilities come from the model's next-token distribution at the first token of
the label value (top-20 logprobs, renormalized over the three labels); evidence probabilities are
1 for listed IDs and 0 otherwise (a generative model gives no per-sentence score).

All test claims are generated in one batch; single-claim latency (batch size 1, the setting
of the demo) is measured separately on the first --latency-claims claims.
Runs in the separate vLLM environment (vLLM pins its own torch).
"""

import argparse
import json
import math
import statistics
import time

from common import EVAL_SETS, LABELS, RESULTS_DIR, eval_samples, result_name, run_baseline, sentence_keys

MODEL = "Qwen/Qwen3.8-27B-FP8"
SYSTEM = "You are a careful scientific fact-checker."
INSTRUCTIONS = """Decide whether the cited abstracts, taken together, SUPPORT the claim, CONTRADICT it, \
or do not provide enough information to decide (NEI). Use only the abstracts, not background knowledge. \
Choose NEI when the abstracts are on topic but do not establish or refute the claim.
Then list the IDs of the sentences that justify a SUPPORT or CONTRADICT verdict; for NEI the list is empty.
Answer with JSON only: {"label": "SUPPORT" | "CONTRADICT" | "NEI", "evidence": ["<sentence id>", ...]}"""


def sentence_id(key):
    return f"{key[0]}-{key[1]}"


def build_prompt(sample):
    parts = [f"Claim: {sample.claim}", "", "Cited abstracts:"]
    for article in sample.articles:
        parts.append(f"\n[Article {article.doc_id}] {article.title}")
        parts += [f"[{article.doc_id}-{i}] {s.strip()}" for i, s in enumerate(article.sentences) if s]
    parts += ["", INSTRUCTIONS]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


def schema(sample):
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["SUPPORT", "CONTRADICT", "NEI"]},
            "evidence": {"type": "array", "items": {"type": "string",
                                                   "enum": [sentence_id(k) for k in sentence_keys(sample)]}},
        },
        "required": ["label", "evidence"],
        "additionalProperties": False,
    }


def label_distribution(completion):
    """P(label) from the top logprobs at the token where the label value starts."""
    tokens = [lp[tid] for tid, lp in zip(completion.token_ids, completion.logprobs)]
    text = "".join(t.decoded_token for t in tokens)
    label = json.loads(completion.text)["label"]
    start = text.index(label, text.index('"label"'))
    offset = 0
    for position, token in enumerate(tokens):
        if offset + len(token.decoded_token) > start:
            break
        offset += len(token.decoded_token)
    within = start - offset  # characters of this token that precede the label value
    chosen = tokens[position].decoded_token
    mass = dict.fromkeys(LABELS, 0.0)
    for candidate in completion.logprobs[position].values():
        piece = candidate.decoded_token
        if piece[:within] != chosen[:within]:
            continue
        piece = piece[within:].strip().strip('"')
        for name in LABELS:
            if piece and (name.startswith(piece) or piece.startswith(name)):
                mass[name] += math.exp(candidate.logprob)
                break
    total = sum(mass.values())
    if total == 0:  # should not happen; fall back to the decoded label
        return [1.0 if name == label else 0.0 for name in LABELS]
    floor = 1e-6
    probs = [max(mass[name] / total, floor) for name in LABELS]
    return [p / sum(probs) for p in probs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--latency-claims", type=int, default=30)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-set", choices=EVAL_SETS, default="test")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    samples = eval_samples(args.eval_set)
    name = result_name("b5_llm_qwen3.8_27b", args.eval_set)
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              seed=0, max_num_seqs=128, limit_mm_per_prompt={"image": 0, "video": 0})
    chat_kwargs = {"chat_template_kwargs": {"enable_thinking": False}}

    def params(sample):
        return SamplingParams(temperature=0.0, max_tokens=512, logprobs=20,
                              structured_outputs=StructuredOutputsParams(json=schema(sample)))

    prompts = [build_prompt(s) for s in samples]
    latencies = []
    for sample, prompt in zip(samples[:args.latency_claims], prompts):
        started = time.perf_counter()
        llm.chat([prompt], params(sample), use_tqdm=False, **chat_kwargs)
        latencies.append(time.perf_counter() - started)
    started = time.perf_counter()
    outputs = llm.chat(prompts, [params(s) for s in samples], **chat_kwargs)
    batch_seconds = time.perf_counter() - started

    cache, raw = {}, []
    for sample, output in zip(samples, outputs):
        completion = output.outputs[0]
        parsed = json.loads(completion.text)
        chosen = set(parsed["evidence"])
        cache[sample.key] = (label_distribution(completion),
                             {k: float(sentence_id(k) in chosen) for k in sentence_keys(sample)})
        raw.append({"sample_id": sample.key, "output": completion.text,
                    "prompt_tokens": len(output.prompt_token_ids), "output_tokens": len(completion.token_ids)})

    result = run_baseline(name, lambda s: cache[s.key], {
        "description": "zero-shot Qwen3.8-27B-FP8, non-thinking, greedy, JSON-schema constrained",
        "model": args.model, "system_prompt": SYSTEM, "instructions": INSTRUCTIONS,
        "label_probability": "top-20 logprobs at the label's first token, renormalized over labels",
        "evidence_probability": "1 if listed in the JSON evidence, else 0",
    }, samples=samples, overwrite=args.overwrite, latency={
        "median_seconds_per_claim": statistics.median(latencies),
        "mean_seconds_per_claim": statistics.fmean(latencies),
        "claims_timed": len(latencies),
        "batched_total_seconds": batch_seconds,
        "batched_seconds_per_claim": batch_seconds / len(samples),
        "note": "median/mean: one claim per call (batch size 1); batched_*: whole test set in one call",
    })
    out = RESULTS_DIR / name / "raw_outputs.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
