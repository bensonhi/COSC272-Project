"""Collect every results/*/metrics.json into one comparison table (results/summary.md + .json).

Adds, from predictions.jsonl:
  * 95% bootstrap CI of label macro-F1 (1000 claim resamples, seed 0)
  * threshold-free evidence metrics: average precision (AUPRC) over all available sentences,
    and top-1 hit rate (is the highest-scored sentence gold evidence, over claims that have any)
  * evidence F1 restricted to claims whose gold label is SUPPORT/CONTRADICT
  * evidence F1 against human rationales only: for the 249 test claims from SciFact train, the gold
    evidence also contains sentences added by Grok (label *_more); those sentences are neither
    rewarded nor penalised here
"""

import json
import random

from common import LABELS, RESULTS_DIR, TEST_DIR


def llm_added_evidence():
    """(claim_id, doc_id, original_index) of gold evidence sentences added by Grok, not annotators."""
    added = set()
    for path in TEST_DIR.glob("*.json"):
        claim = json.loads(path.read_text())
        added |= {(claim["id"], e["doc_id"], e["original_index"]) for e in claim["evidence"]
                  if e["evidence_label"].endswith("_more")}
    return added


LLM_ADDED = llm_added_evidence()


def macro_f1(pairs):
    scores = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in pairs)
        support = sum(g == label for g, _ in pairs)
        predicted = sum(p == label for _, p in pairs)
        scores.append(2 * tp / (support + predicted) if support + predicted else 0.0)
    return sum(scores) / len(scores)


def average_precision(scored):
    scored = sorted(scored, key=lambda x: -x[0])
    positives = sum(t for _, t in scored)
    hits, total = 0, 0.0
    for rank, (_, target) in enumerate(scored, start=1):
        if target:
            hits += 1
            total += hits / rank
    return total / positives if positives else 0.0


def diagnostics(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    pairs = [(r["gold_label"], r["predicted_label"]) for r in records]
    rng = random.Random(0)
    boot = sorted(macro_f1([rng.choice(pairs) for _ in pairs]) for _ in range(1000))
    scored, top1, top1_n, tp, fp, fn = [], 0, 0, 0, 0, 0
    human = [0, 0, 0]
    for r in records:
        for a in r["articles"]:
            for s in a["sentences"]:
                if not s["available"] or (r["claim_id"], a["doc_id"], s["original_index"]) in LLM_ADDED:
                    continue
                human[0] += s["gold_evidence"] and s["predicted_evidence"]
                human[1] += not s["gold_evidence"] and s["predicted_evidence"]
                human[2] += s["gold_evidence"] and not s["predicted_evidence"]
        rows = [s for a in r["articles"] for s in a["sentences"] if s["available"]]
        scored += [(s["evidence_probability"], s["gold_evidence"]) for s in rows]
        if any(s["gold_evidence"] for s in rows):
            top1_n += 1
            top1 += max(rows, key=lambda s: s["evidence_probability"])["gold_evidence"]
        if r["gold_label"] != "NEI":
            tp += sum(s["gold_evidence"] and s["predicted_evidence"] for s in rows)
            fp += sum(not s["gold_evidence"] and s["predicted_evidence"] for s in rows)
            fn += sum(s["gold_evidence"] and not s["predicted_evidence"] for s in rows)
    return {
        "macro_f1_ci95": [boot[25], boot[974]],
        "evidence_auprc": average_precision(scored),
        "evidence_top1_hit": top1 / top1_n,
        "evidence_f1_on_gold_evidence_claims": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "evidence_f1_human_rationales": 2 * human[0] / (2 * human[0] + human[1] + human[2]) if any(human) else 0.0,
    }


def main():
    rows = []
    for metrics_path in sorted(RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        extra = diagnostics(metrics_path.with_name("predictions.jsonl"))
        rows.append({
            "system": metrics_path.parent.name,
            "macro_f1": m["label_macro_f1"], **extra,
            "accuracy": m["label_accuracy"],
            **{f"f1_{label}": m["per_label"][label]["f1"] for label in LABELS},
            "evidence_f1": m["sentence_level_evidence_f1"],
            "evidence_precision": m["evidence"]["precision"], "evidence_recall": m["evidence"]["recall"],
            "ece": m["label_ece"],
            "median_ms_per_claim": m["latency"]["median_seconds_per_claim"] * 1000,
            "confusion_gold_rows_NEI_SUP_CON": m["label_confusion_matrix"],
        })
    (RESULTS_DIR / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    header = ("| system | macro-F1 [95% CI] | acc | F1 NEI / SUP / CON | evid. F1 (P / R) | evid. F1 human | "
              "evid. AUPRC | top-1 hit | ECE | median ms/claim |\n|---|---|---|---|---|---|---|---|---|---|")
    lines = [header]
    for r in rows:
        lo, hi = r["macro_f1_ci95"]
        lines.append(
            f"| {r['system']} | {r['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}] | {r['accuracy']:.3f} | "
            f"{r['f1_NEI']:.3f} / {r['f1_SUPPORT']:.3f} / {r['f1_CONTRADICT']:.3f} | "
            f"{r['evidence_f1']:.3f} ({r['evidence_precision']:.2f} / {r['evidence_recall']:.2f}) | "
            f"{r['evidence_f1_human_rationales']:.3f} | "
            f"{r['evidence_auprc']:.3f} | {r['evidence_top1_hit']:.3f} | {r['ece']:.3f} | {r['median_ms_per_claim']:.0f} |")
    text = "\n".join(lines) + "\n" + stress_table() + stress_table("_silent")
    (RESULTS_DIR / "summary.md").write_text(text)
    print(text)


def predictions(folder):
    path = folder / "predictions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else None


def stress_table(suffix=""):
    """Evidence-removal stress test on the 207 test claims that have gold evidence.

    suffix "_silent": the same deletions with sentences cut out and renumbered instead of blanked.
    """
    how = "cut out and the rest renumbered (no visible gap)" if suffix else "blanked (empty slots, the training format)"
    lines = ["", f"Stress test{' (silent deletion)' if suffix else ''} on the 207 test claims with gold evidence, deleted sentences {how} "
             "(evidence_removed: all gold evidence sentences, target NEI; "
             "random_removed: as many non-evidence sentences, label unchanged).", "",
             "| system | original acc | evidence_removed -> NEI | mean P(NEI) | random_removed acc | "
             "random_removed -> NEI | selectivity |", "|---|---|---|---|---|---|---|"]
    for folder in sorted((RESULTS_DIR / f"evidence_removed{suffix}").glob("*/")):
        removed = predictions(folder)
        control = predictions(RESULTS_DIR / f"random_removed{suffix}" / folder.name)
        original = predictions(RESULTS_DIR / folder.name)
        if not removed:
            continue
        ids = {r["claim_id"] for r in removed}
        nei_rate = sum(r["predicted_label"] == "NEI" for r in removed) / len(removed)
        p_nei = sum(r["label_probabilities"]["NEI"] for r in removed) / len(removed)
        cells = [folder.name, "-", f"{nei_rate:.3f}", f"{p_nei:.3f}", "-", "-", "-"]
        if original:
            same = [r for r in original if r["claim_id"] in ids]
            cells[1] = f"{sum(r['label_correct'] for r in same) / len(same):.3f}"
        if control:
            control_nei = sum(r["predicted_label"] == "NEI" for r in control) / len(control)
            cells[4] = f"{sum(r['label_correct'] for r in control) / len(control):.3f}"
            cells[5] = f"{control_nei:.3f}"
            cells[6] = f"{nei_rate - control_nei:+.3f}"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
