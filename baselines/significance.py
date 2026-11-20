"""Seed ensembles and paired bootstrap tests between d1 model families, from saved predictions.

    python significance.py [--resamples 1000]

For each family (all seeds of one training recipe) it reports the seed-averaged metric and the
probability-averaged ensemble on: test macro-F1 and ECE, deletion selectivity (NEI rate on
evidence_removed minus random_removed) and counterfactual pair accuracy (original and edit both
correct). Differences between families get a 95% CI from a paired bootstrap over test claims (every
eval set is resampled by the same claims). Writes results/significance.md.
"""

import argparse
import json
import random

from common import RESULTS_DIR
from LiquidAI.config import LABELS

FAMILIES = {
    "B2 original-data SFT": [f"b2_d1_original_sft_seed{s}" for s in (42, 43, 44)],
    "augmented SFT": ["d1_augmented_sft_seed42", "d1_augmented_sft_seed43"],
    "+ all counterfactuals": ["d1_cf_all_seed42", "d1_cf_all_seed43"],
    "+ evidence-side counterfactuals": [f"d1_cf_evidence_only_seed{s}" for s in (42, 43, 44)],
}
COMPARISONS = [("+ evidence-side counterfactuals", "augmented SFT"),
               ("+ evidence-side counterfactuals", "B2 original-data SFT"),
               ("augmented SFT", "B2 original-data SFT"),
               ("+ all counterfactuals", "augmented SFT")]
SETS = ("test", "evidence_removed", "random_removed", "cf_evidence_flip", "cf_evidence_weaken")
METRICS = ("macro_f1", "selectivity", "flip_pair", "weaken_pair")


def load(name, eval_set):
    path = RESULTS_DIR / ("" if eval_set == "test" else eval_set) / name / "predictions.jsonl"
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["claim_id"], []).append(
            (r["gold_label"], [r["label_probabilities"][label] for label in LABELS]))
    return rows


def ensemble(runs):
    """Average label probabilities over seeds (same samples in the same order)."""
    out = {}
    for claim, rows in runs[0].items():
        out[claim] = [(gold, [sum(run[claim][i][1][k] for run in runs) / len(runs) for k in range(len(LABELS))])
                      for i, (gold, _) in enumerate(rows)]
    return out


def predicted(probs):
    return LABELS[max(range(len(LABELS)), key=probs.__getitem__)]


def macro_f1(pairs):
    scores = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in pairs)
        wrong = sum((g == label) != (p == label) for g, p in pairs)
        scores.append(2 * tp / (2 * tp + wrong) if tp else 0.0)
    return sum(scores) / len(scores)


def ece(rows, bins=10):
    buckets = [[] for _ in range(bins)]
    for gold, probs in rows:
        confidence = max(probs)
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, predicted(probs) == gold))
    return sum(len(b) / len(rows) * abs(sum(c for c, _ in b) / len(b) - sum(a for _, a in b) / len(b))
               for b in buckets if b)


def metrics(run, claims):
    """run: eval set -> claim -> [(gold, probs)]; claims may repeat (bootstrap)."""
    test = [(g, predicted(p)) for c in claims for g, p in run["test"][c]]
    original_ok = {c: all(g == predicted(p) for g, p in run["test"][c]) for c in set(claims)}

    def nei_rate(eval_set):
        rows = [predicted(p) for c in claims for _, p in run[eval_set].get(c, [])]
        return sum(p == "NEI" for p in rows) / max(len(rows), 1)

    def pair(eval_set):
        rows = [original_ok[c] and g == predicted(p) for c in claims for g, p in run[eval_set].get(c, [])]
        return sum(rows) / max(len(rows), 1)

    return {"macro_f1": macro_f1(test), "selectivity": nei_rate("evidence_removed") - nei_rate("random_removed"),
            "flip_pair": pair("cf_evidence_flip"), "weaken_pair": pair("cf_evidence_weaken")}


def family_metrics(runs, claims):
    per_seed = [metrics(run, claims) for run in runs]
    return {k: sum(m[k] for m in per_seed) / len(per_seed) for k in METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    runs = {f: [{s: load(n, s) for s in SETS} for n in names] for f, names in FAMILIES.items()}
    claims = sorted(next(iter(runs.values()))[0]["test"])
    lines = ["# Seed ensembles and paired bootstrap", "",
             "| family | seeds | macro-F1 (mean / ensemble) | ECE (mean / ensemble) | selectivity | "
             "evidence_flip pair | evidence_weaken pair |", "|---|---|---|---|---|---|---|"]
    base = {}
    for family, seeds in runs.items():
        base[family] = family_metrics(seeds, claims)
        merged = {s: ensemble([r[s] for r in seeds]) for s in SETS}
        ens = metrics(merged, claims)
        mean_ece = sum(ece([x for rows in r["test"].values() for x in rows]) for r in seeds) / len(seeds)
        ens_ece = ece([x for rows in merged["test"].values() for x in rows])
        m = base[family]
        lines.append(f"| {family} | {len(seeds)} | {m['macro_f1']:.3f} / {ens['macro_f1']:.3f} | "
                     f"{mean_ece:.3f} / {ens_ece:.3f} | {m['selectivity']:+.3f} / {ens['selectivity']:+.3f} | "
                     f"{m['flip_pair']:.3f} / {ens['flip_pair']:.3f} | {m['weaken_pair']:.3f} / {ens['weaken_pair']:.3f} |")
    rng = random.Random(0)
    boots = []
    for _ in range(args.resamples):
        sample = [rng.choice(claims) for _ in claims]
        boots.append({f: family_metrics(seeds, sample) for f, seeds in runs.items()})
    lo, hi = int(0.025 * args.resamples), int(0.975 * args.resamples) - 1
    lines += ["", f"Difference of seed-averaged metrics, 95% CI from {args.resamples} paired bootstrap "
              "resamples of the test claims.", "",
              "| comparison | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for a, b in COMPARISONS:
        cells = []
        for k in METRICS:
            d = sorted(x[a][k] - x[b][k] for x in boots)
            cells.append(f"{base[a][k] - base[b][k]:+.3f} [{d[lo]:+.3f}, {d[hi]:+.3f}]")
        lines.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "significance.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
