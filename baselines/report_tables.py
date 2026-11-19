"""Numbers for the report's Results and Analysis section, from saved predictions (no GPU).

    python report_tables.py        -> results/report_tables.md

For every baseline, the three final d1 runs and the three-seed reruns (seeds 42-44, identical
hyperparameters; mean +- sample standard deviation): test accuracy / macro-F1 / evidence F1 / ECE;
the deletion stress tests (NEI rate with all gold evidence removed, with as many other sentences
removed, and accuracy with all but one evidence group removed), with blank slots and silent
deletion; label error counts on the test set; and paired-bootstrap 95% CIs of seed-averaged
differences between the training recipes.
"""

import json
import random
import statistics

from common import RESULTS_DIR

BASELINES = {"d1-3B zero-shot": "b1_d1_zero_shot", "NLI DeBERTa-v3": "b3_nli_zero_shot_abstract",
             "MultiVerS (fine-tuned)": "b4b_multivers_finetuned", "Qwen3.8-27B": "b5_llm_qwen3.8_27b"}
FINAL = {"Original SFT": "final_d1_original", "Augmented SFT": "final_d1_augment", "Paired hinge": "final_d1_augment_pair"}
SEEDS = {"Original SFT": [f"v2_d1_original_seed{s}" for s in (42, 43, 44)],
         "Augmented SFT": [f"v2_d1_augmented_seed{s}" for s in (42, 43, 44)],
         "Paired hinge": [f"v2_d1_pair_seed{s}" for s in (42, 43, 44)]}
LABELS = ("NEI", "SUPPORT", "CONTRADICT")


def records(name, eval_set="test"):
    folder = RESULTS_DIR / ("" if eval_set == "test" else eval_set) / name
    return {r["claim_id"]: r for r in map(json.loads, (folder / "predictions.jsonl").read_text().splitlines())}


def metrics(name, eval_set="test"):
    folder = RESULTS_DIR / ("" if eval_set == "test" else eval_set) / name
    return json.loads((folder / "metrics.json").read_text())


def nei_rate(rows):
    return 100 * sum(r["predicted_label"] == "NEI" for r in rows) / len(rows)


def accuracy(rows):
    return 100 * sum(r["predicted_label"] == r["gold_label"] for r in rows) / len(rows)


def macro_f1(rows):
    scores = []
    for label in LABELS:
        tp = sum(r["gold_label"] == label and r["predicted_label"] == label for r in rows)
        wrong = sum((r["gold_label"] == label) != (r["predicted_label"] == label) for r in rows)
        scores.append(2 * tp / (2 * tp + wrong) if tp else 0.0)
    return 100 * sum(scores) / 3


def row(name):
    m = metrics(name)
    test = list(records(name).values())
    out = {"acc": 100 * m["label_accuracy"], "f1": 100 * m["label_macro_f1"],
           "evf1": 100 * m["sentence_level_evidence_f1"], "ece": 100 * m["label_ece"],
           "ms": 1000 * m["latency"]["median_seconds_per_claim"],
           "unsupported": sum(r["gold_label"] == "NEI" and r["predicted_label"] != "NEI" for r in test),
           "abstain": sum(r["gold_label"] != "NEI" and r["predicted_label"] == "NEI" for r in test),
           "polarity": sum({r["gold_label"], r["predicted_label"]} == {"SUPPORT", "CONTRADICT"} for r in test)}
    for suffix in ("", "_silent"):
        out["gold" + suffix] = nei_rate(records(name, "evidence_removed" + suffix).values())
        out["control" + suffix] = nei_rate(records(name, "random_removed" + suffix).values())
        partial = list(records(name, "partial_removed" + suffix).values())
        out["partial" + suffix] = accuracy(partial)
        out["partial_nei" + suffix] = nei_rate(partial)
    return out


COLUMNS = [("acc", "Acc."), ("f1", "Macro-F1"), ("evf1", "Evid. F1"), ("ece", "ECE"), ("gold", "Gold removed: NEI%"),
           ("control", "Control: NEI%"), ("partial", "Partial: acc."), ("partial_nei", "Partial: NEI%"),
           ("gold_silent", "Silent gold: NEI%"), ("control_silent", "Silent control: NEI%"),
           ("partial_silent", "Silent partial: acc."), ("unsupported", "Unsupported"), ("abstain", "Unnecessary abstention"),
           ("polarity", "Polarity"), ("ms", "ms/claim")]


def fmt(values):
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"


def bootstrap(a_runs, b_runs, statistic, eval_set, resamples=2000):
    """95% CI of mean_a - mean_b (seed-averaged), resampling claims jointly."""
    a = [records(n, eval_set) for n in a_runs]
    b = [records(n, eval_set) for n in b_runs]
    claims = sorted(a[0])
    def diff(sample):
        mean = lambda runs: statistics.mean(statistic([r[c] for c in sample]) for r in runs)
        return mean(a) - mean(b)
    rng = random.Random(0)
    boots = sorted(diff([rng.choice(claims) for _ in claims]) for _ in range(resamples))
    return diff(claims), boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]


def main():
    lines = ["# Report tables", "", "| system | " + " | ".join(h for _, h in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    groups = [(f"{k}", [v]) for k, v in BASELINES.items()] + [(f"{k} (final run)", [v]) for k, v in FINAL.items()] + \
             [(f"{k} (3 seeds)", v) for k, v in SEEDS.items()]
    for label, runs in groups:
        rows = [row(r) for r in runs]
        lines.append(f"| {label} | " + " | ".join(fmt([x[k] for x in rows]) for k, _ in COLUMNS) + " |")
    lines += ["", "Paired bootstrap over claims (2000 resamples), seed-averaged difference [95% CI]:", ""]
    for a, b in [("Augmented SFT", "Original SFT"), ("Paired hinge", "Original SFT"), ("Paired hinge", "Augmented SFT")]:
        cells = []
        for label, stat, eval_set in [("macro-F1", macro_f1, "test"), ("gold-removed NEI%", nei_rate, "evidence_removed"),
                                      ("control NEI%", nei_rate, "random_removed"), ("partial acc.", accuracy, "partial_removed")]:
            d, lo, hi = bootstrap(SEEDS[a], SEEDS[b], stat, eval_set)
            cells.append(f"{label} {d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append(f"- {a} vs {b}: " + "; ".join(cells))
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "report_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
