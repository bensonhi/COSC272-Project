"""Shared evaluation harness for every baseline.

Each baseline only implements `predict(sample) -> (label_probabilities, sentence_probabilities)`:
label probabilities in LiquidAI.config.LABELS order (NEI, SUPPORT, CONTRADICT) and one
P(evidence) per available sentence, keyed by (doc_id, original_index). Scoring reuses the
LiquidAI package's prediction records and TestMetrics unchanged, so every baseline is
reported with exactly the same metrics as the fine-tuned d1 models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LiquidAI.config import LABELS, LABEL_TO_ID  # noqa: E402
from LiquidAI.data import Sample, dataset_summary, load_samples  # noqa: E402
from LiquidAI.evaluation import prediction_record  # noqa: E402
from LiquidAI.metrics import TestMetrics  # noqa: E402

TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
RESULTS_DIR = ROOT / "baselines" / "results"
# Large artifacts (adapters, caches) live on node-local scratch; results stay in the repo.
SCRATCH = Path(os.environ.get("COSC272_SCRATCH", "/scratch/f008k3y/cosc272"))

Predictor = Callable[[Sample], tuple[list[float], dict[tuple[int, int], float]]]


def sentence_keys(sample: Sample) -> list[tuple[int, int]]:
    return [
        (a.doc_id, i) for a in sample.articles
        for i, available in enumerate(a.sentence_mask) if available
    ]


def run_baseline(name: str, predict: Predictor, config: dict, samples: list[Sample] | None = None,
                 evidence_threshold: float = 0.5, ece_bins: int = 10, sync: Callable[[], None] | None = None,
                 overwrite: bool = False, latency: dict | None = None) -> dict:
    """Score `predict` on the full test set and write metrics.json / predictions.jsonl.

    `latency` replaces the measured predict() timings, for baselines that run batched
    and measure single-claim latency separately (predict() then only reads a cache).
    """
    samples = samples if samples is not None else load_samples(TEST_DIR)
    out_dir = RESULTS_DIR / name
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise ValueError(f"{out_dir} is not empty; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = TestMetrics(evidence_threshold, ece_bins)
    latencies = []
    with (out_dir / "predictions.partial.jsonl").open("w", encoding="utf-8") as handle:
        for i, sample in enumerate(samples, start=1):
            started = time.perf_counter()
            label_probs, sentence_probs = predict(sample)
            if sync is not None:
                sync()
            latencies.append(time.perf_counter() - started)
            record = prediction_record(sample, label_probs, sentence_probs, evidence_threshold)
            if latency is None:
                record["latency_seconds"] = latencies[-1]
            rows = [r for a in record["articles"] for r in a["sentences"] if r["available"]]
            metrics.update(sample.label, label_probs, [r["gold_evidence"] for r in rows],
                           [r["evidence_probability"] for r in rows])
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if i == 1 or i % 50 == 0 or i == len(samples):
                print(f"[{name}] {i}/{len(samples)}", flush=True)
    (out_dir / "predictions.partial.jsonl").rename(out_dir / "predictions.jsonl")
    result = metrics.result()
    result["latency"] = {
        "median_seconds_per_claim": statistics.median(latencies),
        "mean_seconds_per_claim": statistics.fmean(latencies),
        "total_seconds": sum(latencies),
        "note": "wall clock of predict(sample), one claim at a time",
    } if latency is None else latency
    result["baseline"] = {"name": name, **config, "test_summary": dataset_summary(samples),
                          "evidence_threshold": evidence_threshold}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("label_accuracy", "label_macro_f1",
                                             "sentence_level_evidence_f1", "label_ece")}
                     | {"median_latency_s": result["latency"]["median_seconds_per_claim"]}, indent=2))
    return result


EVAL_SETS = ("test", "evidence_removed", "random_removed")


def eval_samples(eval_set: str, seed: int = 0) -> list[Sample]:
    """The test split, or a deterministic stress variant built from it in memory (data/ is untouched).

    evidence_removed  claims with gold evidence; every gold evidence sentence becomes an empty
                      slot (the training-time deletion format) and the target becomes NEI.
    random_removed    control for deletion artifacts: the same claims with the same number of
                      randomly chosen NON-evidence sentences removed; label and evidence unchanged.
    Gold rationales can omit valid evidence, so some evidence_removed inputs may still be
    decidable; NEI rate there is therefore a lower bound on correct abstention, not exact.
    """
    import random
    from dataclasses import replace

    samples = load_samples(TEST_DIR)
    if eval_set == "test":
        return samples
    rng = random.Random(seed)
    out = []
    for s in samples:
        if s.label == "NEI":
            continue
        gold = [(a.doc_id, i) for a in s.articles for i, t in enumerate(a.evidence_targets) if t]
        if eval_set == "evidence_removed":
            drop = set(gold)
        else:
            others = [(a.doc_id, i) for a in s.articles for i, ok in enumerate(a.sentence_mask)
                      if ok and not a.evidence_targets[i]]
            drop = set(rng.sample(others, min(len(gold), len(others))))
        articles = []
        for a in s.articles:
            sentences = ["" if (a.doc_id, i) in drop else t for i, t in enumerate(a.sentences)]
            targets = [0 if eval_set == "evidence_removed" else t for t in a.evidence_targets]
            articles.append(replace(a, sentences=sentences, evidence_targets=targets,
                                    sentence_mask=[bool(t) for t in sentences]))
        label = "NEI" if eval_set == "evidence_removed" else s.label
        out.append(replace(s, articles=articles, label=label, variant=eval_set))
    return out


def result_name(name: str, eval_set: str) -> str:
    return name if eval_set == "test" else f"{eval_set}/{name}"


def one_hot(label: str, confidence: float = 1.0) -> list[float]:
    """A distribution putting `confidence` on `label` and the rest evenly on the others."""
    rest = (1.0 - confidence) / (len(LABELS) - 1)
    return [confidence if i == LABEL_TO_ID[label] else rest for i in range(len(LABELS))]
