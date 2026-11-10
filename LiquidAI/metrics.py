"""Label accuracy/macro-F1, sentence evidence micro-F1 and label ECE."""

from __future__ import annotations

import math
from typing import Any

from .config import LABELS, LABEL_TO_ID


class TestMetrics:
    """Streaming metrics. All reported scores are fractions in [0, 1].

    Macro-F1 always includes all three labels, with zero_division=0. Evidence
    F1 pools TP/FP/FN over available sentences across samples and articles; it
    is independent of the predicted stance. ECE uses maximum class probability
    and bins [b/M, (b+1)/M), with the final bin also including confidence 1.
    """

    def __init__(self, evidence_threshold: float = 0.5, ece_bins: int = 10):
        if not 0 <= evidence_threshold <= 1:
            raise ValueError("Evidence threshold must be in [0, 1]")
        if not isinstance(ece_bins, int) or isinstance(ece_bins, bool) or ece_bins < 1:
            raise ValueError("ECE bin count must be a positive integer")
        self.threshold = evidence_threshold
        self.confusion = [[0] * len(LABELS) for _ in LABELS]
        self.bin_count = [0] * ece_bins
        self.bin_correct = [0] * ece_bins
        self.bin_confidence = [0.0] * ece_bins
        self.tp = self.fp = self.fn = self.tn = 0
        self.samples = 0

    def update(self, gold_label, label_probabilities, sentence_targets, sentence_probabilities):
        if gold_label not in LABEL_TO_ID:
            raise ValueError(f"Unknown gold label {gold_label!r}")
        if len(label_probabilities) != len(LABELS):
            raise ValueError("Expected probabilities for NEI, SUPPORT, CONTRADICT")
        for value in (*label_probabilities, *sentence_probabilities):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid predicted probability: {value}")
        if not math.isclose(sum(label_probabilities), 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("Label probabilities must sum to one")
        if len(sentence_targets) != len(sentence_probabilities):
            raise ValueError("Evidence targets and predictions must align")
        if any(value not in (0, 1) for value in sentence_targets):
            raise ValueError("Evidence targets must be binary")

        gold = LABEL_TO_ID[gold_label]
        prediction = max(range(len(LABELS)), key=lambda i: label_probabilities[i])
        confidence = label_probabilities[prediction]
        self.confusion[gold][prediction] += 1
        self.samples += 1
        bin_index = min(int(confidence * len(self.bin_count)), len(self.bin_count) - 1)
        self.bin_count[bin_index] += 1
        self.bin_correct[bin_index] += int(prediction == gold)
        self.bin_confidence[bin_index] += confidence
        for target, probability in zip(sentence_targets, sentence_probabilities):
            selected = probability >= self.threshold
            self.tp += int(target == 1 and selected)
            self.fp += int(target == 0 and selected)
            self.fn += int(target == 1 and not selected)
            self.tn += int(target == 0 and not selected)

    def result(self) -> dict[str, Any]:
        if not self.samples:
            raise ValueError("Cannot compute metrics on zero samples")

        def divide(numerator, denominator):
            return numerator / denominator if denominator else 0.0

        per_label = {}
        for i, label in enumerate(LABELS):
            tp = self.confusion[i][i]
            support = sum(self.confusion[i])
            predicted = sum(row[i] for row in self.confusion)
            per_label[label] = {
                "precision": divide(tp, predicted), "recall": divide(tp, support),
                "f1": divide(2 * tp, support + predicted), "support": support,
            }
        bins, ece = [], 0.0
        for index, count in enumerate(self.bin_count):
            accuracy = self.bin_correct[index] / count if count else None
            confidence = self.bin_confidence[index] / count if count else None
            contribution = count / self.samples * abs(accuracy - confidence) if count else 0.0
            ece += contribution
            bins.append({
                "lower": index / len(self.bin_count), "upper": (index + 1) / len(self.bin_count),
                "count": count, "accuracy": accuracy, "mean_confidence": confidence,
                "ece_contribution": contribution,
            })
        return {
            "label_accuracy": sum(self.confusion[i][i] for i in range(len(LABELS))) / self.samples,
            "label_macro_f1": sum(row["f1"] for row in per_label.values()) / len(LABELS),
            "sentence_level_evidence_f1": divide(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "label_ece": ece,
            "num_samples": self.samples,
            "num_valid_sentences": self.tp + self.fp + self.fn + self.tn,
            "label_order": list(LABELS),
            "label_confusion_matrix": self.confusion,  # rows=gold, columns=predicted
            "per_label": per_label,
            "evidence": {
                "averaging": "micro, evidence positive class", "threshold": self.threshold,
                "precision": divide(self.tp, self.tp + self.fp),
                "recall": divide(self.tp, self.tp + self.fn),
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "conditioned_on_predicted_label": False,
            },
            "calibration": {
                "type": "top-label ECE", "num_bins": len(self.bin_count),
                "binning": "equal-width, left-closed/right-open; last bin includes 1",
                "bins": bins,
            },
            "zero_division": 0,
            "score_scale": "0 to 1",
        }
