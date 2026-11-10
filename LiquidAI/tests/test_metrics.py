"""Hand-calculated metrics and calibration edge cases."""

import json
import unittest

from ..metrics import TestMetrics


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        metrics = TestMetrics(ece_bins=10)
        metrics.update("NEI", [0.8, 0.1, 0.1], [0, 0], [0.9, 0.1])
        metrics.update("SUPPORT", [0.1, 0.7, 0.2], [1, 0], [0.8, 0.2])
        metrics.update("CONTRADICT", [0.1, 0.6, 0.3], [1, 1], [0.7, 0.4])
        metrics.update("SUPPORT", [0.1, 0.1, 0.8], [0], [0.6])
        result = metrics.result()
        self.assertEqual(result["label_accuracy"], 0.5)
        self.assertEqual(result["label_macro_f1"], 0.5)
        self.assertAlmostEqual(result["sentence_level_evidence_f1"], 4 / 7)
        self.assertAlmostEqual(result["label_ece"], 0.375)
        self.assertEqual(result["label_confusion_matrix"], [[1, 0, 0], [0, 1, 1], [0, 1, 0]])
        self.assertEqual(result["num_valid_sentences"], 7)
        self.assertEqual(result["evidence"]["tp"], 2)
        self.assertEqual(result["evidence"]["fp"], 2)
        self.assertEqual(result["evidence"]["fn"], 1)

    def test_confidence_one_empty_bins_and_absent_classes(self):
        metrics = TestMetrics()
        metrics.update("NEI", [1.0, 0.0, 0.0], [], [])
        result = metrics.result()
        self.assertEqual(result["label_accuracy"], 1.0)
        self.assertAlmostEqual(result["label_macro_f1"], 1 / 3)
        self.assertEqual(result["sentence_level_evidence_f1"], 0.0)
        self.assertEqual(result["label_ece"], 0.0)
        self.assertEqual(result["calibration"]["bins"][-1]["count"], 1)
        self.assertIsNone(result["calibration"]["bins"][0]["accuracy"])
        json.dumps(result, allow_nan=False)

    def test_nei_false_positives_and_threshold_equality(self):
        metrics = TestMetrics(evidence_threshold=0.5)
        metrics.update("NEI", [1.0, 0.0, 0.0], [0, 0], [0.5, 0.49])
        self.assertEqual(metrics.result()["evidence"]["fp"], 1)
        self.assertEqual(metrics.result()["evidence"]["tn"], 1)

    def test_rejects_invalid_or_empty_predictions(self):
        with self.assertRaisesRegex(ValueError, "zero samples"):
            TestMetrics().result()
        for probs in ([float("nan"), 0, 0], [0.2, 0.2, 0.2], [1.1, 0, -0.1]):
            with self.assertRaises(ValueError):
                TestMetrics().update("NEI", probs, [], [])
        with self.assertRaisesRegex(ValueError, "align"):
            TestMetrics().update("NEI", [1, 0, 0], [1], [])
