"""Evaluation records, inference without gold targets and CLI output."""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .. import cli, data, encoding, evaluation, modeling
from ..config import LABEL_TO_ID
from .fixtures import fixture


class EvaluationTests(unittest.TestCase):
    def test_prediction_record_preserves_masks_and_document_ids(self):
        sample = data.samples_from_record(fixture())[0]
        probs = {(100, 0): 0.2, (100, 2): 0.9, (200, 0): 0.8, (200, 2): 0.1}
        record = evaluation.prediction_record(sample, [0.8, 0.1, 0.1], probs, 0.5)
        self.assertEqual(record["predicted_label"], "NEI")
        first, second = record["articles"]
        # A predicted NEI does not hide evidence false positives.
        self.assertEqual(first["predicted_evidence_indices"], [2])
        self.assertEqual(second["predicted_evidence_indices"], [0])
        self.assertEqual(first["gold_evidence_indices"], [2])
        self.assertIsNone(first["sentences"][1]["evidence_probability"])
        self.assertIsNone(first["sentences"][3]["gold_evidence"])
        with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
            evaluation.prediction_record(sample, [0.8, 0.1, 0.1], {}, 0.5)

    def test_test_mode_cannot_silently_evaluate_subset(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                cli.parse_args(["--mode", "test", "--max-groups", "1"])

    def test_missing_adapter_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            args = cli.parse_args(["--mode", "test", "--adapter-dir", directory])
            with self.assertRaisesRegex(FileNotFoundError, "Finish training first"):
                modeling.load_test_model(args)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.codec = SimpleNamespace(stance_ids=[[0, 3], [1], [2]], evidence_ids=[[4, 6], [5]])
        self.questions = [
            encoding.EncodedQuestion([1], 1, True),
            encoding.EncodedQuestion([1, 2], 1, False, (100, 2)),
            encoding.EncodedQuestion([1, 2, 3], 0, False, (200, 0)),
        ]

    def test_inference_ignores_gold_targets_and_disables_gradients(self):
        torch = self.torch

        class FixedModel:
            def __call__(self, input_ids, logits_to_keep, **kwargs):
                assert not torch.is_grad_enabled()
                logits = torch.arange(8, dtype=torch.float32).repeat(input_ids.shape[0], len(logits_to_keep), 1)
                return SimpleNamespace(logits=logits)

        encoded = encoding.EncodedSample("a", [8, 9], deepcopy(self.questions))
        before = evaluation.predict_sample(FixedModel(), self.codec, encoded, 0, "cpu", 2)
        for q in encoded.questions:
            q.target = 1 - q.target
        after = evaluation.predict_sample(FixedModel(), self.codec, encoded, 0, "cpu", 1)
        self.assertEqual(before, after)

    def test_test_cli_saves_metrics_for_all_files_and_groups(self):
        torch = self.torch

        class FixedModel(torch.nn.Module):
            def forward(self, input_ids, logits_to_keep, **kwargs):
                assert not torch.is_grad_enabled()
                assert not self.training
                logits = torch.arange(8, dtype=torch.float32).repeat(input_ids.shape[0], len(logits_to_keep), 1)
                return SimpleNamespace(logits=logits)

        def encode(sample, max_length):
            qs = [encoding.EncodedQuestion([1], LABEL_TO_ID[sample.label], True)]
            for article in sample.articles:
                for i, available in enumerate(article.sentence_mask):
                    if available:
                        qs.append(encoding.EncodedQuestion([1, 2], article.evidence_targets[i], False, (article.doc_id, i)))
            return encoding.EncodedSample(sample.key, [8, 9], qs)

        codec = SimpleNamespace(**vars(self.codec), encode=encode)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = directory / "data"
            data.mkdir()
            grouped = fixture()
            grouped["groups"].append(deepcopy(grouped["groups"][0]))
            (data / "10.json").write_text(json.dumps(grouped))
            flat = {"id": 11, **fixture()["groups"][0]}
            (data / "11.json").write_text(json.dumps(flat))
            output = directory / "results"
            loaded = (FixedModel(), codec, 0, "cpu", {"base_model": "unit-test-only"})
            with patch.object(evaluation, "load_test_model", return_value=loaded), redirect_stdout(io.StringIO()):
                cli.main(["--mode", "test", "--test-dir", str(data), "--eval-output-dir", str(output)])
            metrics = json.loads((output / "metrics.json").read_text())
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["num_samples"], 3)
            self.assertEqual(metrics["num_valid_sentences"], 12)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["sample_id"] for row in rows}), 3)
            self.assertFalse(metrics["evaluation"]["gold_evidence_used_as_input"])
            self.assertTrue((output / "test_config.json").is_file())
            self.assertFalse((output / "predictions.partial.jsonl").exists())
            for key in ("label_accuracy", "label_macro_f1", "sentence_level_evidence_f1", "label_ece"):
                self.assertTrue(0 <= metrics[key] <= 1)
