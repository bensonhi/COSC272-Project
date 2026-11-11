"""Exercise epoch evaluation with real CPU optimization and held-out scoring."""

from collections import Counter
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

from .. import cli, data, evaluation, training
from ..config import LABEL_TO_ID
from ..encoding import EncodedQuestion, EncodedSample
from .fixtures import fixture


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft")),
                     "Training dependencies not installed")
class EpochEvaluationTests(unittest.TestCase):
    def run_training(self, root, max_steps=None, loss="ce", train_groups=1):
        import torch

        for split, claim_id in (("train", 10), ("validation", 11), ("test", 12)):
            directory = root / split
            directory.mkdir()
            record = fixture()
            record["id"] = claim_id
            if split == "train" and loss == "pair":
                original = record["groups"][0]
                original["variant"] = "original"
                deleted = deepcopy(original)
                deleted.update(variant="drop_all_evidence", label="NEI")
                for article in deleted["articles"]:
                    article["sentences"] = [s for s in article["sentences"] if s["original_index"] != 2]
                    deleted["deleted_sentences"].append({"doc_id": article["doc_id"], "original_index": 2})
                for evidence in deleted["evidence"]:
                    evidence["kept"] = False
                record["groups"].append(deleted)
            elif split != "train":
                record["groups"].append(deepcopy(record["groups"][0]))
            else:
                record["groups"].extend(deepcopy(record["groups"][0]) for _ in range(train_groups - 1))
            (directory / f"{claim_id}.json").write_text(json.dumps(record))
        args = cli.parse_args([
            "--mode", "train", "--device", "cpu", "--epochs", "2", "--loss", loss,
            "--train-dir", str(root / "train"), "--validation-dir", str(root / "validation"),
            "--test-dir", str(root / "test"), "--output-dir", str(root / "out"),
            "--max-groups", "2" if loss == "pair" else str(train_groups),
            "--gradient-accumulation-steps", "2", "--question-batch-size", "2",
            *(["--max-steps", str(max_steps)] if max_steps is not None else []),
        ])
        samples = data.load_samples(args.train_dir, args.max_groups)
        encode_counts = Counter()

        def encode(sample, max_length):
            encode_counts[sample.claim_id] += 1
            questions = [EncodedQuestion([1], LABEL_TO_ID[sample.label], True)]
            for article in sample.articles:
                for index, valid in enumerate(article.sentence_mask):
                    if valid:
                        questions.append(EncodedQuestion([1], article.evidence_targets[index], False,
                                                         (article.doc_id, index)))
            return EncodedSample(sample.key, [1], questions)

        codec = SimpleNamespace(stance_ids=[[0], [1], [2]], evidence_ids=[[3], [4]], encode=encode)

        class ScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.zeros(5))
                self.config = SimpleNamespace(_commit_hash="local-test")
                self.forward_modes = []

            def forward(self, input_ids, logits_to_keep, **kwargs):
                self.forward_modes.append(self.training)
                assert torch.is_grad_enabled() == self.training
                logits = self.scores[None, None, :].expand(input_ids.shape[0], len(logits_to_keep), -1)
                return SimpleNamespace(logits=logits)

        model = ScoreModel()
        snapshots = []

        def save_adapter(model, tokenizer, destination, metadata):
            snapshots.append((destination.name, model.scores.detach().clone()))

        with patch.object(training, "load_train_model", return_value=(model, None, 0, "cpu", "float32")), \
             patch.object(training, "DecisionCodec", return_value=codec), \
             patch.object(training, "attach_lora", return_value=model), \
             patch.object(training, "save_adapter", side_effect=save_adapter), \
             patch.object(evaluation, "load_test_model", side_effect=AssertionError("Epoch evaluation reloaded model")), \
             redirect_stdout(io.StringIO()):
            training.train(args, samples, data.dataset_summary(samples))
        return args, model, snapshots, encode_counts

    def test_each_epoch_scores_both_complete_splits_and_resumes_training(self):
        import torch

        for loss in ("ce", "pair"):
            with self.subTest(loss=loss), tempfile.TemporaryDirectory() as directory:
                args, model, snapshots, counts = self.run_training(Path(directory), loss=loss)
                history = [json.loads(line) for line in (args.output_dir / "epoch_metrics.jsonl").read_text().splitlines()]
                self.assertEqual([(row["epoch"], row["step"]) for row in history], [(1, 1), (2, 2)])
                self.assertEqual(counts, {10: 2 if loss == "pair" else 1, 11: 2, 12: 2})
                for epoch, row in enumerate(history, start=1):
                    for split in ("validation", "test"):
                        output = args.output_dir / f"epoch-{epoch}" / split
                        metrics = json.loads((output / "metrics.json").read_text())
                        self.assertEqual(metrics, row[split])
                        self.assertEqual(metrics["num_samples"], 2)
                        self.assertEqual(metrics["num_valid_sentences"], 8)
                        self.assertEqual(metrics["evaluation"]["epoch"], epoch)
                        self.assertEqual(metrics["evaluation"]["split"], split)
                        self.assertTrue((output / f"{split}_config.json").is_file())
                        self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 2)
                        self.assertFalse((output / "predictions.partial.jsonl").exists())
                        for key in ("label_accuracy", "label_macro_f1", "sentence_level_evidence_f1", "label_ece"):
                            self.assertTrue(0 <= metrics[key] <= 1)
                self.assertTrue(model.training)
                self.assertTrue(model.scores.requires_grad)
                self.assertIn(False, model.forward_modes)
                last_eval = model.forward_modes.index(False)
                self.assertIn(True, model.forward_modes[last_eval:])
                self.assertEqual([name for name, _ in snapshots], ["checkpoint-step-1", "checkpoint-step-2", "final"])
                self.assertFalse(torch.equal(snapshots[0][1], snapshots[1][1]))
                torch.testing.assert_close(snapshots[1][1], snapshots[2][1])

    def test_max_steps_still_evaluates_both_splits_before_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            args, model, snapshots, counts = self.run_training(Path(directory), max_steps=1)
            history = [json.loads(line) for line in (args.output_dir / "epoch_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["validation"]["num_samples"], 2)
            self.assertEqual(history[0]["test"]["num_samples"], 2)
            self.assertFalse((args.output_dir / "epoch-2").exists())
            self.assertEqual([name for name, _ in snapshots], ["checkpoint-step-1", "final"])

    def test_mid_epoch_step_limit_saves_final_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            args, model, snapshots, counts = self.run_training(Path(directory), max_steps=1, train_groups=3)
            history = [json.loads(line) for line in (args.output_dir / "epoch_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), 1)
            self.assertFalse(history[0]["epoch_complete"])
            self.assertEqual([name for name, _ in snapshots], ["final"])
            for split in ("validation", "test"):
                self.assertEqual(history[0][split]["num_samples"], 2)
                self.assertIsNone(history[0][split]["evaluation"]["adapter_dir"])

    def test_multiple_updates_only_save_completed_epoch_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            args, model, snapshots, counts = self.run_training(Path(directory), train_groups=3)
            self.assertEqual([name for name, _ in snapshots], ["checkpoint-step-2", "checkpoint-step-4", "final"])
            history = [json.loads(line) for line in (args.output_dir / "epoch_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in history], [2, 4])
            self.assertTrue(all(row["epoch_complete"] for row in history))

    def test_failed_evaluation_restores_training_mode_and_gradients(self):
        import torch

        model = torch.nn.Linear(1, 1)
        for initial_training in (True, False):
            with self.subTest(initial_training=initial_training):
                model.train(initial_training)
                with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                    with evaluation.evaluation_mode(model):
                        self.assertFalse(model.training)
                        self.assertFalse(torch.is_grad_enabled())
                        raise RuntimeError("prediction failed")
                self.assertEqual(model.training, initial_training)
                self.assertTrue(torch.is_grad_enabled())
                self.assertTrue(model.weight.requires_grad)
