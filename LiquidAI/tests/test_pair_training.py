"""Check the actual training loop against the full-dataset objective."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .. import cli, training
from ..config import LABEL_TO_ID
from ..data import dataset_summary
from ..encoding import EncodedQuestion, EncodedSample
from .test_pairing import paired_samples


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft")), "Training dependencies not installed")
class PairTrainingTests(unittest.TestCase):
    def test_accumulated_gradients_match_mean_ce_bce_plus_mean_pairs(self):
        import torch
        import torch.nn.functional as F

        samples = paired_samples(10) + paired_samples(11)
        unpaired = paired_samples(12)[1]
        unpaired.variant = "original"
        samples.append(unpaired)
        encoded = {}
        for i, sample in enumerate(samples):
            questions = [EncodedQuestion([6], LABEL_TO_ID[sample.label], True)]
            for article in sample.articles:
                for j, valid in enumerate(article.sentence_mask):
                    if valid:
                        questions.append(EncodedQuestion([6, 6], article.evidence_targets[j], False))
            encoded[sample.key] = EncodedSample(sample.key, [i], questions)
        codec = SimpleNamespace(
            stance_ids=[[0], [1], [2]], evidence_ids=[[3], [4]],
            encode=lambda sample, max_length: encoded[sample.key],
        )

        class ScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.zeros(len(samples), 7))
                self.config = SimpleNamespace(_commit_hash="local-test")

            def forward(self, input_ids, logits_to_keep, **kwargs):
                scores = self.scores[input_ids[:, 0]]
                return SimpleNamespace(logits=scores[:, None, :].expand(-1, len(logits_to_keep), -1))

        for mode in ("ce", "pair"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                model = ScoreModel()
                gradients = []

                class RecordingOptimizer(torch.optim.SGD):
                    def step(self, closure=None):
                        # Keep parameters fixed so the weighted sum of window
                        # gradients can be compared to one full objective.
                        gradients.append(model.scores.grad.clone())

                args = cli.parse_args([
                    "--mode", "train", "--loss", mode, "--device", "cpu", "--epochs", "1",
                    "--gradient-accumulation-steps", "2", "--question-batch-size", "2",
                    "--max-grad-norm", "1000000", "--pair-loss-weight", "0.7",
                    "--evidence-loss-weight", "0.4", "--output-dir", str(Path(directory) / "out"),
                ])
                with patch.object(training, "load_train_model", return_value=(model, None, 0, torch.device("cpu"), "float32")), \
                     patch.object(training, "DecisionCodec", return_value=codec), \
                     patch.object(training, "attach_lora", return_value=model), \
                     patch.object(training, "save_adapter"), \
                     patch.object(training, "load_samples", return_value=samples), \
                     patch.object(training, "evaluate_model", return_value={}), \
                     patch.object(torch.optim, "AdamW", RecordingOptimizer):
                    training.train(args, samples, dataset_summary(samples))

                reference = model.scores.detach().clone().requires_grad_()
                targets = torch.tensor([LABEL_TO_ID[s.label] for s in samples])
                objective = F.cross_entropy(reference[:, :3], targets)
                evidence_losses = []
                for i, sample in enumerate(samples):
                    targets = torch.tensor([float(q.target) for q in encoded[sample.key].questions if not q.is_stance])
                    evidence_losses.append(F.binary_cross_entropy_with_logits((reference[i, 3] - reference[i, 4]).expand_as(targets), targets))
                objective = objective + 0.4 * torch.stack(evidence_losses).mean()
                if mode == "pair":
                    margins = reference[:, 1] - reference[:, 0]
                    objective = objective + 0.7 * F.relu(1 - (margins[[0, 2]] - margins[[1, 3]])).mean()
                objective.backward()
                rows = [json.loads(line) for line in (args.output_dir / "train_metrics.jsonl").read_text().splitlines()]
                self.assertEqual([r["groups"] for r in rows], [2, 2, 1])
                combined = sum(g * r["groups"] / len(samples) for g, r in zip(gradients, rows))
                torch.testing.assert_close(combined, reference.grad)
                self.assertAlmostEqual(sum(r["loss"] * r["groups"] / len(samples) for r in rows), objective.item(), places=6)
                if mode == "pair":
                    self.assertEqual(sum(r["pairs"] for r in rows), 2)
                    for row in rows:
                        self.assertAlmostEqual(row["loss"], row["stance_loss"] + 0.4 * row["evidence_loss"] + row["pair_loss_contribution"], places=6)
                    saved = json.loads((args.output_dir / "training_config.json").read_text())
                    self.assertEqual(saved["pairing"]["count"], 2)
                else:
                    self.assertTrue(all("pair_loss" not in r for r in rows))
