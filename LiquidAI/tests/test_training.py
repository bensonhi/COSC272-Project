"""Loss, gradient accumulation and variable-length model batches."""

import importlib.util
from types import SimpleNamespace
import unittest

from .. import encoding, losses, modeling


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class TrainingMathTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.codec = SimpleNamespace(stance_ids=[[0, 3], [1], [2]], evidence_ids=[[4, 6], [5]])
        self.questions = [
            encoding.EncodedQuestion([1], 1, True),
            encoding.EncodedQuestion([1, 2], 1, False, (100, 2)),
            encoding.EncodedQuestion([1, 2, 3], 0, False, (200, 0)),
        ]

    def test_microbatch_loss_and_gradients_equal_joint_loss(self):
        torch = self.torch
        torch.manual_seed(2)
        full = torch.randn(3, 8, requires_grad=True)
        parts = full.detach().clone().requires_grad_()
        weight = torch.tensor([1.0, 2.0, 3.0])
        loss = losses.joint_loss(full, self.questions, self.codec, 2, 0.7, 2.0, weight)[0]
        loss.backward()
        chunk_loss = 0.0
        for indices in ([0, 1], [2]):
            value = losses.joint_loss(parts[indices], [self.questions[i] for i in indices], self.codec, 2, 0.7, 2.0, weight)[0]
            value.backward()
            chunk_loss += value.item()
        self.assertAlmostEqual(loss.item(), chunk_loss, places=6)
        torch.testing.assert_close(full.grad, parts.grad)
        self.assertTrue(torch.isfinite(parts.grad).all())

    def test_evidence_uses_yes_minus_no_and_mean(self):
        torch = self.torch
        logits = torch.zeros(2, 8, requires_grad=True)
        logits.data[0, 4] = 2
        logits.data[1, 5] = 3
        loss, stance, evidence = losses.joint_loss(logits, self.questions[1:], self.codec, 2, 1, 1)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([2.0, -3.0]), torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(loss, expected)
        self.assertEqual(stance.item(), 0.0)
        torch.testing.assert_close(evidence, expected)

    def test_zero_valid_sentences_still_trains_stance(self):
        torch = self.torch
        logits = torch.randn(1, 8, requires_grad=True)
        loss, stance, evidence = losses.joint_loss(logits, self.questions[:1], self.codec, 0, 1, 1)
        loss.backward()
        self.assertEqual(evidence.item(), 0.0)
        torch.testing.assert_close(loss, stance)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_right_padding_and_per_row_answer_position(self):
        torch = self.torch
        sample = encoding.EncodedSample("sample", [8, 9], self.questions)
        ids, mask, positions = encoding.question_batch(sample, self.questions, 0, "cpu")
        self.assertEqual(ids.tolist(), [[8, 9, 1, 0, 0], [8, 9, 1, 2, 0], [8, 9, 1, 2, 3]])
        self.assertEqual(positions.tolist(), [2, 3, 4])
        self.assertEqual(mask.sum(1).tolist(), [3, 4, 5])

        class PositionModel:
            def __call__(self, input_ids, logits_to_keep, **kwargs):
                scores = input_ids[:, logits_to_keep, None].float()
                return SimpleNamespace(logits=scores)

        logits = modeling.forward_answer_logits(PositionModel(), ids, mask, positions)
        self.assertEqual(logits.flatten().tolist(), [1, 2, 3])
