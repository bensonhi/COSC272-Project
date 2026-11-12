"""Train/test sequencing and automatic adapter selection."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .. import cli
from ..config import DEFAULT_MODEL
from .fixtures import fixture


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train_dir = self.root / "train"
        self.validation_dir = self.root / "validation"
        self.test_dir = self.root / "test"
        self.output_dir = self.root / "model"
        self.train_dir.mkdir()
        self.validation_dir.mkdir()
        self.test_dir.mkdir()
        record = fixture()
        record["groups"].append(deepcopy(record["groups"][0]))
        (self.train_dir / "10.json").write_text(json.dumps(record))
        validation_record = fixture()
        validation_record["id"] = 13
        (self.validation_dir / "13.json").write_text(json.dumps(validation_record))
        for claim_id in (11, 12):
            record = fixture()
            record["id"] = claim_id
            (self.test_dir / f"{claim_id}.json").write_text(json.dumps(record))
        self.arguments = [
            "--train-dir", str(self.train_dir), "--validation-dir", str(self.validation_dir),
            "--test-dir", str(self.test_dir),
            "--output-dir", str(self.output_dir),
        ]

    def test_defaults_follow_training_output_directory(self):
        args = cli.parse_args(self.arguments)
        self.assertEqual(args.mode, "train_test")
        self.assertEqual(args.loss, "ce")
        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertEqual(args.device, 0)
        self.assertIsInstance(args.device, int)
        self.assertEqual(args.adapter_dir, self.output_dir / "final")
        self.assertEqual(args.eval_output_dir, self.output_dir / "test")
        self.assertEqual(args.validation_dir, self.validation_dir)
        self.assertFalse(args.original)

    def test_original_only_filters_training_and_dry_run_counts(self):
        record = json.loads((self.train_dir / "10.json").read_text())
        record["groups"][0]["variant"] = "drop_1_groups"
        record["groups"][1]["variant"] = "original"
        (self.train_dir / "10.json").write_text(json.dumps(record))
        for mode in ("train", "train_test"):
            with self.subTest(mode=mode):
                with patch.object(cli, "train") as train, patch.object(cli, "evaluate") as evaluate:
                    cli.main(self.arguments + ["--mode", mode, "--original", "--max-groups", "1"])
                args, samples, summary = train.call_args.args
                self.assertTrue(args.original)
                self.assertEqual([(s.variant, s.group_index) for s in samples], [("original", 1)])
                self.assertEqual(summary["groups"], 1)
                if mode == "train_test":
                    self.assertEqual(len(evaluate.call_args.args[1]), 2)
                output = io.StringIO()
                with redirect_stdout(output):
                    cli.main(self.arguments + ["--mode", mode, "--original", "--dry-run", "--max-groups", "1"])
                counts = json.loads(output.getvalue())
                self.assertEqual(counts["train"]["groups"], 1)
                self.assertEqual(counts["validation"]["groups"], 1)
                self.assertEqual(counts["test"]["groups"], 2)

    def test_original_does_not_filter_standalone_test(self):
        with patch.object(cli, "evaluate") as evaluate:
            cli.main(self.arguments + ["--mode", "test", "--original"])
        self.assertEqual(len(evaluate.call_args.args[1]), 2)

    def test_original_rejects_incompatible_pair_loss(self):
        with patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit):
            cli.parse_args(self.arguments + ["--original", "--loss", "pair"])
        self.assertIn("requires drop_all_evidence", stderr.getvalue())

    def test_device_accepts_gpu_index_or_explicit_cpu(self):
        for value, expected in (("0", 0), ("2", 2), ("cpu", "cpu")):
            with self.subTest(value=value):
                self.assertEqual(cli.parse_args(self.arguments + ["--device", value]).device, expected)
        for value in ("auto", "cuda", "cuda:0", "-1", "1.5"):
            with self.subTest(value=value), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                cli.parse_args(self.arguments + ["--device", value])

    def test_pair_loss_options(self):
        args = cli.parse_args(self.arguments + ["--loss", "pair", "--pair-margin", "0.5", "--pair-loss-weight", "0.2"])
        self.assertEqual((args.loss, args.pair_margin, args.pair_loss_weight), ("pair", 0.5, 0.2))
        for options in (["--loss", "other"], ["--pair-margin", "0"], ["--pair-margin", "nan"],
                        ["--pair-loss-weight", "-1"], ["--pair-loss-weight", "inf"]):
            with self.subTest(options=options), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                cli.parse_args(self.arguments + options)

    def test_training_then_complete_test_with_fresh_adapter(self):
        events = []

        def train(args, samples, summary):
            events.append("train")
            self.assertEqual(len(samples), 1)
            self.assertEqual(args.model, DEFAULT_MODEL)
            self.assertEqual(args.revision, "main")

        def evaluate(args, samples, summary):
            events.append("test")
            self.assertEqual(args.mode, "test")
            self.assertEqual(args.adapter_dir, self.output_dir / "final")
            self.assertEqual(args.eval_output_dir, self.output_dir / "test")
            self.assertIsNone(args.model)
            self.assertIsNone(args.revision)
            self.assertIsNone(args.max_groups)
            self.assertIsNone(args.max_steps)
            self.assertEqual([s.claim_id for s in samples], [11, 12])
            self.assertEqual(summary["groups"], 2)

        with patch.object(cli, "train", side_effect=train), patch.object(cli, "evaluate", side_effect=evaluate):
            cli.main(self.arguments + ["--max-groups", "1", "--max-steps", "1", "--revision", "main"])
        self.assertEqual(events, ["train", "test"])

    def test_train_mode_passes_held_out_directories_without_final_reload(self):
        with patch.object(cli, "train") as train, patch.object(cli, "evaluate") as evaluate:
            cli.main(self.arguments + ["--mode", "train"])
        train.assert_called_once()
        args = train.call_args.args[0]
        self.assertEqual(args.validation_dir, self.validation_dir)
        self.assertEqual(args.test_dir, self.test_dir)
        evaluate.assert_not_called()

    def test_failed_training_does_not_start_evaluation(self):
        with patch.object(cli, "train", side_effect=RuntimeError("training failed")), patch.object(cli, "evaluate") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                cli.main(self.arguments)
        evaluate.assert_not_called()

    def test_dry_run_checks_all_splits_without_training(self):
        for mode in ("train", "train_test"):
            with self.subTest(mode=mode):
                output = io.StringIO()
                with patch.object(cli, "train") as train, patch.object(cli, "evaluate") as evaluate, redirect_stdout(output):
                    cli.main(self.arguments + ["--mode", mode, "--dry-run", "--max-groups", "1"])
                summary = json.loads(output.getvalue())
                self.assertEqual(summary["train"]["groups"], 1)
                self.assertEqual(summary["validation"]["groups"], 1)
                self.assertEqual(summary["test"]["groups"], 2)
                train.assert_not_called()
                evaluate.assert_not_called()

    def test_combined_mode_rejects_unrelated_existing_adapter(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            cli.parse_args(self.arguments + ["--adapter-dir", str(self.root / "old-adapter")])

    def test_existing_test_results_fail_before_training(self):
        results = self.root / "existing-results"
        results.mkdir()
        (results / "metrics.json").write_text("{}")
        with patch.object(cli, "train") as train:
            with self.assertRaisesRegex(ValueError, "not empty"):
                cli.main(self.arguments + ["--eval-output-dir", str(results)])
        train.assert_not_called()
