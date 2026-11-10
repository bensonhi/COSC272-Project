"""LoRA training and held-out evaluation of d1-3B on SciFact.

Usage (from this directory):
    python run.py --dry-run
    python run.py --epochs 3 --output-dir outputs/d1-scifact
    python run.py --mode train --epochs 3
    python run.py --mode test --dry-run
    python run.py --mode test --adapter-dir outputs/d1-scifact/final

The default mode, train_test, trains first and then evaluates the freshly saved
output-dir/final adapter on the complete test set. Results default to
output-dir/test. Use train or test to run either phase on its own.

Each JSON's groups are separate training samples. A sample contains the claim
and ALL its articles; its stance label is not copied onto individual articles.
Sentence arrays, targets and masks are indexed by original_index, never by
new_index. Removed sentences are empty strings and contribute no evidence loss.

We retain d1's native choice/noul prompts and answer-token readouts. Training
uses ordinary differentiable forwards, NOT system_one(), whose inference path
disables autograd. Questions are processed in small batches with full context.
Loss = CE(stance) + evidence_loss_weight * mean_valid_sentences(BCE(evidence)).
Gradient accumulation averages this joint loss over groups, not questions.

Test mode reads ALL JSON files/groups in data/test, including flat JSON records.
Gold evidence and labels are evaluation targets; only claim/articles enter the
model prompt. It writes metrics.json, predictions.jsonl and test_config.json.
Metrics: label accuracy, three-class macro-F1, positive-class sentence micro-F1,
and top-label ECE (10 equal-width bins by default). Sentence predictions use a
fixed threshold, including on NEI examples, and exclude deleted/masked slots.
Test data is never used to fit thresholds, calibration parameters, or adapters.
Dependencies are listed in the adjacent requirements.txt. --dry-run needs only
the Python standard library and neither downloads nor loads the model."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
import sys

from .config import ROOT, DEFAULT_MODEL, LOGGER
from .data import load_samples, dataset_summary
from .evaluation import evaluate
from .training import train


def parse_device(value: str) -> int | str:
    """Accept an explicit GPU index or CPU, without probing hardware."""
    if value == "cpu":
        return value
    try:
        index = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("device must be a non-negative GPU index or cpu") from None
    if index < 0:
        raise argparse.ArgumentTypeError("GPU index must be non-negative")
    return index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train_test", "train", "test"), default="train_test",
                        help="Default: train, save the adapter, then evaluate the complete test set")
    parser.add_argument("--train-dir", type=Path, default=ROOT.parent / "data" / "train")
    parser.add_argument("--test-dir", type=Path, default=ROOT.parent / "data" / "test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "d1-scifact", help="Training output directory")
    parser.add_argument("--adapter-dir", type=Path, help="Existing LoRA adapter for test mode; defaults to output-dir/final")
    parser.add_argument("--eval-output-dir", type=Path, help="Test metrics and predictions directory; defaults to output-dir/test")
    parser.add_argument("--evidence-threshold", type=float, default=0.5, help="Fixed test evidence selection threshold (>=); never fitted on test data")
    parser.add_argument("--ece-bins", type=int, default=10, help="Equal-width bins for top-label ECE")
    parser.add_argument("--model", help="Base checkpoint; training defaults to LiquidAI/d1-3B, testing uses adapter metadata")
    parser.add_argument("--revision", help="Optional Hub revision/commit, recorded with the adapter")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--question-batch-size", type=int, default=2, help="Questions per forward, each with full group context")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Groups per optimizer update")
    parser.add_argument("--max-length", type=int, default=8192, help="Maximum tokens per question including full context; no truncation")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evidence-loss-weight", type=float, default=1.0)
    parser.add_argument("--evidence-pos-weight", type=float, default=1.0, help="Optional positive-class weight for evidence BCE")
    parser.add_argument("--balance-labels", action="store_true", help="Inverse-frequency stance weights computed from training groups")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", type=parse_device, default=0, metavar="GPU_ID|cpu",
                        help="GPU index (e.g. 0 or 1; default: 0), or cpu")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-groups", type=int, help="Use the first N groups for a smoke run")
    parser.add_argument("--max-steps", type=int, help="Stop after N optimizer updates")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/masks and print counts, without ML dependencies")
    args = parser.parse_args(argv)
    if args.mode in ("train", "train_test") and args.model is None:
        args.model = DEFAULT_MODEL
    if args.mode == "train_test" and args.adapter_dir is not None:
        parser.error("train_test evaluates its newly trained adapter; --adapter-dir is for --mode test")
    if args.adapter_dir is None:
        args.adapter_dir = args.output_dir / "final"
    if args.eval_output_dir is None:
        args.eval_output_dir = args.output_dir / "test"
    if args.mode == "train_test" and args.eval_output_dir.resolve() in (
        args.output_dir.resolve(), *args.output_dir.resolve().parents,
    ):
        parser.error("--eval-output-dir must not be the training output directory or its ancestor")
    if args.mode == "test" and (args.max_groups is not None or args.max_steps is not None):
        parser.error("Test mode evaluates all samples; --max-groups and --max-steps are training-only")
    if not math.isfinite(args.evidence_threshold) or not 0 <= args.evidence_threshold <= 1:
        parser.error("--evidence-threshold must be in [0, 1]")
    if args.ece_bins <= 0:
        parser.error("--ece-bins must be positive")
    for name in ("epochs", "question_batch_size", "gradient_accumulation_steps", "max_length", "lora_r", "lora_alpha", "logging_steps", "max_groups", "max_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "max_grad_norm", "evidence_pos_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("evidence_loss_weight", "weight_decay"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.lora_dropout < 1:
        parser.error("--warmup-ratio and --lora-dropout must be in [0, 1)")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_dir = args.test_dir if args.mode == "test" else args.train_dir
    samples = load_samples(data_dir, args.max_groups)
    summary = dataset_summary(samples)
    if args.dry_run:
        if args.mode == "train_test":
            test_samples = load_samples(args.test_dir)
            print(json.dumps({"train": summary, "test": dataset_summary(test_samples)}, indent=2))
        else:
            print(json.dumps(summary, indent=2))
        return
    if args.mode == "test":
        evaluate(args, samples, summary)
        return

    if args.mode == "train_test" and args.eval_output_dir.exists() and any(args.eval_output_dir.iterdir()):
        raise ValueError(f"Evaluation output directory is not empty: {args.eval_output_dir}; choose a new --eval-output-dir")
    train(args, samples, summary)
    if args.mode == "train_test":
        # Release optimizer/model cycles and cached GPU allocations before the
        # evaluator reloads the saved base model plus the newly trained adapter.
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and args.device != "cpu":
            torch.cuda.empty_cache()
        test_args = argparse.Namespace(**vars(args))
        test_args.mode = "test"
        test_args.adapter_dir = args.output_dir / "final"
        # Use the saved base-model identity and resolved revision, rather than
        # resolving the training model's potentially moving Hub branch again.
        test_args.model = None
        test_args.revision = None
        test_args.max_groups = None
        test_args.max_steps = None
        LOGGER.info("Training finished; evaluating saved adapter %s", test_args.adapter_dir)
        test_samples = load_samples(args.test_dir)
        evaluate(test_args, test_samples, dataset_summary(test_samples))
