"""B2: "Original-data SFT" from the proposal's Table 1 — d1-3B + LoRA trained only on the
unmodified training groups (variant == "original", 776 claims), no evidence-deletion copies.

It calls the LiquidAI training code unchanged with the same defaults as the augmented run;
the only difference is the sample filter. Extra arguments are passed through to
LiquidAI.cli (e.g. --seed, --epochs). Evaluate the result with d1_adapter_eval.py.

    CUDA_VISIBLE_DEVICES=4 python d1_original_sft.py --seed 42
"""

import sys

from common import SCRATCH, dataset_summary, load_samples

from LiquidAI.cli import parse_args
from LiquidAI.training import train


def main():
    import logging

    argv = sys.argv[1:]
    if "--output-dir" not in argv:
        seed = argv[argv.index("--seed") + 1] if "--seed" in argv else "42"
        argv += ["--output-dir", str(SCRATCH / "outputs" / f"d1-original-sft-seed{seed}")]
    args = parse_args(["--mode", "train", "--device", "0", *argv])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = [s for s in load_samples(args.train_dir, args.max_groups) if s.variant == "original"]
    summary = dataset_summary(samples)
    summary["filter"] = "variant == original (no evidence-deletion augmentation)"
    train(args, samples, summary)


if __name__ == "__main__":
    main()
