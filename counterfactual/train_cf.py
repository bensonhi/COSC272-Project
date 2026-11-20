"""Train d1-3B + LoRA on the team's augmented training groups plus verified counterfactual pairs.

    CUDA_VISIBLE_DEVICES=4 python counterfactual/train_cf.py --seed 42 [--edits claim_negation evidence_flip]

Calls the LiquidAI training code unchanged, with the same defaults as the augmented-SFT run
(question batch 2, 3 epochs, lr 2e-4, r16); the only difference is the extra groups. Extra
arguments go to LiquidAI.cli. Evaluate with baselines/d1_adapter_eval.py.
"""

import argparse
import logging
import os
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent)]

from data import EDIT_TYPES, ROOT, counterfactual_samples  # noqa: E402

from LiquidAI.cli import parse_args  # noqa: E402
from LiquidAI.data import dataset_summary, load_samples  # noqa: E402
from LiquidAI.training import train  # noqa: E402

SCRATCH = Path(os.environ.get("COSC272_SCRATCH", "/scratch/f008k3y/cosc272"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edits", nargs="+", choices=EDIT_TYPES, default=list(EDIT_TYPES))
    parser.add_argument("--seed", default="42")
    parser.add_argument("--output-dir")
    known, rest = parser.parse_known_args()
    tag = "all" if set(known.edits) == set(EDIT_TYPES) else "+".join(known.edits)
    output = known.output_dir or str(SCRATCH / "outputs" / f"d1-cf-{tag}-seed{known.seed}")
    args = parse_args(["--mode", "train", "--device", "0", "--seed", known.seed, "--question-batch-size", "2",
                       "--output-dir", output, *rest])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = load_samples(args.train_dir, args.max_groups)
    extra = counterfactual_samples("train", args.train_dir, known.edits)
    summary = dataset_summary(samples + extra)
    summary["counterfactual_groups"] = {e: sum(s.variant == e for s in extra) for e in known.edits}
    train(args, samples + extra, summary)


if __name__ == "__main__":
    main()
