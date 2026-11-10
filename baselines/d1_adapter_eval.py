"""Evaluate any d1-3B LoRA adapter (original-only SFT, or the team's augmented SFT) with the
same harness, metrics and latency measurement as the other baselines.

    CUDA_VISIBLE_DEVICES=4 python d1_adapter_eval.py --adapter-dir PATH/final --name b2_d1_original_sft_seed42
"""

import argparse

from common import EVAL_SETS, eval_samples, result_name, run_baseline

from LiquidAI.cli import parse_args
from LiquidAI.encoding import DecisionCodec  # noqa: F401  (load_test_model builds it)
from LiquidAI.evaluation import predict_sample
from LiquidAI.modeling import load_test_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--question-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-set", choices=EVAL_SETS, default="test")
    args = parser.parse_args()

    import torch

    model_args = parse_args(["--mode", "test", "--device", "0", "--adapter-dir", args.adapter_dir,
                             "--max-length", str(args.max_length)])
    model, codec, pad, device, info = load_test_model(model_args)

    def predict(sample):
        encoded = codec.encode(sample, args.max_length)
        return predict_sample(model, codec, encoded, pad, device, args.question_batch_size)

    run_baseline(result_name(args.name, args.eval_set), predict, {
        "description": "d1-3B + LoRA adapter, native prompts", **info,
        "question_batch_size": args.question_batch_size, "gpu": torch.cuda.get_device_name(device),
    }, samples=eval_samples(args.eval_set), sync=torch.cuda.synchronize, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
