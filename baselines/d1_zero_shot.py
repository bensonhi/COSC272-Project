"""B1: zero-shot d1-3B, no LoRA.

Same native choice/noul questions, prompt encoding and readout as the fine-tuned models
(LiquidAI.prompts / encoding / evaluation.predict_sample), so the only difference from the
SFT configurations is the absence of an adapter. Run with CUDA_VISIBLE_DEVICES set to one GPU.
"""

import argparse

from common import EVAL_SETS, eval_samples, result_name, run_baseline

from LiquidAI.config import DEFAULT_MODEL
from LiquidAI.encoding import DecisionCodec
from LiquidAI.evaluation import predict_sample


def load_base(model_id=DEFAULT_MODEL, device="cuda:0"):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.to(device).eval().requires_grad_(False)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.config.text_config.pad_token_id
    return model, tokenizer, pad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--question-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-set", choices=EVAL_SETS, default="test")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda:0")
    model, tokenizer, pad = load_base(args.model, device)
    codec = DecisionCodec(model, tokenizer)

    def predict(sample):
        encoded = codec.encode(sample, args.max_length)
        return predict_sample(model, codec, encoded, pad, device, args.question_batch_size)

    run_baseline(result_name("b1_d1_zero_shot", args.eval_set), predict, {
        "description": "d1-3B without adapter, native prompts identical to the SFT runs",
        "model": args.model, "resolved_revision": getattr(model.config, "_commit_hash", None),
        "dtype": "bfloat16", "question_batch_size": args.question_batch_size,
        "gpu": torch.cuda.get_device_name(device),
    }, samples=eval_samples(args.eval_set), sync=torch.cuda.synchronize, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
