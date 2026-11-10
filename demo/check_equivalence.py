"""Check that the demo's one-pass tree inference reproduces the evaluation path
(LiquidAI.evaluation.predict_sample) on real test claims, for each mounted model.

    CUDA_VISIBLE_DEVICES=4 python demo/check_equivalence.py --n 20 [--adapter name=PATH ...]
"""

import argparse
from pathlib import Path

from engine import ROOT, ZERO_SHOT, D1Verifier

from LiquidAI.data import load_samples
from LiquidAI.evaluation import predict_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--adapter", action="append", default=[], help="name=PATH")
    args = parser.parse_args()
    adapters = dict(item.split("=", 1) for item in args.adapter)
    engine = D1Verifier(adapters)
    pad = engine.tokenizer.pad_token_id
    samples = load_samples(ROOT / "data" / "test")[:args.n]
    for name in engine.model_names:
        worst_label = worst_sentence = 0.0
        flips = 0
        for sample in samples:
            articles = [{"doc_id": a.doc_id, "title": a.title, "sentences": a.sentences} for a in sample.articles]
            fast = engine.verify(sample.claim, articles, name)
            with engine._adapter(name):
                reference = predict_sample(engine.model, engine.codec, engine.codec.encode(sample, 8192),
                                           pad, engine.device, 8)
            worst_label = max(worst_label, *(abs(a - b) for a, b in zip(fast.label_probabilities.values(), reference[0])))
            worst_sentence = max(worst_sentence, *(abs(fast.sentence_probabilities[k] - v) for k, v in reference[1].items()))
            flips += max(range(3), key=reference[0].__getitem__) != list(fast.label_probabilities).index(fast.label)
        print(f"{name}: max |dP(label)|={worst_label:.4f}  max |dP(evidence)|={worst_sentence:.4f}  "
              f"argmax disagreements={flips}/{len(samples)}")


if __name__ == "__main__":
    main()
