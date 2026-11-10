"""B3: zero-shot NLI cross-encoder (DeBERTa-v3-large trained on MNLI/FEVER/ANLI/LingNLI/WANLI).

No SciFact data is used. Mapping: entailment -> SUPPORT, contradiction -> CONTRADICT,
neutral -> NEI.

Evidence: every available sentence is a premise for the claim; P(evidence) = P(entail) + P(contra).
Label, two variants (--label-source):
  abstract  premise = title + full abstract (truncated to 512 tokens); with several cited
            abstracts, use the one with the lowest P(neutral).
  sentence  use the single sentence with the lowest P(neutral) across all cited abstracts.
"""

import argparse

from common import EVAL_SETS, LABELS, eval_samples, result_name, run_baseline, sentence_keys

MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-source", choices=("abstract", "sentence"), default="abstract")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-set", choices=EVAL_SETS, default="test")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    index = {name.lower(): i for i, name in model.config.id2label.items()}
    order = [index["neutral"], index["entailment"], index["contradiction"]]  # = LABELS order
    assert LABELS == ("NEI", "SUPPORT", "CONTRADICT")

    @torch.inference_mode()
    def nli(premises, claim):
        out = []
        for start in range(0, len(premises), args.batch_size):
            chunk = premises[start:start + args.batch_size]
            batch = tokenizer(chunk, [claim] * len(chunk), truncation="only_first", max_length=512,
                              padding=True, return_tensors="pt").to(device)
            probs = model(**batch).logits.float().softmax(-1)[:, order]
            out.extend(probs.tolist())
        return out  # rows: [P(NEI), P(SUPPORT), P(CONTRADICT)]

    def predict(sample):
        keys = sentence_keys(sample)
        texts = {a.doc_id: a.sentences for a in sample.articles}
        sentence_probs = nli([texts[d][i] for d, i in keys], sample.claim)
        evidence = {key: p[1] + p[2] for key, p in zip(keys, sentence_probs)}
        if args.label_source == "sentence":
            label = min(sentence_probs, key=lambda p: p[0])
        else:
            premises = [" ".join([a.title] + [s for s in a.sentences if s]) for a in sample.articles]
            label = min(nli(premises, sample.claim), key=lambda p: p[0])
        total = sum(label)
        return [p / total for p in label], evidence

    run_baseline(result_name(f"b3_nli_zero_shot_{args.label_source}", args.eval_set), predict, {
        "description": "zero-shot DeBERTa-v3-large NLI; entail->SUPPORT, contra->CONTRADICT, neutral->NEI",
        "model": MODEL, "label_source": args.label_source, "dtype": "float16", "max_length": 512,
        "gpu": torch.cuda.get_device_name(device),
    }, samples=eval_samples(args.eval_set), sync=torch.cuda.synchronize, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
