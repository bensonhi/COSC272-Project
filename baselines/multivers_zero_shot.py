"""B4: MultiVerS (Wadden et al., 2022), `fever_sci` checkpoint, zero-shot on our test split.

`fever_sci` was trained on FEVER + weakly labelled PubMedQA / Evidence Inference and has
never seen SciFact. The official `scifact` checkpoint is NOT used: it was trained on the
SciFact train split, and 249 of our 333 test claims come from that split (leakage).

The original code pins torch 1.7 / transformers 4.2 / Lightning 1.2, which do not run on L40S.
This file re-implements its inference path on current transformers and loads the released
weights (encoder + label / rationale FeedForward heads) strictly:
  input  = <s> claim </s> title </s> sent_0 </s> ... sent_n </s>, global attention on the
           claim and every <s>/</s>; label from the pooled <s>; rationale for sentence i from
           [pooled; hidden state of the </s> closing sentence i].
Each cited abstract is scored separately; with several abstracts the claim label comes from
the abstract with the lowest P(NEI), and each sentence keeps its own abstract's P(rationale).

    wget -O $SCRATCH/ext/mv_ckpt/fever_sci.ckpt \
        https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/fever_sci.ckpt
"""

import argparse

from common import EVAL_SETS, SCRATCH, eval_samples, result_name, run_baseline

ENCODER = "allenai/longformer-large-4096"
ADDITIONAL_TOKENS = ["<|sec|>", "</|sec|>", "<|sec-title|>", "</|sec-title|>", "<|abs|>", "</|abs|>",
                     "<|title|>", "</|title|>", "<|sent|>", "<|par|>"]
CHECKPOINT_LABELS = ("CONTRADICT", "NEI", "SUPPORT")  # MultiVerS label ids 0, 1, 2


def feed_forward(torch, input_dim, hidden_dim, output_dim):
    # allennlp FeedForward(num_layers=2, activations=[GELU, Identity]); dropout is a no-op at eval.
    class FeedForward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._linear_layers = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, hidden_dim), torch.nn.Linear(hidden_dim, output_dim)])

        def forward(self, x):
            return self._linear_layers[1](torch.nn.functional.gelu(self._linear_layers[0](x)))

    return FeedForward()


def safe_load_state_dict(torch, checkpoint):
    """Load the Lightning checkpoint with torch's weights-only unpickler. Every non-tensor
    global it references (callbacks, hparams) is replaced by an inert stub class, so no code
    from the checkpoint can run; only the tensors in state_dict are used."""
    import pickle
    import re

    stubs = []
    for _ in range(50):
        try:
            with torch.serialization.safe_globals(stubs):
                return torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        except pickle.UnpicklingError as error:
            found = re.search(r"Unsupported global: GLOBAL ([\w.]+)", str(error))
            if not found:
                raise
            stubs.append((type("Stub", (), {"__setstate__": lambda self, state: None}), found.group(1)))
    raise RuntimeError("too many non-tensor globals in checkpoint")


def load_multivers(checkpoint, device):
    import torch
    from transformers import AutoConfig, AutoTokenizer, LongformerModel

    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    tokenizer.add_tokens(ADDITIONAL_TOKENS)
    model = torch.nn.Module()
    model.encoder = LongformerModel(AutoConfig.from_pretrained(ENCODER))
    model.encoder.resize_token_embeddings(len(tokenizer))
    hidden = model.encoder.config.hidden_size
    model.label_classifier = feed_forward(torch, hidden, hidden, 3)
    model.rationale_classifier = feed_forward(torch, 2 * hidden, hidden, 1)
    state = safe_load_state_dict(torch, checkpoint)
    state = {k: v for k, v in state.items() if not k.startswith("metrics")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # position_ids is a non-persistent buffer in current transformers; anything else is an error.
    missing = [k for k in missing if not k.endswith("position_ids")]
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), tokenizer


def tokenize_article(tokenizer, claim, article, device):
    """MultiVerS input for one (claim, abstract) pair; empty sentences are unavailable and skipped."""
    import torch

    eos = tokenizer.eos_token
    indices = [i for i, s in enumerate(article.sentences) if s]
    text = claim + eos + article.title + eos + eos.join(article.sentences[i] for i in indices)
    ids = torch.tensor([tokenizer(text)["input_ids"]], device=device)
    if ids.shape[1] > 4096:
        raise ValueError(f"article {article.doc_id}: {ids.shape[1]} tokens exceed Longformer's 4096")
    special = (ids == tokenizer.bos_token_id) | (ids == tokenizer.eos_token_id)
    is_claim = torch.arange(ids.shape[1], device=device)[None] < torch.nonzero(ids[0] == tokenizer.eos_token_id)[0]
    sentence_positions = torch.nonzero(ids[0] == tokenizer.eos_token_id).squeeze(1)[2:]
    assert len(sentence_positions) == len(indices)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
              "global_attention_mask": (special | is_claim).long()}
    return inputs, sentence_positions, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(SCRATCH / "ext" / "mv_ckpt" / "fever_sci.ckpt"))
    parser.add_argument("--weights", help="state dict saved by multivers_finetune.py (replaces fever_sci weights)")
    parser.add_argument("--name", default="b4_multivers_fever_sci")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-set", choices=EVAL_SETS, default="test")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda:0")
    model, tokenizer = load_multivers(args.checkpoint, device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

    @torch.inference_mode()
    def score_article(claim, article):
        inputs, sentence_positions, indices = tokenize_article(tokenizer, claim, article, device)
        out = model.encoder(**inputs)
        pooled = out.pooler_output
        label = model.label_classifier(pooled).float().softmax(-1)[0].tolist()
        states = out.last_hidden_state[0, sentence_positions]
        rationale = model.rationale_classifier(torch.cat([pooled.expand_as(states), states], -1))
        by_name = dict(zip(CHECKPOINT_LABELS, label))
        probs = [by_name["NEI"], by_name["SUPPORT"], by_name["CONTRADICT"]]  # LABELS order
        return probs, dict(zip(((article.doc_id, i) for i in indices), rationale.float().sigmoid()[:, 0].tolist()))

    def predict(sample):
        scored = [score_article(sample.claim, a) for a in sample.articles]
        evidence = {k: v for _, sentences in scored for k, v in sentences.items()}
        return min((probs for probs, _ in scored), key=lambda p: p[0]), evidence

    run_baseline(result_name(args.name, args.eval_set), predict, {
        "description": ("MultiVerS fine-tuned on our training split from fever_sci" if args.weights else
                        "MultiVerS fever_sci checkpoint (no SciFact training), re-implemented inference"),
        "checkpoint": "https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/fever_sci.ckpt",
        "weights": args.weights, "encoder": ENCODER, "dtype": "float32",
        "multi_abstract_aggregation": "lowest P(NEI)", "gpu": torch.cuda.get_device_name(device),
    }, samples=eval_samples(args.eval_set), sync=torch.cuda.synchronize, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
