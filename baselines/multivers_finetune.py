"""B4b: MultiVerS fine-tuned on OUR training split (proposal: "MultiVerS ... using a training
partition that excludes our evaluation claims").

Starts from the `fever_sci` checkpoint (no SciFact) and fine-tunes on the unmodified training
groups (variant == "original"), one (claim, cited abstract) pair per instance as in MultiVerS:
the pair's label is the claim label if that abstract holds gold evidence, else NEI; rationale
targets are the gold evidence sentences. Loss = CE(label) + 15 * mean BCE(rationales), the
MultiVerS defaults (rationale_weight=15). Hyperparameters are fixed in advance; the test split
is not used for any choice. Evaluate with multivers_zero_shot.py --weights OUT/model.pt.
"""

import argparse
import json
import math
import random
import time

from common import SCRATCH, TRAIN_DIR, load_samples

from multivers_zero_shot import CHECKPOINT_LABELS, load_multivers, tokenize_article


def instances(samples):
    out = []
    for s in samples:
        for a in s.articles:
            has_evidence = any(a.evidence_targets)
            label = s.label if has_evidence else "NEI"
            indices = [i for i, ok in enumerate(a.sentence_mask) if ok]
            out.append((s.claim, a, CHECKPOINT_LABELS.index(label), [a.evidence_targets[i] for i in indices]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(SCRATCH / "ext" / "mv_ckpt" / "fever_sci.ckpt"))
    parser.add_argument("--output-dir", default=str(SCRATCH / "outputs" / "multivers-finetuned-seed42"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--rationale-weight", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from transformers import get_linear_schedule_with_warmup

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    model, tokenizer = load_multivers(args.checkpoint, device)
    model.train()
    data = instances([s for s in load_samples(TRAIN_DIR) if s.variant == "original"])
    steps = math.ceil(len(data) / args.accumulate) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    log = (out / "train_log.jsonl").open("w")
    started, step = time.monotonic(), 0
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        random.Random(args.seed + epoch).shuffle(order)
        running = []
        for n, index in enumerate(order, start=1):
            claim, article, label, rationale = data[index]
            inputs, positions, _ = tokenize_article(tokenizer, claim, article, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = model.encoder(**inputs)
                pooled = encoded.pooler_output
                label_logits = model.label_classifier(pooled).float()
                states = encoded.last_hidden_state[0, positions]
                rationale_logits = model.rationale_classifier(torch.cat([pooled.expand_as(states), states], -1)).float()[:, 0]
            loss = F.cross_entropy(label_logits, torch.tensor([label], device=device)) + args.rationale_weight * \
                F.binary_cross_entropy_with_logits(rationale_logits, torch.tensor(rationale, dtype=torch.float, device=device))
            (loss / args.accumulate).backward()
            running.append(loss.item())
            if n % args.accumulate == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    record = {"step": step, "epoch": epoch + 1, "loss": sum(running) / len(running),
                              "elapsed": round(time.monotonic() - started)}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(record, flush=True)
                    running = []
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps({**vars(args), "instances": len(data), "optimizer_steps": step}, indent=2))
    print("saved", out / "model.pt")


if __name__ == "__main__":
    main()
