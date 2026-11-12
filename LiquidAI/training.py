"""LoRA optimization, gradient accumulation, logging and checkpoints."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import time

from .config import ROOT, LABELS, LABEL_TO_ID, LOGGER
from .data import Sample, load_samples, dataset_summary
from .encoding import DecisionCodec, question_batch
from .evaluation import evaluate_model
from .losses import joint_loss, pair_hinge_loss
from .modeling import attach_lora, forward_answer_logits, load_train_model, pool_options, save_adapter
from .pairing import build_evidence_pairs
from .prompts import STANCE_QUESTION, EVIDENCE_INSTRUCTIONS


def train(args, samples: list[Sample], summary: dict) -> None:
    pairs = build_evidence_pairs(samples) if args.loss == "pair" else {}
    if args.loss == "pair":
        summary["evidence_pairs"] = len(pairs)
    # Always evaluate complete held-out splits, including in smoke runs with a
    # training-only --max-groups/--max-steps limit. Fail before loading the model
    # if either split is missing or invalid.
    evaluation_sets = {
        split: load_samples(getattr(args, f"{split}_dir"))
        for split in ("validation", "test")
    }
    try:
        import torch
        import transformers
        from transformers import get_linear_schedule_with_warmup
        import peft
    except ImportError as exc:
        raise RuntimeError(f"Install training dependencies: python -m pip install -r {ROOT / 'requirements.txt'}") from exc

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}; choose a new --output-dir")
    base_model, tokenizer, pad_token_id, device, dtype_name = load_train_model(args)

    codec = DecisionCodec(base_model, tokenizer)
    LOGGER.info("Encoding %d groups; retaining every available sentence", len(samples))
    encoded = [codec.encode(sample, args.max_length) for sample in samples]
    summary["max_question_tokens"] = max(
        len(s.prefix_ids) + len(q.suffix_ids) for s in encoded for q in s.questions
    )
    summary["decision_questions"] = sum(len(s.questions) for s in encoded)
    evaluation_inputs = {
        split: [codec.encode(sample, args.max_length) for sample in held_out]
        for split, held_out in evaluation_sets.items()
    }
    evaluation_summaries = {split: dataset_summary(held_out) for split, held_out in evaluation_sets.items()}
    model = attach_lora(base_model, args).to(device)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(encoded) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps,
    )
    counts = Counter(s.label for s in samples)
    label_weights = None
    if args.balance_labels:
        present_classes = len(counts)
        label_weights = torch.tensor(
            [len(samples) / (present_classes * counts[label]) if counts[label] else 0.0 for label in LABELS],
            dtype=torch.float32, device=device,
        )
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "base_model": args.model,
        "resolved_revision": getattr(base_model.config, "_commit_hash", None),
        "dtype": dtype_name,
        "device": str(device),
        "labels": list(LABELS),
        "stance_question": STANCE_QUESTION,
        "evidence_instructions": EVIDENCE_INSTRUCTIONS,
        "sentence_index": "original_index (zero-based), scoped by doc_id",
        "state_format": "claim + articles[{doc_id, title, sentences: {original_index: text}}]",
        "loss": "CE(stance) + evidence_loss_weight * mean_valid_sentences(BCEWithLogits(yes-no))",
        "label_weights": label_weights.tolist() if label_weights is not None else None,
        "summary": summary,
        "evaluation_summaries": evaluation_summaries,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
    }
    if pairs:
        metadata["loss"] += " + pair_loss_weight * mean_pairs(hinge(pair_margin - (margin_original - margin_deleted)))"
        metadata["pairing"] = {
            "versions": ["original", "drop_all_evidence"],
            "count": len(pairs),
            "margin": "log P(original_label) - log P(NEI), using the original label for both versions",
            "normalization": "Each pair is evaluated at its original sample; its loss is scaled by num_samples / num_pairs / window_groups",
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Training summary: %s", json.dumps(summary))
    LOGGER.info("Planned optimizer updates: %d; evaluating validation and test after each epoch", total_steps)
    started, global_step = time.monotonic(), 0
    with (args.output_dir / "train_metrics.jsonl").open("w", encoding="utf-8") as log, \
         (args.output_dir / "epoch_metrics.jsonl").open("w", encoding="utf-8") as epoch_log:
        for epoch in range(args.epochs):
            order = list(range(len(encoded)))
            random.Random(args.seed + epoch).shuffle(order)
            for window_start in range(0, len(order), args.gradient_accumulation_steps):
                window = order[window_start:window_start + args.gradient_accumulation_steps]
                optimizer.zero_grad(set_to_none=True)
                totals = [0.0, 0.0, 0.0]
                for index in window:
                    sample = encoded[index]
                    evidence_count = len(sample.questions) - 1
                    for start in range(0, len(sample.questions), args.question_batch_size):
                        questions = sample.questions[start:start + args.question_batch_size]
                        inputs = question_batch(sample, questions, pad_token_id, device)
                        logits = forward_answer_logits(model, *inputs)
                        losses = joint_loss(
                            logits, questions, codec, evidence_count,
                            args.evidence_loss_weight, args.evidence_pos_weight, label_weights,
                        )
                        if not torch.isfinite(losses[0]):
                            raise FloatingPointError(f"Non-finite training loss at {sample.key}")
                        (losses[0] / len(window)).backward()
                        for j, loss in enumerate(losses):
                            totals[j] += loss.detach().item() / len(window)
                        del inputs, logits, losses
                pair_sum, pair_count, pair_contribution = 0.0, 0, 0.0
                for sufficient_index in window:
                    if sufficient_index not in pairs:
                        continue
                    scores = []
                    # Separate stance-only forwards retain gradients for both
                    # inputs, even if they occur in different CE windows. Each
                    # pair is visited once per epoch, at its original sample.
                    for index in (sufficient_index, pairs[sufficient_index]):
                        sample = encoded[index]
                        questions = [q for q in sample.questions if q.is_stance]
                        inputs = question_batch(sample, questions, pad_token_id, device)
                        logits = forward_answer_logits(model, *inputs)
                        scores.append(pool_options(logits, codec.stance_ids)[0])
                        del inputs, logits
                    pair_loss = pair_hinge_loss(
                        scores[0], scores[1], LABEL_TO_ID[samples[sufficient_index].label], args.pair_margin,
                    )
                    if not torch.isfinite(pair_loss):
                        raise FloatingPointError(f"Non-finite paired loss at {encoded[sufficient_index].key}")
                    # Scale from mean-over-groups to mean-over-pairs. This keeps
                    # the pair weight independent of the number of unpaired
                    # groups, and handles the final partial window exactly.
                    contribution = pair_loss * (args.pair_loss_weight * len(encoded) / len(pairs) / len(window))
                    contribution.backward()
                    pair_sum += pair_loss.detach().item()
                    pair_count += 1
                    pair_contribution += contribution.detach().item()
                    del scores, pair_loss, contribution
                totals[0] += pair_contribution
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                global_step += 1
                metrics = {
                    "step": global_step, "epoch": epoch + 1,
                    "groups": len(window), "loss": totals[0],
                    "stance_loss": totals[1], "evidence_loss": totals[2],
                    "grad_norm": float(grad_norm), "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
                if pairs:
                    metrics.update({
                        "pairs": pair_count,
                        "pair_loss": pair_sum / pair_count if pair_count else 0.0,
                        "pair_loss_contribution": pair_contribution,
                    })
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                if global_step == 1 or global_step % args.logging_steps == 0 or global_step == total_steps:
                    LOGGER.info("step %d/%d loss=%.4f stance=%.4f evidence=%.4f", global_step, total_steps, *totals)
                    if pairs:
                        LOGGER.info("  pairs=%d hinge=%.4f weighted_contribution=%.4f", pair_count, metrics["pair_loss"], pair_contribution)
                if global_step >= total_steps:
                    break
            epoch_complete = window_start + len(window) == len(order)
            metadata.update({"global_step": global_step, "last_epoch": epoch + 1,
                             "epoch_complete": epoch_complete})
            checkpoint_dir = args.output_dir / f"checkpoint-step-{global_step}"
            # A step limit may end training partway through an epoch. Only
            # completed epochs produce checkpoints; final is still saved below.
            if epoch_complete:
                save_adapter(model, tokenizer, checkpoint_dir, metadata)
            optimizer.zero_grad(set_to_none=True)
            epoch_metrics = {"epoch": epoch + 1, "step": global_step, "epoch_complete": epoch_complete}
            model_info = {
                "base_model": args.model,
                "adapter_dir": str(checkpoint_dir.resolve()) if epoch_complete else None,
                "epoch_complete": epoch_complete,
                "resolved_revision": metadata["resolved_revision"], "dtype": dtype_name,
                "device": str(device),
            }
            for split, held_out in evaluation_sets.items():
                epoch_metrics[split] = evaluate_model(
                    args, held_out, evaluation_summaries[split], model, codec, pad_token_id, device, model_info,
                    output_dir=args.output_dir / f"epoch-{epoch + 1}" / split,
                    split=split, encoded=evaluation_inputs[split], epoch=epoch + 1, step=global_step,
                )
            epoch_log.write(json.dumps(epoch_metrics, allow_nan=False) + "\n")
            epoch_log.flush()
            if global_step >= total_steps:
                break
    save_adapter(model, tokenizer, args.output_dir / "final", metadata)
    LOGGER.info("Saved trained LoRA adapter and tokenizer to %s", args.output_dir / "final")
