"""Held-out inference, prediction records and metric-file output."""

from __future__ import annotations

import json
import time
from typing import Any

from .config import LABELS, LOGGER
from .data import Sample
from .encoding import question_batch
from .metrics import TestMetrics
from .modeling import forward_answer_logits, load_test_model, pool_options
from .prompts import STANCE_QUESTION, EVIDENCE_INSTRUCTIONS


def predict_sample(model, codec, encoded, pad_token_id, device, question_batch_size):
    """Predict without accessing EncodedQuestion.target or any gold label."""
    import torch

    label_probabilities = None
    sentence_probabilities = {}
    with torch.inference_mode():
        for start in range(0, len(encoded.questions), question_batch_size):
            questions = encoded.questions[start:start + question_batch_size]
            logits = forward_answer_logits(model, *question_batch(encoded, questions, pad_token_id, device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite prediction logits at {encoded.key}")
            for row, question in zip(logits, questions):
                token_ids = codec.stance_ids if question.is_stance else codec.evidence_ids
                probabilities = pool_options(row, token_ids).softmax(-1).tolist()
                if question.is_stance:
                    label_probabilities = probabilities
                else:
                    sentence_probabilities[question.sentence_key] = probabilities[0]
    if label_probabilities is None:
        raise ValueError(f"Missing stance prediction at {encoded.key}")
    return label_probabilities, sentence_probabilities


def prediction_record(sample, label_probabilities, sentence_probabilities, threshold):
    """Attach gold targets only AFTER inference, for auditing and scoring."""
    expected = {
        (a.doc_id, i) for a in sample.articles
        for i, available in enumerate(a.sentence_mask) if available
    }
    if set(sentence_probabilities) != expected:
        raise ValueError(f"Missing or unexpected sentence predictions for {sample.key}")
    predicted = max(range(len(LABELS)), key=lambda i: label_probabilities[i])
    articles = []
    for article in sample.articles:
        rows, selected, gold = [], [], []
        for index, text in enumerate(article.sentences):
            available = article.sentence_mask[index]
            probability = sentence_probabilities[(article.doc_id, index)] if available else None
            is_evidence = probability >= threshold if available else None
            target = article.evidence_targets[index] if available else None
            if is_evidence:
                selected.append(index)
            if target:
                gold.append(index)
            rows.append({
                "original_index": index, "sentence": text, "available": available,
                "gold_evidence": target, "predicted_evidence": is_evidence,
                "evidence_probability": probability,
            })
        articles.append({
            "doc_id": article.doc_id, "title": article.title,
            "gold_evidence_indices": gold, "predicted_evidence_indices": selected,
            "sentences": rows,
        })
    return {
        "sample_id": sample.key, "claim_id": sample.claim_id, "group_index": sample.group_index,
        "variant": sample.variant, "source_file": sample.source_file, "claim": sample.claim,
        "gold_label": sample.label, "predicted_label": LABELS[predicted],
        "label_probabilities": dict(zip(LABELS, label_probabilities)),
        "label_confidence": label_probabilities[predicted],
        "label_correct": LABELS[predicted] == sample.label,
        "articles": articles,
    }


def evaluate(args, samples: list[Sample], summary: dict) -> dict[str, Any]:
    output_dir = args.eval_output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Evaluation output directory is not empty: {output_dir}; choose a new --eval-output-dir")
    model, codec, pad_token_id, device, model_info = load_test_model(args)
    # Preflight every input: never report metrics on an unnoticed truncated or
    # skipped subset. Test mode deliberately has no --max-groups shortcut.
    encoded = [codec.encode(sample, args.max_length) for sample in samples]
    summary = {**summary, "max_question_tokens": max(
        len(s.prefix_ids) + len(q.suffix_ids) for s in encoded for q in s.questions
    )}
    config = {
        **model_info, "test_dir": str(args.test_dir.resolve()), "summary": summary,
        "evidence_threshold": args.evidence_threshold, "ece_bins": args.ece_bins,
        "max_length": args.max_length, "question_batch_size": args.question_batch_size,
        "stance_question": STANCE_QUESTION, "evidence_instructions": EVIDENCE_INSTRUCTIONS,
        "gold_evidence_used_as_input": False,
        "evidence_f1_conditioned_on_predicted_label": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metrics = TestMetrics(args.evidence_threshold, args.ece_bins)
    started = time.monotonic()
    partial = output_dir / "predictions.partial.jsonl"
    with partial.open("w", encoding="utf-8") as handle:
        for i, (sample, item) in enumerate(zip(samples, encoded), start=1):
            label_probs, sentence_probs = predict_sample(
                model, codec, item, pad_token_id, device, args.question_batch_size,
            )
            record = prediction_record(sample, label_probs, sentence_probs, args.evidence_threshold)
            valid_rows = [row for a in record["articles"] for row in a["sentences"] if row["available"]]
            metrics.update(
                sample.label, label_probs, [r["gold_evidence"] for r in valid_rows],
                [r["evidence_probability"] for r in valid_rows],
            )
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            if i == 1 or i % args.logging_steps == 0 or i == len(samples):
                LOGGER.info("Test progress: %d/%d samples", i, len(samples))
    result = metrics.result()
    if result["num_samples"] != len(samples) or result["num_valid_sentences"] != summary["valid_sentences"]:
        raise RuntimeError("Evaluation did not cover every sample and valid sentence")
    result.update({"elapsed_seconds": round(time.monotonic() - started, 2), "evaluation": config})
    partial.rename(output_dir / "predictions.jsonl")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("Saved metrics to %s", metrics_path)
    print(json.dumps({name: result[name] for name in (
        "label_accuracy", "label_macro_f1", "sentence_level_evidence_f1", "label_ece",
    )}, indent=2))
    return result
