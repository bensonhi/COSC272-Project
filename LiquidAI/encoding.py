"""Native d1 prompt encoding and variable-length question batches."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

from .config import LABEL_TO_ID
from .data import Sample
from .prompts import STANCE_QUESTION, evidence_question


@dataclass
class EncodedQuestion:
    suffix_ids: list[int]
    target: int
    is_stance: bool
    # Retained for tracing predictions back to the original articles later.
    sentence_key: tuple[int, int] | None = None


@dataclass
class EncodedSample:
    key: str
    prefix_ids: list[int]
    questions: list[EncodedQuestion]


class DecisionCodec:
    """Use the checkpoint's own prompts and exact answer-token aliases."""

    def __init__(self, base_model, tokenizer):
        package = base_model.__class__.__module__.rsplit(".", 1)[0]
        self.prompt = importlib.import_module(package + ".prompt")
        self.tokenizer = tokenizer
        self.lead = self.prompt.default_lead(base_model.config.model_type)
        stance = self.prompt.as_question(STANCE_QUESTION)
        evidence = self.prompt.as_question(evidence_question(0, 0))
        self.stance_ids = self.prompt.readout_ids(tokenizer, stance)
        self.evidence_ids = self.prompt.readout_ids(tokenizer, evidence)  # [yes forms, no forms]
        if len(self.stance_ids) != 3 or len(self.evidence_ids) != 2:
            raise ValueError("Unexpected d1 decision readout schema")

    def encode(self, sample: Sample, max_length: int) -> EncodedSample:
        tokenizer, prompt = self.tokenizer, self.prompt
        prefix = prompt.prefix_text(tokenizer, sample.state(), bos=tokenizer.bos_token or "")
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        questions = []

        def add(schema, target, is_stance, sentence_key=None):
            suffix = prompt.suffix_text(tokenizer, prompt.as_question(schema), lead=self.lead)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            length = len(prefix_ids) + len(suffix_ids)
            if length > max_length:
                raise ValueError(
                    f"{sample.key} needs {length} tokens (limit {max_length}). "
                    "Increase --max-length within the model context limit. "
                    "Inputs are never silently truncated and labels are never changed."
                )
            questions.append(EncodedQuestion(suffix_ids, target, is_stance, sentence_key))

        add(STANCE_QUESTION, LABEL_TO_ID[sample.label], True)
        for article in sample.articles:
            for original_index, available in enumerate(article.sentence_mask):
                if available:
                    add(
                        evidence_question(article.doc_id, original_index),
                        article.evidence_targets[original_index], False,
                        (article.doc_id, original_index),
                    )
        return EncodedSample(sample.key, prefix_ids, questions)


def question_batch(sample: EncodedSample, questions: list[EncodedQuestion], pad_token_id: int, device):
    import torch

    rows = [sample.prefix_ids + q.suffix_ids for q in questions]
    lengths = torch.tensor([len(row) for row in rows], device=device)
    width = max(map(len, rows))
    # d1's hybrid stack expects RIGHT padding. Causal attention and convolution
    # prevent these pads from affecting the preceding real tokens.
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        input_ids[i, :len(row)] = torch.tensor(row, device=device)
    mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
    return input_ids, mask, lengths - 1
