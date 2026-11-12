"""JSON loading, original-index alignment, evidence targets and sentence masks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import LABEL_TO_ID


@dataclass
class Article:
    doc_id: int
    title: str
    sentences: list[str]
    evidence_targets: list[int]
    sentence_mask: list[bool]


@dataclass
class Sample:
    claim_id: int
    group_index: int
    variant: str
    claim: str
    label: str
    articles: list[Article]
    source_file: str

    @property
    def key(self) -> str:
        return f"{self.claim_id}/group-{self.group_index}/{self.variant}"

    def state(self) -> dict[str, Any]:
        # No gold labels, evidence records, deletion reasons or variant names
        # enter the prompt. Explicit original indices avoid implicit renumbering.
        return {
            "claim": self.claim,
            "articles": [
                {
                    "doc_id": a.doc_id,
                    "title": a.title,
                    "sentences": {str(i): s for i, s in enumerate(a.sentences)},
                }
                for a in self.articles
            ],
        }


def integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer, got {value!r}")
    return value


def record_key(row: dict) -> tuple[int, int]:
    return integer(row["doc_id"], "doc_id"), integer(row["original_index"], "original_index")


def samples_from_record(record: dict, source_file: str = "<memory>") -> list[Sample]:
    claim_id = integer(record["id"], "claim id")
    if "groups" in record:
        groups = record["groups"]
    else:
        # Test files are flat, unmodified examples. Their evidence has no kept
        # field: all listed evidence must be present. Do not infer its text or
        # silently discard an annotation whose sentence is missing.
        group = dict(record)
        group["evidence"] = [{"kept": True, **e} for e in record.get("evidence", [])]
        group.setdefault("variant", "original")
        groups = [group]
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list")

    # Use every variant's index metadata to recover trailing deleted slots too.
    # This only recovers array lengths; removed text is never restored.
    lengths: dict[int, int] = {}
    for group in groups:
        for article in group["articles"]:
            doc_id = integer(article["doc_id"], "doc_id")
            lengths.setdefault(doc_id, 0)
            for row in article["sentences"]:
                index = integer(row["original_index"], "original_index")
                lengths[doc_id] = max(lengths[doc_id], index + 1)
        for row in group.get("deleted_sentences", []) + group.get("evidence", []):
            doc_id, index = record_key(row)
            lengths[doc_id] = max(lengths.get(doc_id, 0), index + 1)

    samples = []
    for group_index, group in enumerate(groups):
        label = group["label"]
        if label not in LABEL_TO_ID:
            raise ValueError(f"group {group_index}: unknown label {label!r}")
        if not isinstance(group["claim"], str) or not group["claim"].strip():
            raise ValueError(f"group {group_index}: claim must be nonempty text")
        deleted = {record_key(row) for row in group.get("deleted_sentences", [])}
        articles: dict[int, Article] = {}
        for raw_article in group["articles"]:
            doc_id = integer(raw_article["doc_id"], "doc_id")
            if not isinstance(raw_article["title"], str):
                raise ValueError(f"article {doc_id}: title must be a string")
            sentences = [""] * lengths[doc_id]
            seen = set()
            for row in raw_article["sentences"]:
                index = integer(row["original_index"], "original_index")
                if index in seen:
                    raise ValueError(f"article {doc_id}: duplicate original_index {index}")
                seen.add(index)
                text = row["sentence"]
                if not isinstance(text, str):
                    raise ValueError(f"article {doc_id}, sentence {index}: expected text")
                if (doc_id, index) in deleted and text.strip():
                    raise ValueError(f"deleted sentence {(doc_id, index)} is still present")
                sentences[index] = text if text.strip() else ""
            article = Article(
                doc_id, raw_article["title"], sentences,
                [0] * len(sentences), [bool(s) for s in sentences],
            )
            # Some source claims cite an identical article twice. Deduplicate
            # those entries, but never silently combine conflicting versions.
            if doc_id in articles and articles[doc_id] != article:
                raise ValueError(f"group {group_index}: conflicting duplicate article {doc_id}")
            articles[doc_id] = article

        for row in group.get("evidence", []):
            doc_id, index = record_key(row)
            if doc_id not in articles:
                raise ValueError(f"evidence references missing article {doc_id}")
            article = articles[doc_id]
            kept = row["kept"]
            if not isinstance(kept, bool):
                raise ValueError("evidence.kept must be a boolean")
            if kept != article.sentence_mask[index]:
                raise ValueError(f"evidence.kept conflicts with available text at {(doc_id, index)}")
            if kept:
                evidence_label = row["evidence_label"].removesuffix("_more")
                if evidence_label != label:
                    raise ValueError(f"evidence label {evidence_label} conflicts with group label {label}")
                article.evidence_targets[index] = 1

        if any(doc_id not in articles for doc_id, _ in deleted):
            raise ValueError("deleted_sentences references an absent article")
        positive_count = sum(sum(a.evidence_targets) for a in articles.values())
        if label != "NEI" and not positive_count:
            raise ValueError(f"group {group_index}: {label} has no available evidence")
        if label == "NEI" and positive_count:
            raise ValueError(f"group {group_index}: NEI has positive evidence targets")
        samples.append(Sample(
            claim_id, group_index, group.get("variant", str(group_index)),
            group["claim"], label, list(articles.values()), source_file,
        ))
    return samples


def load_samples(data_dir: Path, max_groups: int | None = None, *, original_only: bool = False) -> list[Sample]:
    """Load groups, optionally keeping original variants before max_groups."""
    files = sorted(data_dir.glob("*.json"), key=lambda p: (len(p.stem), p.stem))
    if not files:
        raise ValueError(f"No JSON data files in {data_dir}")
    samples, claim_ids = [], set()
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["id"] in claim_ids:
                raise ValueError(f"duplicate claim id {record['id']}")
            claim_ids.add(record["id"])
            groups = samples_from_record(record, str(path))
            samples.extend(sample for sample in groups if not original_only or sample.variant == "original")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if max_groups is not None and len(samples) >= max_groups:
            return samples[:max_groups]
    if original_only and not samples:
        raise ValueError(f"No groups with variant='original' in {data_dir}")
    return samples


def dataset_summary(samples: list[Sample]) -> dict[str, Any]:
    valid = sum(sum(a.sentence_mask) for s in samples for a in s.articles)
    positive = sum(sum(a.evidence_targets) for s in samples for a in s.articles)
    slots = sum(len(a.sentences) for s in samples for a in s.articles)
    return {
        "claims": len({s.claim_id for s in samples}),
        "groups": len(samples),
        "labels": dict(Counter(s.label for s in samples)),
        "valid_sentences": valid,
        "positive_sentences": positive,
        "negative_sentences": valid - positive,
        "masked_sentence_slots": slots - valid,
        "max_articles_per_group": max(len(s.articles) for s in samples),
    }
