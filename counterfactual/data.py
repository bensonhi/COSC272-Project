"""Counterfactual records (generate.py output) -> LiquidAI Samples for training and evaluation."""

from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

from LiquidAI.data import Sample, load_samples  # noqa: E402

CF_DIR = Path(__file__).resolve().parent / "data"
EDIT_TYPES = ("claim_negation", "evidence_flip", "evidence_weaken")


def load_records(split: str, verified_only: bool = True) -> list[dict]:
    path = CF_DIR / f"{split}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in records if r["verified"] or not verified_only]


def build(base: Sample, record: dict) -> Sample:
    """Apply one record to the claim's original sample."""
    evidence = set(record["evidence"])
    articles = []
    for a in base.articles:
        sentences = [record["edited_sentences"].get(f"{a.doc_id}-{i}", s) for i, s in enumerate(a.sentences)]
        if record["edit"] == "claim_negation":
            targets = list(a.evidence_targets)
        else:
            targets = [int(f"{a.doc_id}-{i}" in evidence) for i in range(len(sentences))]
        articles.append(replace(a, title=record["edited_titles"].get(str(a.doc_id), a.title),
                                sentences=sentences, evidence_targets=targets,
                                sentence_mask=[bool(s) for s in sentences]))
    return replace(base, claim=record["claim"], label=record["label"], articles=articles,
                   variant=record["edit"], group_index=1000 + EDIT_TYPES.index(record["edit"]))


def counterfactual_samples(split: str, data_dir: Path, edits=EDIT_TYPES) -> list[Sample]:
    """Edits of the claims in data_dir (records of claims not in data_dir, e.g. ones since moved to
    data/validation, are skipped)."""
    originals = {s.claim_id: s for s in load_samples(data_dir) if s.variant == "original"}
    return [build(originals[r["claim_id"]], r) for r in load_records(split)
            if r["edit"] in edits and r["claim_id"] in originals]
