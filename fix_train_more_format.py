"""Align claims_train_more.jsonl with two format rules.

1. When evidence is non-empty, cited_doc_ids lists exactly those abstracts,
   as integers, in evidence key order. Extra ids are removed. When evidence
   is empty, the NEI case, cited_doc_ids is copied from claims_train.jsonl
   for the same claim id and is not cleared.
2. Each rationale lists one sentence index. A rationale with several indices
   is split into one rationale per index, and the label is copied unchanged.

The script rewrites data/claims_train_more.jsonl in place. Running it again
leaves an already corrected file unchanged.

    .venv/bin/python fix_train_more_format.py
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "claims_train_more.jsonl"
TRAIN_PATH = DATA_DIR / "claims_train.jsonl"


def load_jsonl(path):
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                rows.append((line_number, json.loads(line)))
    return rows


def align_cited_doc_ids(evidence):
    cited = []
    for doc_id in evidence:
        if not str(doc_id).isdigit():
            raise ValueError(f"evidence doc id is not an integer: {doc_id}")
        cited.append(int(doc_id))
    return cited


def split_annotations(annotations):
    split = []
    for annotation in annotations:
        label = annotation["label"]
        sentences = annotation["sentences"]
        if not sentences:
            raise ValueError(f"rationale has no sentences: {annotation}")
        for index in sentences:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(f"sentence index is not an int: {index}")
            split.append({"sentences": [index], "label": label})
    return split


def cited_doc_ids_of(row):
    cited = []
    for doc_id in row.get("cited_doc_ids") or []:
        if isinstance(doc_id, bool) or not isinstance(doc_id, int):
            raise ValueError(f"cited doc id is not an int: {doc_id}")
        cited.append(doc_id)
    return cited


def fix_row(row, train_row):
    if row["claim"] != train_row["claim"]:
        raise ValueError("claim text does not match claims_train.jsonl")

    evidence = {}
    for doc_id, annotations in (row.get("evidence") or {}).items():
        evidence[str(doc_id)] = split_annotations(annotations)

    train_evidence = train_row.get("evidence") or {}
    if bool(evidence) != bool(train_evidence):
        raise ValueError("empty-evidence status does not match claims_train.jsonl")

    if evidence:
        cited = align_cited_doc_ids(evidence)
    else:
        cited = cited_doc_ids_of(train_row)

    return {
        "id": row["id"],
        "claim": row["claim"],
        "evidence": evidence,
        "cited_doc_ids": cited,
    }


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    train_by_id = {row["id"]: row for _, row in load_jsonl(TRAIN_PATH)}
    original = load_jsonl(DATA_PATH)
    fixed = []
    cited_changed = 0
    nei_restored = 0
    rationales_before = 0
    rationales_after = 0

    for line_number, row in original:
        train_row = train_by_id.get(row.get("id"))
        if train_row is None:
            raise ValueError(f"line {line_number} id {row.get('id')} is missing from {TRAIN_PATH.name}")
        try:
            updated = fix_row(row, train_row)
        except ValueError as error:
            raise ValueError(f"line {line_number} id {row.get('id')}: {error}") from error

        old_evidence = row.get("evidence") or {}
        for annotations in old_evidence.values():
            rationales_before += len(annotations)
        for annotations in updated["evidence"].values():
            rationales_after += len(annotations)

        old_cited = cited_doc_ids_of(row)
        if old_cited != updated["cited_doc_ids"]:
            cited_changed += 1
            if not updated["evidence"]:
                nei_restored += 1
        fixed.append(updated)

    write_jsonl(DATA_PATH, fixed)
    print(f"rewrote {DATA_PATH}")
    print(f"claims: {len(fixed)}")
    print(f"cited_doc_ids changed: {cited_changed}")
    print(f"  NEI cited_doc_ids restored from {TRAIN_PATH.name}: {nei_restored}")
    print(f"rationales: {rationales_before} -> {rationales_after}")


if __name__ == "__main__":
    main()
