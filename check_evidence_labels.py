"""Check whether each train/dev claim uses a single evidence label.

A claim with a non-empty evidence field is uniform when every annotation is
SUPPORT, or every annotation is CONTRADICT. Mixed claims and any other label
are reported with their ids.
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SPLITS = ("claims_train.jsonl", "claims_dev.jsonl")


def load_jsonl(path):
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                rows.append((line_number, json.loads(line)))
    return rows


def labels_of(evidence):
    labels = []
    for annotations in evidence.values():
        for annotation in annotations:
            labels.append(annotation["label"])
    return labels


def check_split(path):
    counts = Counter()
    mixed = []
    other = []

    for line_number, row in load_jsonl(path):
        evidence = row.get("evidence") or {}
        if not evidence:
            counts["empty"] += 1
            continue

        labels = labels_of(evidence)
        unique = set(labels)
        if unique == {"SUPPORT"}:
            counts["SUPPORT"] += 1
        elif unique == {"CONTRADICT"}:
            counts["CONTRADICT"] += 1
        elif unique <= {"SUPPORT", "CONTRADICT"}:
            counts["mixed"] += 1
            mixed.append((line_number, row["id"], labels))
        else:
            counts["other"] += 1
            other.append((line_number, row["id"], labels))

    return counts, mixed, other


def main():
    for name in SPLITS:
        path = DATA_DIR / name
        counts, mixed, other = check_split(path)
        total = sum(counts.values())
        with_evidence = total - counts["empty"]
        uniform = counts["SUPPORT"] + counts["CONTRADICT"]

        print(f"{name}: {total} claims")
        print(f"  empty evidence: {counts['empty']}")
        print(f"  with evidence:  {with_evidence}")
        print(f"    all SUPPORT:    {counts['SUPPORT']}")
        print(f"    all CONTRADICT: {counts['CONTRADICT']}")
        print(f"    mixed:          {counts['mixed']}")
        print(f"    other label:    {counts['other']}")
        print(f"  uniform among claims with evidence: {uniform}/{with_evidence}")

        for line_number, claim_id, labels in mixed:
            print(f"  mixed line {line_number} id {claim_id}: {labels}")
        for line_number, claim_id, labels in other:
            print(f"  other line {line_number} id {claim_id}: {labels}")
        print()


if __name__ == "__main__":
    main()
