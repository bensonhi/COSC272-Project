"""Average evidence-label counts in claims_train_more.jsonl.

NEI claims (an empty evidence object) are left out. The remaining claims are
grouped by how many abstracts appear in evidence: 1, 2, 3, and 4 or more.
Each label count is the number of rationale objects with that label, so one
rationale is one count even when it lists several sentence indices.

    .venv/bin/python stats_train_more_labels.py
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "claims_train_more.jsonl"
LABELS = ("SUPPORT", "SUPPORT_more", "CONTRADICT", "CONTRADICT_more")
GROUPS = ((1, "1"), (2, "2"), (3, "3"), (4, "4+"))


def load_jsonl(path):
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label_counts(evidence):
    counts = {label: 0 for label in LABELS}
    for annotations in evidence.values():
        for annotation in annotations:
            label = annotation["label"]
            if label not in counts:
                raise ValueError(f"unexpected evidence label: {label}")
            counts[label] += 1
    return counts


def group_name(doc_count):
    if doc_count in (1, 2, 3):
        return str(doc_count)
    if doc_count >= 4:
        return "4+"
    raise ValueError(f"evidence document count must be positive, found {doc_count}")


def mean(total, count):
    return total / count


def main():
    grouped = {
        name: {"claims": 0, **{label: 0 for label in LABELS}}
        for _, name in GROUPS
    }
    skipped_nei = 0

    for row in load_jsonl(DATA_PATH):
        evidence = row.get("evidence") or {}
        if not evidence:
            skipped_nei += 1
            continue

        bucket = grouped[group_name(len(evidence))]
        bucket["claims"] += 1
        for label, count in label_counts(evidence).items():
            bucket[label] += count

    counted = sum(bucket["claims"] for bucket in grouped.values())
    print(f"{DATA_PATH.name}: {counted + skipped_nei} claims")
    print(f"skipped NEI (empty evidence): {skipped_nei}")
    print(f"counted: {counted}")
    print()
    header = f"{'docs':>6} {'claims':>8}" + "".join(f" {label:>16}" for label in LABELS)
    print(header)
    print("-" * len(header))
    for _, name in GROUPS:
        bucket = grouped[name]
        claims = bucket["claims"]
        cells = [f"{name:>6}", f"{claims:>8}"]
        for label in LABELS:
            if claims:
                cells.append(f"{mean(bucket[label], claims):>16.3f}")
            else:
                cells.append(f"{'n/a':>16}")
        print(" ".join(cells))


if __name__ == "__main__":
    main()
