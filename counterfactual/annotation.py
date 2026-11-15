"""Blind human check of the counterfactual test set: how often is the intended label right?

    python counterfactual/annotation.py sheet [--per-edit 25]   # writes annotation/sheet.csv + key.json
    python counterfactual/annotation.py score annotation/sheet_<name>.csv ...

`sheet` samples verified test edits (stratified by edit type, fixed seed) and writes a CSV with the
claim and the full edited abstract(s), no label and no edit type; annotators fill `label` with
SUPPORT / CONTRADICT / NEI. `score` reports, per edit type, the share of items where the annotator
agrees with the intended label, inter-annotator agreement when several sheets are given, and the
accuracy of every d1 model on the items the annotators confirmed (majority vote).
"""

import argparse
import collections
import csv
import json
from pathlib import Path
import random

from data import CF_DIR, EDIT_TYPES, ROOT, counterfactual_samples

OUT = CF_DIR.parent / "annotation"
RESULTS = ROOT / "baselines" / "results"
LABELS = ("SUPPORT", "CONTRADICT", "NEI")


def item_id(sample):
    return f"{sample.claim_id}/{sample.variant}"


def abstract_text(sample):
    parts = []
    for a in sample.articles:
        parts.append(f"[{a.title}]")
        parts += [f"({i}) {s.strip()}" for i, s in enumerate(a.sentences) if s]
    return "\n".join(parts)


def sheet(per_edit):
    samples = counterfactual_samples("test", ROOT / "data" / "test")
    rng = random.Random(0)
    chosen = []
    for edit in EDIT_TYPES:
        pool = [s for s in samples if s.variant == edit]
        chosen += rng.sample(pool, min(per_edit, len(pool)))
    rng.shuffle(chosen)
    OUT.mkdir(exist_ok=True)
    with (OUT / "sheet.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "claim", "abstract", "label", "note"])
        for n, s in enumerate(chosen, 1):
            writer.writerow([n, s.claim, abstract_text(s), "", ""])
    key = [{"n": n, "id": item_id(s), "edit": s.variant, "intended": s.label} for n, s in enumerate(chosen, 1)]
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {OUT / 'sheet.csv'} ({len(chosen)} items)", collections.Counter(s.variant for s in chosen))


def read_labels(path):
    labels = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].strip().upper().replace("NOT ENOUGH INFO", "NEI")
            label = {"S": "SUPPORT", "C": "CONTRADICT", "N": "NEI"}.get(label, label)
            if label in LABELS:
                labels[int(row["n"])] = label
    return labels


def score(paths):
    key = {k["n"]: k for k in json.loads((OUT / "key.json").read_text())}
    sheets = [read_labels(p) for p in paths]
    for path, labels in zip(paths, sheets):
        agree = collections.defaultdict(list)
        for n, label in labels.items():
            agree[key[n]["edit"]].append(label == key[n]["intended"])
        print(Path(path).name, {e: f"{sum(v)}/{len(v)}" for e, v in agree.items()})
    if len(sheets) > 1:
        shared = set.intersection(*(set(s) for s in sheets))
        pairwise = [sum(a[n] == b[n] for n in shared) / len(shared)
                    for i, a in enumerate(sheets) for b in sheets[i + 1:]]
        print(f"inter-annotator raw agreement on {len(shared)} items: {sum(pairwise) / len(pairwise):.3f}")
    votes = {}
    for n in key:
        counts = collections.Counter(s[n] for s in sheets if n in s)
        if counts:
            label, count = counts.most_common(1)[0]
            if count > sum(counts.values()) / 2:
                votes[n] = label
    confirmed = {key[n]["id"] for n, label in votes.items() if label == key[n]["intended"]}
    print(f"confirmed (majority = intended): {len(confirmed)}/{len(key)}")
    by_edit = collections.Counter(key[n]["edit"] for n, label in votes.items() if label == key[n]["intended"])
    print("models on confirmed items:", dict(by_edit))
    for name in sorted(p.name for p in (RESULTS / "cf_evidence_flip").iterdir()):
        correct = []
        for edit in EDIT_TYPES:
            path = RESULTS / f"cf_{edit}" / name / "predictions.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    if f"{r['claim_id']}/{r['variant']}" in confirmed:
                        correct.append(r["label_correct"])
        if correct:
            print(f"  {name:32s} acc={sum(correct) / len(correct):.3f} (n={len(correct)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sheet").add_argument("--per-edit", type=int, default=25)
    sub.add_parser("score").add_argument("sheets", nargs="+")
    args = parser.parse_args()
    sheet(args.per_edit) if args.cmd == "sheet" else score(args.sheets)


if __name__ == "__main__":
    main()
