"""Extend the static demo's cache with more combinations of removed sentences (resumable).

    python demo/precompute_combos.py --stage curated  --adapter ...   # every combination for the curated examples
    python demo/precompute_combos.py --stage relevant --adapter ...   # every item: all subsets of its relevant sentences
    python demo/precompute_combos.py --stage pairs    --adapter ...   # every item: any two sentences removed

Relevant sentences are the gold evidence plus every sentence a model selects as evidence on the full abstract
(at most 8, the most evidence-like first). A curated example with more than 10 sentences gets the relevant
subsets and the pairs instead of all 2^n subsets. Verdicts go to cache/verdicts.jsonl (as precompute.py) and the
removal sets to cache/extra_sets.jsonl, which export_static.py adds to each item.
"""

import argparse
from itertools import chain, combinations
import json
from pathlib import Path

from cache import cache_key, to_record
from engine import ZERO_SHOT, D1Verifier
from export_static import removal_key
from precompute import counterfactual_items, gold_keys, loaded_articles, masked

from app import load_test

HERE = Path(__file__).resolve().parent
RAW = HERE / "cache" / "verdicts.jsonl"
SETS = HERE / "cache" / "extra_sets.jsonl"
MAX_FULL = 10
MAX_RELEVANT = 8


def subsets(keys, sizes=None):
    keys = sorted(keys)
    sizes = range(len(keys) + 1) if sizes is None else sizes
    return [set(c) for c in chain.from_iterable(combinations(keys, k) for k in sizes)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("curated", "relevant", "pairs"), required=True)
    parser.add_argument("--adapter", action="append", default=[], help="NAME=PATH, same names as precompute.py")
    args = parser.parse_args()
    adapters = dict(item.split("=", 1) for item in args.adapter)
    models = [ZERO_SHOT, *adapters]

    done = {}
    for line in RAW.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:  # a line cut off by a preempted run
            continue
        done[record["key"]] = {(d, i): p for d, i, p in record["sentences"]}
    recorded = set()
    if SETS.exists():
        recorded = {(r["id"], removal_key(map(tuple, r["removed"]))) for r in map(json.loads, SETS.read_text().splitlines())}

    test = load_test()
    items = [(f"t{s.claim_id}", s) for s in test.values()] + [(f"c{s.claim_id}-{s.variant}", s) for s in counterfactual_items()]
    if args.stage == "curated":
        curated = {f"t{e['claim_id']}" for e in json.loads((HERE / "examples.json").read_text())}
        items = [(i, s) for i, s in items if i in curated]

    engine = None
    raw, sets_out = RAW.open("a"), SETS.open("a")
    for n, (item_id, sample) in enumerate(items):
        claim, articles = sample.claim.strip(), loaded_articles(sample)
        keys = [(a["doc_id"], i) for a in articles for i in range(len(a["sentences"]))]
        full = [done.get(cache_key(m, claim, articles), {}) for m in models]
        score = {k: max(f.get(k, 0.0) for f in full) for k in keys}
        relevant = gold_keys(sample, articles) | {k for k in keys if score[k] >= 0.5}
        relevant = set(sorted(relevant, key=lambda k: -score[k])[:MAX_RELEVANT])
        if args.stage == "curated" and len(keys) <= MAX_FULL:
            wanted = subsets(keys)
        elif args.stage == "curated":
            wanted = subsets(relevant) + subsets(keys, (2,))
        elif args.stage == "relevant":
            wanted = subsets(relevant)
        else:
            wanted = subsets(keys, (2,))
        for removed in wanted:
            for model in models:
                key = cache_key(model, claim, masked(articles, removed))
                if key not in done:
                    engine = engine or D1Verifier(adapters)
                    record = to_record(key, engine.verify(claim, masked(articles, removed), model))
                    raw.write(json.dumps(record) + "\n")
                    done[key] = {}
            if (item_id, removal_key(removed)) not in recorded:
                sets_out.write(json.dumps({"id": item_id, "removed": sorted(removed)}) + "\n")
                recorded.add((item_id, removal_key(removed)))
        raw.flush(), sets_out.flush()
        print(f"{args.stage} {n + 1}/{len(items)} {item_id}: {len(wanted)} sets", flush=True)
    print("done", args.stage)


if __name__ == "__main__":
    main()
