"""Precompute every verdict the demo can show for the test claims and counterfactual pairs.

    CUDA_VISIBLE_DEVICES=0 python demo/precompute.py --adapter "Original-data SFT=..." ... [--skip-single]

Inputs, in priority order (so a partial run is still useful), each for every model:
  1. every test claim and counterfactual pair as loaded
  2. "Remove predicted evidence & re-verify", twice in a row, for each model's own evidence
     (every model is run on each resulting input, for "Compare all models")
  3. all gold evidence removed
  4. each single sentence removed
Verdicts are appended to cache/verdicts.jsonl (resumable: inputs already there are skipped), then
`--finish` writes cache/verdicts.jsonl.gz and cache/models.json for the app.
"""

import argparse
import gzip
import json
from pathlib import Path
import sys

from app import articles_to_text, load_test, sample_articles, text_to_articles
from cache import cache_key, to_record
from engine import ROOT, ZERO_SHOT, D1Verifier

OUT = Path(__file__).with_name("cache")


def counterfactual_items():
    sys.path.insert(0, str(ROOT / "counterfactual"))
    from data import counterfactual_samples

    return counterfactual_samples("test", ROOT / "data" / "test")


def loaded_articles(sample):
    """The articles exactly as the app parses them after loading the sample into the text box."""
    return text_to_articles(articles_to_text(sample_articles(sample)))


def masked(articles, removed):
    return [{**a, "sentences": [("" if (a["doc_id"], i) in removed else s) for i, s in enumerate(a["sentences"])]}
            for a in articles]


def gold_keys(sample, articles):
    """Gold evidence as (doc_id, index) in the parsed articles (sentence text matched in order)."""
    gold = {(a.doc_id, a.sentences[i].strip()) for a in sample.articles for i, t in enumerate(a.evidence_targets) if t}
    return {(a["doc_id"], i) for a in articles for i, s in enumerate(a["sentences"]) if (a["doc_id"], s) in gold}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", action="append", default=[], help="NAME=PATH, same names as in app.py")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--finish", action="store_true", help="only compress the cache and write models.json")
    args = parser.parse_args()
    adapters = dict(item.split("=", 1) for item in args.adapter)
    names = [ZERO_SHOT, *adapters]
    OUT.mkdir(exist_ok=True)
    raw = OUT / "verdicts.jsonl"
    if args.finish:
        records = {}
        for line in raw.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:  # a line still being written
                continue
            records[record["key"]] = record
        with gzip.open(OUT / "verdicts.jsonl.gz", "wt") as handle:
            for record in records.values():
                handle.write(json.dumps(record) + "\n")
        (OUT / "models.json").write_text(json.dumps(names))
        print(f"{len(records)} verdicts")
        return

    def selected(record):
        return {(d, i) for d, i, p in record["sentences"] if p >= 0.5}

    # key -> sentences the model selected as evidence, for every verdict already cached
    done = {}
    if raw.exists():
        for line in raw.read_text().splitlines():
            record = json.loads(line)
            done[record["key"]] = selected(record)
    engine = D1Verifier(adapters)
    handle = raw.open("a")

    def verify(claim, articles, model):
        key = cache_key(model, claim, articles)
        if key not in done:
            record = to_record(key, engine.verify(claim, articles, model))
            handle.write(json.dumps(record) + "\n")
            done[key] = selected(record)
        return done[key]

    evidence = verify

    items = list(load_test().values()) + counterfactual_items()
    print(f"{len(items)} inputs x {len(names)} models", flush=True)
    stages = [("loaded", None), ("strip", None), ("gold", None), ("single", None)]
    for stage, _ in stages:
        if stage == "single" and args.skip_single:
            continue
        for n, sample in enumerate(items):
            claim, articles = sample.claim.strip(), loaded_articles(sample)
            if stage == "loaded":
                for model in names:
                    verify(claim, articles, model)
            elif stage == "strip":
                for owner in names:
                    first = evidence(claim, articles, owner)
                    if not first:
                        continue
                    for model in names:
                        verify(claim, masked(articles, first), model)
                    second = first | evidence(claim, masked(articles, first), owner)
                    if second != first:
                        for model in names:
                            verify(claim, masked(articles, second), model)
            elif stage == "gold":
                gold = gold_keys(sample, articles)
                if gold and sample.variant == "original":
                    for model in names:
                        verify(claim, masked(articles, gold), model)
            elif stage == "single" and sample.variant == "original":
                for a in articles:
                    for i in range(len(a["sentences"])):
                        for model in names:
                            verify(claim, masked(articles, {(a["doc_id"], i)}), model)
            if n % 50 == 0:
                handle.flush()
                print(f"{stage} {n}/{len(items)} cached={len(done)}", flush=True)
        handle.flush()
    handle.close()
    print("done", len(done))


if __name__ == "__main__":
    main()
