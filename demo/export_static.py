"""Export the precomputed verdicts as a static site (the public Hugging Face Space).

    python demo/export_static.py --out <dir>      # after precompute.py --finish

<dir>/index.json lists every test claim and counterfactual pair with the models and curated
examples; <dir>/items/<id>.json holds one claim's abstract, gold annotations and every precomputed
verdict, keyed by the set of removed sentences ("" = nothing removed, else sorted "doc-i" joined by
"|"). static/index.html renders them; nothing is computed in the browser.
"""

import argparse
import json
from pathlib import Path
import shutil

from cache import CACHE_FILE, cache_key, read_cache
from engine import ROOT
from precompute import counterfactual_items, gold_keys, loaded_articles, masked

from app import load_test
from LiquidAI.config import LABELS

HERE = Path(__file__).resolve().parent
EDIT_NAMES = {"claim_negation": "claim negated", "evidence_flip": "evidence reversed",
              "evidence_weaken": "evidence made irrelevant"}


def removal_key(removed):
    return "|".join(sorted(f"{d}-{i}" for d, i in removed))


def compact(record):
    return {"label": record["label"], "p": [round(record["probs"][label], 4) for label in LABELS],
            "e": {f"{d}-{i}": round(p, 4) for d, i, p in record["sentences"]},
            "ms": round(record["seconds"] * 1000)}


def candidate_sets(articles, cache, claim, models, gold):
    """Every removal set precompute.py may have produced for this input."""
    sets = [set(), gold] + [{(a["doc_id"], i)} for a in articles for i in range(len(a["sentences"]))]
    for owner in models:
        record = cache.get(cache_key(owner, claim, articles))
        if not record:
            continue
        first = {(d, i) for d, i, p in record["sentences"] if p >= 0.5}
        if first:
            sets.append(first)
            second = cache.get(cache_key(owner, claim, masked(articles, first)))
            if second:
                sets.append(first | {(d, i) for d, i, p in second["sentences"] if p >= 0.5})
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cache = read_cache(CACHE_FILE)
    models = json.loads((CACHE_FILE.parent / "models.json").read_text())
    examples = json.loads((HERE / "examples.json").read_text())
    (args.out / "items").mkdir(parents=True, exist_ok=True)

    entries, verdicts = [], 0
    test = load_test()
    items = [(f"t{s.claim_id}", s) for s in test.values()]
    items += [(f"c{s.claim_id}-{s.variant}", s) for s in counterfactual_items()]
    for item_id, sample in items:
        claim, articles = sample.claim.strip(), loaded_articles(sample)
        gold = gold_keys(sample, articles)
        results = {}
        for removed in candidate_sets(articles, cache, claim, models, gold):
            per_model = {}
            for model in models:
                record = cache.get(cache_key(model, claim, masked(articles, removed)))
                if record:
                    per_model[model] = compact(record)
            if per_model:
                results[removal_key(removed)] = per_model
        if "" not in results:
            continue
        verdicts += sum(map(len, results.values()))
        counterfactual = sample.variant != "original"
        edited, original_claim = [], None
        if counterfactual:
            before = loaded_articles(test[sample.claim_id])
            edited = [f"{a['doc_id']}-{i}" for a, b in zip(articles, before)
                      for i, (new, old) in enumerate(zip(a["sentences"], b["sentences"])) if new != old]
            if claim != test[sample.claim_id].claim.strip():
                original_claim = test[sample.claim_id].claim.strip()
        (args.out / "items" / f"{item_id}.json").write_text(json.dumps({
            "id": item_id, "claim": claim, "label": sample.label,
            "articles": [{"doc_id": a["doc_id"], "title": a["title"], "sentences": a["sentences"]} for a in articles],
            "gold": sorted(f"{d}-{i}" for d, i in gold), "gold_removed": removal_key(gold) if gold else None,
            "edit": EDIT_NAMES.get(sample.variant) if counterfactual else None,
            "original": f"t{sample.claim_id}" if counterfactual else None,
            "edited": edited, "original_claim": original_claim,
            "results": results}, separators=(",", ":")))
        entries.append({"id": item_id, "claim": claim, "label": sample.label,
                        "edit": EDIT_NAMES.get(sample.variant) if counterfactual else None})
    curated = [{"id": f"t{e['claim_id']}", "tag": e["tag"], "note": e.get("note", "")} for e in examples]
    (args.out / "index.json").write_text(json.dumps(
        {"models": models, "labels": list(LABELS), "items": entries, "curated": curated}, separators=(",", ":")))
    for name in ("index.html", "results.json"):  # results.json: the Results tab (same numbers as the slides)
        shutil.copy(HERE / "static" / name, args.out / name)
    print(f"{len(entries)} items, {verdicts} verdicts -> {args.out}")


if __name__ == "__main__":
    main()
