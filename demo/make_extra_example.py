"""Build the page's opening example (a hand-written COSC 272 claim) with every combination of removed sentences.

    python demo/make_extra_example.py --adapter "d1-original=<dir>" --adapter "d1-augment=<dir>" --adapter "d1-pair=<dir>"

The abstract is short (7 sentences), so all 2^7 removal sets are verified with every model and written to
static/extra_items/x272.json in the same format as export_static.py's items; export_static.py puts it first.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path

from cache import to_record
from engine import D1Verifier
from export_static import compact, removal_key

OUT = Path(__file__).with_name("static") / "extra_items" / "x272.json"
CLAIM = "There are multiple quizzes in COSC272."
SENTENCES = [
    "The course project is graded out of 100 points in total.",
    "There are 5 quizzes in COSC272.",
    "The interactive demonstration is worth 15 points and must show representative and challenging cases.",
    "The presentation is limited to three minutes and may use no more than five slides.",
    "Each student also takes part in an individual oral defense worth 20 points.",
    "The final report is due on November 19.",
    "A missing deliverable receives zero for the corresponding rubric category.",
]
DOC = 272
GOLD = {(DOC, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", action="append", default=[], help="NAME=PATH, same names as precompute.py")
    args = parser.parse_args()
    engine = D1Verifier(dict(item.split("=", 1) for item in args.adapter))
    articles = [{"doc_id": DOC, "title": "COSC 272 course information", "sentences": SENTENCES}]
    keys = [(DOC, i) for i in range(len(SENTENCES))]
    results = {}
    for size in range(len(keys) + 1):
        for removed in map(set, combinations(keys, size)):
            masked = [{**a, "sentences": ["" if (a["doc_id"], i) in removed else s for i, s in enumerate(a["sentences"])]}
                      for a in articles]
            results[removal_key(removed)] = {m: compact(to_record("", engine.verify(CLAIM, masked, m)))
                                            for m in engine.model_names}
    item = {"id": "x272", "claim": CLAIM, "label": "SUPPORT", "articles": articles,
            "gold": sorted(f"{d}-{i}" for d, i in GOLD), "gold_removed": removal_key(GOLD),
            "edit": None, "original": None, "edited": [], "original_claim": None, "results": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(item, separators=(",", ":")))
    print(f"{len(results)} removal sets x {len(engine.model_names)} models -> {OUT}")


if __name__ == "__main__":
    main()
