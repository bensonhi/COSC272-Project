"""Pick curated demo examples from a model's test predictions (deterministic rules).

    python demo/pick_examples.py --main d1_augmented_sft_seed42 [--other b5_llm_qwen3.8_27b]
        [--contrast b2_d1_original_sft_seed42]

Writes demo/examples.json: evidence-removal showcases (from the evidence_removed stress run: the
main model is right on the full input and says NEI once the gold evidence is blanked, while the
contrast model keeps its verdict), representative cases per label, challenging cases (several
cited abstracts, numeric claims, disagreement with the other system) and failure cases
(confident unsupported judgement, SUPPORT/CONTRADICT confusion, unnecessary abstention).
Rerun with --main pointing at the final model's results before the presentation.
"""

import argparse
import json
from pathlib import Path
import re

RESULTS = Path(__file__).resolve().parent.parent / "baselines" / "results"


def load(name, eval_set=None):
    path = RESULTS / (f"{eval_set}/{name}" if eval_set else name) / "predictions.jsonl"
    if not path.exists():
        return {}
    return {r["claim_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def n_sentences(r):
    return sum(len(a["sentences"]) for a in r["articles"])


def evidence_f1(r):
    gold = {(a["doc_id"], i) for a in r["articles"] for i in a["gold_evidence_indices"]}
    pred = {(a["doc_id"], i) for a in r["articles"] for i in a["predicted_evidence_indices"]}
    return 2 * len(gold & pred) / (len(gold) + len(pred)) if gold or pred else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", required=True, help="results folder of the model shown in the demo")
    parser.add_argument("--other", default="b5_llm_qwen3.8_27b")
    parser.add_argument("--contrast", default="b2_d1_original_sft_seed42",
                        help="model expected to keep its verdict after evidence removal")
    args = parser.parse_args()
    main_preds = load(args.main)
    other = load(args.other)
    removed_main, removed_contrast = load(args.main, "evidence_removed"), load(args.contrast, "evidence_removed")
    used = set()

    def pick(tag, note, candidates, key, n=1):
        chosen = []
        for r in sorted(candidates, key=key):
            if r["claim_id"] not in used and len(chosen) < n:
                used.add(r["claim_id"])
                chosen.append({"claim_id": r["claim_id"], "tag": tag, "note": note.format(**r)})
        return chosen

    rs = list(main_preds.values())
    short = [r for r in rs if n_sentences(r) <= 12]
    correct = [r for r in short if r["label_correct"]]
    examples = []
    examples += pick("evidence removal", "Gold {gold_label}, model right on the full abstract; with the gold evidence "
                     f"removed it says NEI while {args.contrast} keeps its verdict.",
                     [r for r in short if r["label_correct"] and r["claim_id"] in removed_main
                      and removed_main[r["claim_id"]]["predicted_label"] == "NEI"
                      and removed_contrast.get(r["claim_id"], {}).get("predicted_label", "NEI") != "NEI"],
                     key=lambda r: (-removed_main[r["claim_id"]]["label_confidence"], -r["label_confidence"]), n=2)
    for label in ("SUPPORT", "CONTRADICT", "NEI"):
        examples += pick("representative", f"Correct {label} with high confidence.",
                         [r for r in correct if r["gold_label"] == label],
                         key=lambda r: (-evidence_f1(r), -r["label_confidence"]))
    examples += pick("challenging", "Several cited abstracts; gold {gold_label}, model says {predicted_label}.",
                     [r for r in rs if len(r["articles"]) >= 2 and n_sentences(r) <= 30],
                     key=lambda r: (not r["label_correct"], -r["label_confidence"]))
    examples += pick("challenging", "Numeric claim; gold {gold_label}, model says {predicted_label}.",
                     [r for r in short if re.search(r"\b\d+(?:[.,]\d+)?\b", r["claim"]) and r["gold_label"] != "NEI"],
                     key=lambda r: (r["label_correct"], -r["label_confidence"]))
    if other:
        examples += pick("challenging", f"Disagrees with {args.other}; gold {{gold_label}}, model says {{predicted_label}}.",
                         [r for r in short if r["claim_id"] in other
                          and other[r["claim_id"]]["predicted_label"] != r["predicted_label"]],
                         key=lambda r: -r["label_confidence"])
    wrong = [r for r in short if not r["label_correct"]]
    examples += pick("failure", "Unsupported judgement: gold NEI but the model says {predicted_label}.",
                     [r for r in wrong if r["gold_label"] == "NEI"], key=lambda r: -r["label_confidence"])
    examples += pick("failure", "Polarity error: gold {gold_label}, model says {predicted_label}.",
                     [r for r in wrong if {r["gold_label"], r["predicted_label"]} == {"SUPPORT", "CONTRADICT"}],
                     key=lambda r: -r["label_confidence"])
    examples += pick("failure", "Unnecessary abstention: gold {gold_label}, model says NEI.",
                     [r for r in wrong if r["predicted_label"] == "NEI"], key=lambda r: -r["label_confidence"])
    out = Path(__file__).with_name("examples.json")
    out.write_text(json.dumps(examples, indent=2) + "\n")
    for e in examples:
        r = main_preds[e["claim_id"]]
        print(f"[{e['tag']}] #{e['claim_id']} conf={r['label_confidence']:.2f} sents={n_sentences(r)} | {e['note']}\n    {r['claim']}")


if __name__ == "__main__":
    main()
