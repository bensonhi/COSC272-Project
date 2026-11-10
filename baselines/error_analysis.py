"""Error analysis over one or more systems' test predictions (proposal's error taxonomy).

    python error_analysis.py b1_d1_zero_shot b5_llm_qwen3.8_27b ...   -> results/error_analysis.md

Label errors: unsupported judgement (gold NEI, predicted S/C), unnecessary abstention (gold S/C,
predicted NEI), polarity confusion (SUPPORT <-> CONTRADICT). Evidence errors on correctly labelled
S/C claims: missed gold sentences / extra selected sentences. High-confidence errors: confidence
>= 0.8. Claim subgroups by surface cues (regex, approximate): numeric, causal, negated, and claims
citing several abstracts.
"""

import json
import re
import sys

from common import RESULTS_DIR

NUMERIC = re.compile(r"\b\d+(?:[.,]\d+)?\b|%|\bpercent\b", re.I)
CAUSAL = re.compile(r"\b(caus\w*|lead[s]? to|induc\w*|increas\w*|decreas\w*|reduc\w*|promot\w*|inhibit\w*|"
                    r"prevent\w*|improv\w*|impair\w*|enhanc\w*|suppress\w*|block\w*|activat\w*|result\w* in)\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|none|neither|nor|without|cannot|fails?|lack\w*)\b|n't", re.I)
GROUPS = {"numeric": NUMERIC.search, "causal": CAUSAL.search, "negated": NEGATION.search}


def analyse(name):
    records = [json.loads(l) for l in (RESULTS_DIR / name / "predictions.jsonl").read_text().splitlines()]
    n = len(records)
    counts = {"unsupported judgement": 0, "unnecessary abstention": 0, "polarity confusion": 0}
    high = dict.fromkeys(counts, 0)
    missed = extra = evidence_claims = 0
    for r in records:
        g, p, conf = r["gold_label"], r["predicted_label"], r["label_confidence"]
        kind = None
        if g == "NEI" and p != "NEI":
            kind = "unsupported judgement"
        elif g != "NEI" and p == "NEI":
            kind = "unnecessary abstention"
        elif g != p:
            kind = "polarity confusion"
        if kind:
            counts[kind] += 1
            high[kind] += conf >= 0.8
        if g != "NEI" and g == p:
            evidence_claims += 1
            gold = {(a["doc_id"], i) for a in r["articles"] for i in a["gold_evidence_indices"]}
            pred = {(a["doc_id"], i) for a in r["articles"] for i in a["predicted_evidence_indices"]}
            missed += bool(gold - pred)
            extra += bool(pred - gold)
    subgroups = {}
    for group, match in [*GROUPS.items(), ("several abstracts", None)]:
        members = [r for r in records if (len(r["articles"]) > 1 if match is None else match(r["claim"]))]
        if members:
            errors = [r for r in members if not r["label_correct"]]
            subgroups[group] = (len(members), sum(r["label_correct"] for r in members) / len(members),
                                sum(r["label_confidence"] >= 0.8 for r in errors))
    return n, counts, high, missed, extra, evidence_claims, subgroups


def main():
    names = sys.argv[1:] or sorted(p.parent.name for p in RESULTS_DIR.glob("*/predictions.jsonl"))
    lines = ["## Label errors (count; of which confidence >= 0.8)", "",
             "| system | unsupported judgement | unnecessary abstention | polarity confusion | "
             "correct S/C with missed gold sentence | ... with extra selected sentence |", "|---|---|---|---|---|---|"]
    sub_lines = ["", "## Accuracy by claim subgroup (n; accuracy; high-confidence errors)", "",
                 "| system | " + " | ".join([*GROUPS, "several abstracts"]) + " |", "|---" * (len(GROUPS) + 2) + "|"]
    for name in names:
        n, counts, high, missed, extra, ev, subgroups = analyse(name)
        lines.append(f"| {name} | " + " | ".join(f"{counts[k]} ({high[k]})" for k in counts)
                     + f" | {missed}/{ev} | {extra}/{ev} |")
        sub_lines.append(f"| {name} | " + " | ".join(
            f"{m}; {acc:.2f}; {h}" for m, acc, h in (subgroups.get(g, (0, 0, 0)) for g in [*GROUPS, "several abstracts"])) + " |")
    text = "\n".join(lines + sub_lines) + "\n"
    (RESULTS_DIR / "error_analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
