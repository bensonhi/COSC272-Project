# Counterfactual minimal pairs

Evidence-deletion augmentation teaches the model whether evidence is *present*; it does not teach
what the evidence *says*. On minimal pairs where only the evidence finding is reversed, the
augmented model is fooled more often than original-data SFT (B2). We add LLM-written, LLM-verified
counterfactual edits of the training claims to the same training run.

| file | purpose |
|---|---|
| `generate.py` | writes three minimal edits per SUPPORT / CONTRADICT claim with Qwen3.8-27B (vLLM) and keeps those its zero-shot judge (the B5 prompt) labels as intended |
| `data/{train,test}.jsonl` | one record per (claim, edit type), with the edited text, the judge's verdicts and `verified` |
| `data.py` | records -> `LiquidAI.data.Sample`s (training groups and the `cf_*` eval sets in `baselines/common.py`) |
| `train_cf.py` | the team's `LiquidAI.training.train`, unchanged, on all training groups plus the verified edits |
| `annotation.py`, `annotation/` | blind human check of the test edits (`sheet.csv`; do not open `key.json` before labelling) |

Edit types (label of the edited input):

* `claim_negation`: the claim asserts the opposite (antonym preferred over "not"); SUPPORT <-> CONTRADICT.
* `evidence_flip`: the gold evidence (plus any sentence or title restating the finding) reports the
  opposite finding; SUPPORT <-> CONTRADICT, edited sentences become evidence.
* `evidence_weaken`: the finding no longer bears on the claim (other population / outcome ...); NEI.

Verified edits: train 363 / 281 / 61, test 138 / 128 / 14 (claim_negation / evidence_flip / evidence_weaken).

```bash
PATH=$S/env_vllm/bin:$PATH $S/env_vllm/bin/python counterfactual/generate.py --split train   # and --split test
CUDA_VISIBLE_DEVICES=0 python counterfactual/train_cf.py --seed 42 --edits evidence_flip evidence_weaken
cd baselines && python d1_adapter_eval.py --adapter-dir <out>/final --name d1_cf_evidence_only_seed42 --eval-set cf_evidence_flip
python counterfactual/annotation.py score counterfactual/annotation/sheet_A.csv counterfactual/annotation/sheet_B.csv
```

## Results (test, means over seeds; per-seed rows in `baselines/results/summary.md`)

| model | seeds | macro-F1 | evidence_flip pair | evidence_weaken pair (n=14) | deletion selectivity |
|---|---|---|---|---|---|
| original-data SFT (B2) | 3 | 0.856 | 0.784 | 0.19 | +0.33 |
| augmented SFT | 2 | 0.844 | 0.672 | 0.29 | +0.64 |
| + all counterfactuals | 2 | 0.835 | 0.863 | 0.64 | +0.64 |
| + evidence-side counterfactuals | 3 | 0.841 | 0.844 | 0.74 | +0.64 |

pair = correct on both the original and the edited claim. Counterfactual training fixes the
evidence-content blind spot without losing deletion robustness; standard-test macro-F1 is unchanged
within seed variance. Claim negations add little and cost some F1.

Caveats: the counterfactual test set is generated and filtered by the same LLM as the training
edits (so B5 is excluded from it, and a shared generator style could inflate the gain) until the
human check in `annotation/` is done. These models were trained on the 776-claim training split from
before `data/validation` was split off (commit 9786b42), so they have seen the validation claims.
