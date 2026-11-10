# Baselines

Every system is scored on the same 333 held-out claims in `data/test` with the LiquidAI package's
`prediction_record` and `TestMetrics` (label accuracy, three-class macro-F1, sentence-level
evidence micro-F1 at p >= 0.5, top-label ECE with 10 bins), plus median per-claim latency.
`common.py` is the shared harness: a baseline only implements
`predict(sample) -> (P(NEI, SUPPORT, CONTRADICT), {(doc_id, original_index): P(evidence)})`.
No baseline uses the test split to choose anything (thresholds, prompts, epochs).

| id | script | system | training data |
|---|---|---|---|
| B0 | `trivial.py` | label prior (majority class) | train label counts |
| B1 | `d1_zero_shot.py` | d1-3B, no adapter, same prompts as the SFT runs | none |
| B2 | `d1_original_sft.py` + `d1_adapter_eval.py` | d1-3B + LoRA, team training code, **original groups only** (proposal Table 1, "Original-data SFT"); seeds 42/43/44 | 776 original train groups |
| B3 | `nli_zero_shot.py` | DeBERTa-v3-large NLI (MNLI/FEVER/ANLI/LingNLI/WANLI); label from abstract-level or best-sentence NLI, evidence = P(entail)+P(contra) per sentence | none (no SciFact) |
| B4 | `multivers_zero_shot.py` | MultiVerS `fever_sci` checkpoint, inference re-implemented on current transformers | FEVER + weak PubMedQA / EvidenceInference (no SciFact) |
| B4b | `multivers_finetune.py` | MultiVerS fine-tuned from `fever_sci` on our train split | 776 original train claims |
| B5 | `llm_zero_shot.py` | Qwen3.8-27B-FP8 (vLLM), non-thinking, greedy, JSON-schema constrained output | none |

`d1_adapter_eval.py` also evaluates the team's augmented-SFT adapter in the same harness, so all
rows of the final table come from one evaluation code path.

## Leakage notes
* The official MultiVerS `scifact` checkpoint was trained on the SciFact train split; 249 of our
  333 test claims come from that split, so it is **not** used. `fever_sci` has seen no SciFact data,
  and B4b is fine-tuned only on our training split.
* Qwen3.8-27B was not used for any data construction (augmentation used grok-4.7 on training claims
  only); its pre-training data cannot be audited, which we state as a limitation.

## Stress test: does the model abstain when evidence is removed?
`--eval-set evidence_removed` builds, in memory, a copy of the 207 test claims that have gold
evidence with every gold evidence sentence blanked (the same empty-slot format as the training
augmentation); the target is NEI. `--eval-set random_removed` is the control: the same claims with
as many randomly chosen non-evidence sentences blanked and the label unchanged. A model that only
reacts to "text is missing" abstains in both; a model that tracks evidence abstains only in the
first. `summarize.py` reports both NEI rates and their difference (selectivity). Gold rationales can
omit valid evidence, so evidence_removed NEI rates are a lower bound on correct abstention, and these
inputs were not manually verified.

## Running
Environment (node-local scratch on l40sx8, because the home quota is ~5 GB):
```bash
S=/scratch/f008k3y/cosc272
conda create -p $S/env python=3.12 && $S/env/bin/pip install "transformers>=5.14.1" torch peft accelerate gradio scikit-learn pandas
conda create -p $S/env_vllm python=3.12 && $S/env_vllm/bin/pip install vllm==0.33.0 ninja   # B5 only
export HF_HOME=$S/hf_cache
```
One GPU per run (`CUDA_VISIBLE_DEVICES=<id>`), from this directory:
```bash
python trivial.py
python d1_zero_shot.py                      [--eval-set evidence_removed|random_removed]
python d1_original_sft.py --seed 42         # ~75 min on one L40S; adapter -> $S/outputs/d1-original-sft-seed42
python d1_adapter_eval.py --adapter-dir $S/outputs/d1-original-sft-seed42/final --name b2_d1_original_sft_seed42
python nli_zero_shot.py --label-source abstract
python multivers_zero_shot.py               # needs fever_sci.ckpt, see the script docstring
python multivers_finetune.py && python multivers_zero_shot.py --weights $S/outputs/multivers-finetuned-seed42/model.pt --name b4b_multivers_finetuned
PATH=$S/env_vllm/bin:$PATH $S/env_vllm/bin/python llm_zero_shot.py
python summarize.py                         # -> results/summary.md, results/summary.json
```
`run_b2_rest.sh`, `run_b2_gpu5.sh`, `run_stress_zero_shot.sh` and `run_multivers_ft.sh` are the exact job chains we ran.

Outputs: `results/<system>/metrics.json` and `predictions.jsonl` (per-claim label probabilities,
per-sentence evidence probabilities, gold targets); stress tests under `results/<eval_set>/<system>/`.
