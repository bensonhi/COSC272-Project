# Reproducing the report

Every number in the report comes from predictions saved under `baselines/results/`, so the tables can be
rebuilt without a GPU (step 5). Steps 2-4 retrain and re-evaluate the models from scratch.

## 1. Environment

- Python 3.12, PyTorch 2.14+ with CUDA, Transformers 5.20+, PEFT 0.21 (training and evaluation, bf16).
- `pip install gradio==6.31.0` for the demo; a separate environment with vLLM 0.33 for the Qwen baseline.
- One GPU with 24 GB or more; training one configuration takes about 3-4 hours on an NVIDIA L40S.

## 2. Data

`data/train`, `data/validation` and `data/test` already hold the splits used in the report (698 / 78 / 333
claims). `README.md` documents how they were built from SciFact (`fix_train_more_format.py`,
`build_claim_splits.py`; the Grok evidence expansion in `augment_train_evidence.py` is cached under
`data/cache/`).

## 3. Train the three configurations

```bash
python -m LiquidAI --mode train_test --original           --output-dir LiquidAI/outputs/d1-original
python -m LiquidAI --mode train_test --learning-rate 1e-4  --output-dir LiquidAI/outputs/d1-augment
python -m LiquidAI --mode train_test --loss pair          --output-dir LiquidAI/outputs/d1-pair
```

All other settings are the defaults in `LiquidAI/cli.py` (LoRA r=16, alpha=32, 3 epochs, seed 42).
These are the single runs shown in the slides and demo. The three-seed comparison in the report retrains
every configuration with identical settings:

```bash
for seed in 42 43 44; do
  python -m LiquidAI --mode train --original  --seed $seed --question-batch-size 2 --output-dir outputs/v2-original-seed$seed
  python -m LiquidAI --mode train             --seed $seed --question-batch-size 2 --output-dir outputs/v2-augmented-seed$seed
  python -m LiquidAI --mode train --loss pair --seed $seed --question-batch-size 2 --output-dir outputs/v2-pair-seed$seed
done
``` Each run writes `final/` (the adapter), `training_config.json`
and test metrics.

## 4. Evaluate: test set, stress tests and baselines

```bash
cd baselines
for set in test evidence_removed random_removed evidence_removed_silent random_removed_silent \
           partial_removed partial_removed_silent; do   # all stress sets are built in memory from data/test
  python d1_adapter_eval.py --adapter-dir ../LiquidAI/outputs/d1-augment/final --name final_d1_augment --eval-set $set
done
python d1_zero_shot.py                         # d1-3B zero-shot   (each baseline also takes --eval-set)
python nli_zero_shot.py --label-source abstract  # DeBERTa-v3 NLI
python multivers_finetune.py && python multivers_zero_shot.py --name b4b_multivers_finetuned \
    --weights $COSC272_SCRATCH/outputs/multivers-finetuned-seed42/model.pt
python llm_zero_shot.py                          # Qwen3.8-27B, vLLM environment
```

`baselines/README.md` describes every baseline, the stress sets and where external checkpoints go.

## 5. Rebuild the tables (no GPU)

```bash
python baselines/report_tables.py  # results/report_tables.md: every number in the report's Results section
python baselines/summarize.py      # results/summary.md: test metrics, stress tests, per-system tables
python baselines/significance.py   # seed ensembles and paired bootstrap confidence intervals
python baselines/error_analysis.py # error categories
```

## 6. Demo

`demo/README.md`: `demo/app.py` runs the live Gradio app on a GPU; `demo/precompute.py` and
`demo/export_static.py` build the static site hosted at
https://huggingface.co/spaces/COSC272-Project/claim-verification-demo.
