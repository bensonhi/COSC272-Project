#!/bin/bash
# GPU 4: evaluate B2 seed 42, then train and evaluate seed 43 (replaces run_b2_chain.sh).
set -u
S=/scratch/f008k3y/cosc272
export CUDA_VISIBLE_DEVICES=4 HF_HOME=$S/hf_cache
P=$S/env/bin/python
cd "$(dirname "$0")"
for seed in 42 43; do
  out=$S/outputs/d1-original-sft-seed$seed
  if [ ! -f $out/final/adapter_config.json ]; then
    $P d1_original_sft.py --seed $seed > logs/b2_train_seed$seed.log 2>&1 || { echo "train seed $seed failed"; continue; }
  fi
  $P d1_adapter_eval.py --adapter-dir $out/final --name b2_d1_original_sft_seed$seed > logs/b2_eval_seed$seed.log 2>&1 \
    || echo "eval seed $seed failed"
done
echo done
