#!/bin/bash
# GPU 5: B2 seed 44 (in parallel with run_b2_chain.sh on GPU 4), then B2 stress tests for all seeds.
set -u
S=/scratch/f008k3y/cosc272
export CUDA_VISIBLE_DEVICES=5 HF_HOME=$S/hf_cache
P=$S/env/bin/python
cd "$(dirname "$0")"
out=$S/outputs/d1-original-sft-seed44
$P d1_original_sft.py --seed 44 > logs/b2_train_seed44.log 2>&1 && \
  $P d1_adapter_eval.py --adapter-dir $out/final --name b2_d1_original_sft_seed44 > logs/b2_eval_seed44.log 2>&1 || echo "seed 44 failed"
for seed in 42 43 44; do
  until [ -f results/b2_d1_original_sft_seed$seed/metrics.json ]; do sleep 60; done
  for set in evidence_removed random_removed; do
    $P d1_adapter_eval.py --adapter-dir $S/outputs/d1-original-sft-seed$seed/final --name b2_d1_original_sft_seed$seed \
      --eval-set $set > logs/stress_${set}_b2_seed$seed.log 2>&1 || echo "stress $set seed $seed failed"
  done
done
echo done
