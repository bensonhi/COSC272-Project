#!/bin/bash
# Stress tests (evidence_removed / random_removed) for the zero-shot baselines on GPU 5.
set -u
S=/scratch/f008k3y/cosc272
export CUDA_VISIBLE_DEVICES=5 HF_HOME=$S/hf_cache PATH=$S/env_vllm/bin:$PATH
cd "$(dirname "$0")"
for set in evidence_removed random_removed; do
  $S/env/bin/python d1_zero_shot.py --eval-set $set > logs/stress_${set}_b1.log 2>&1 || echo "b1 $set failed"
  $S/env/bin/python nli_zero_shot.py --label-source abstract --eval-set $set > logs/stress_${set}_b3.log 2>&1 || echo "b3 $set failed"
  $S/env/bin/python multivers_zero_shot.py --eval-set $set > logs/stress_${set}_b4.log 2>&1 || echo "b4 $set failed"
  $S/env_vllm/bin/python llm_zero_shot.py --eval-set $set --latency-claims 1 > logs/stress_${set}_b5.log 2>&1 || echo "b5 $set failed"
done
echo done
