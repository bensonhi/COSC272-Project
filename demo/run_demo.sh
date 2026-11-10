#!/bin/bash
# Launch the demo on the l40sx8 allocation.
#   GPU A: d1-3B with every available adapter (Gradio app, port 7860)
#   GPU B: optional Qwen3.8-27B vLLM server for live LLM comparison in batch mode (port 8000)
# Usage: demo/run_demo.sh <gpu_a> [<gpu_b>] [--share]
# Then on your laptop:  ssh -N -L 7860:localhost:7860 <netid>@l40sx8.hpcc.dartmouth.edu  -> http://localhost:7860
set -eu
S=/scratch/f008k3y/cosc272
export HF_HOME=$S/hf_cache
cd "$(dirname "$0")"
GPU_A=$1; GPU_B=${2:-}; SHARE=${3:-}
ADAPTERS=()
[ -f $S/outputs/d1-original-sft-seed42/final/adapter_config.json ] && \
  ADAPTERS+=(--adapter "Original-data SFT=$S/outputs/d1-original-sft-seed42/final")
AUG=${AUGMENTED_ADAPTER:-$S/outputs/d1-augmented-sft/final}
[ -f $AUG/adapter_config.json ] && ADAPTERS+=(--adapter "Augmented SFT (ours)=$AUG")
LLM=()
if [ -n "$GPU_B" ] && [ "$GPU_B" != "--share" ]; then
  CUDA_VISIBLE_DEVICES=$GPU_B PATH=$S/env_vllm/bin:$PATH nohup $S/env_vllm/bin/vllm serve Qwen/Qwen3.8-27B-FP8 \
    --port 8000 --max-model-len 16384 --max-num-seqs 16 --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    > $S/vllm_server.log 2>&1 &
  until curl -s localhost:8000/v1/models > /dev/null; do sleep 5; done
  LLM=(--llm-url http://localhost:8000/v1)
else
  SHARE=${GPU_B:-$SHARE}
fi
CUDA_VISIBLE_DEVICES=$GPU_A exec $S/env/bin/python app.py "${ADAPTERS[@]}" "${LLM[@]}" ${SHARE:+--share}
