#!/bin/bash
# GPU 5: regression-check the refactored MultiVerS inference, fine-tune on our train split, evaluate.
set -u
S=/scratch/f008k3y/cosc272
export CUDA_VISIBLE_DEVICES=5 HF_HOME=$S/hf_cache
P=$S/env/bin/python
cd "$(dirname "$0")"
$P multivers_zero_shot.py --name _regression_b4 > logs/b4_regression.log 2>&1 && \
  $P -c "import json;a=json.load(open('results/b4_multivers_fever_sci/metrics.json'));b=json.load(open('results/_regression_b4/metrics.json'));print('regression', a['label_macro_f1'], b['label_macro_f1'], a['sentence_level_evidence_f1'], b['sentence_level_evidence_f1'])"
rm -rf results/_regression_b4
W=$S/outputs/multivers-finetuned-seed42
[ -f $W/model.pt ] || $P multivers_finetune.py --output-dir $W > logs/b4b_train.log 2>&1 || { echo "finetune failed"; exit 1; }
for set in test evidence_removed random_removed; do
  $P multivers_zero_shot.py --weights $W/model.pt --name b4b_multivers_finetuned --eval-set $set > logs/b4b_eval_$set.log 2>&1 || echo "eval $set failed"
done
echo done
