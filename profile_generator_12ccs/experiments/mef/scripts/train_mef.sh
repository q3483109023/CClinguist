#!/usr/bin/env bash

set -e
set -o pipefail

cd "$HOME/dl-workspace/projects/CClinguist/profile_generator_12ccs"

ROOT="$HOME/dl-workspace/experiment-results/CClinguist/mef"
RUN="$ROOT/train-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$RUN"

for mode in baseline mef; do
    echo "========== TRAIN: $mode =========="

    OUT="$RUN/$mode"
    mkdir -p "$OUT"

    CUDA_VISIBLE_DEVICES=1 python -u mef_experiment.py \
        --phase train \
        --mode "$mode" \
        --train-data "$PWD/data_12ccs/simulation_hk2jp_1201to1203" \
        --test-data "$PWD/data_12ccs/simulation_pk2hk_12_20" \
        --shared "$ROOT/shared" \
        --classifier "$PWD/models/best_model_accuracy.pth.tar" \
        --out "$OUT" \
        --seed 3402 \
        --epochs 5 \
        --batch-size 36 \
        --k-epochs 10 \
        --max-steps 7 \
        --decay 0.5 \
        --threshold 0.7 \
        2>&1 | tee "$OUT/console.log"
done

echo "========== ALL TRAINING COMPLETED =========="
echo "Results: $RUN"
