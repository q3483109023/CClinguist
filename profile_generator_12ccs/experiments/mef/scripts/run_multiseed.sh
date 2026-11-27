#!/usr/bin/env bash

set -e
set -o pipefail

cd "$HOME/dl-workspace/projects/CClinguist/profile_generator_12ccs"

ROOT="$HOME/dl-workspace/experiment-results/CClinguist/mef"

export CUDA_VISIBLE_DEVICES=1

COMMON=(
    --train-data "$PWD/data_12ccs/simulation_hk2jp_1201to1203"
    --test-data "$PWD/data_12ccs/simulation_pk2hk_12_20"
    --shared "$ROOT/shared"
    --classifier "$PWD/models/best_model_accuracy.pth.tar"
    --batch-size 36
    --max-steps 7
    --decay 0.5
    --threshold 0.7
)

for seed in 3403 3404; do
    RUN="$ROOT/multiseed/seed-$seed"

    for mode in baseline no_mask fusion_mask mef; do
        echo "========== TRAIN: seed=$seed mode=$mode =========="

        OUT="$RUN/$mode"
        mkdir -p "$OUT"

        python -u mef_experiment.py \
            "${COMMON[@]}" \
            --phase train \
            --mode "$mode" \
            --seed "$seed" \
            --epochs 5 \
            --k-epochs 10 \
            --out "$OUT" \
            2>&1 | tee "$OUT/console.log"

        echo "========== TEST: seed=$seed mode=$mode =========="

        TEST="$RUN/test/$mode"
        mkdir -p "$TEST"

        python -u mef_experiment.py \
            "${COMMON[@]}" \
            --phase test \
            --mode "$mode" \
            --seed "$seed" \
            --checkpoint "$OUT/epoch-5.pt" \
            --indices 0 1 2 3 4 5 6 7 8 \
            --out "$TEST" \
            2>&1 | tee "$TEST/console.log"
    done
done

echo "========== ALL MULTISEED EXPERIMENTS COMPLETED =========="
