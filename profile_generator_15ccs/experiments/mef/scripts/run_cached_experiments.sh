#!/usr/bin/env bash
set -e
set -o pipefail

GPU="$1"

ROOT="$HOME/dl-workspace/experiment-results/CClinguist/mef-15ccs"

cd "$HOME/dl-workspace/projects/CClinguist-mef-15ccs/profile_generator_15ccs"

test "$(git branch --show-current)" = "experiment/mef-15ccs"

export CUDA_VISIBLE_DEVICES="$GPU"

COMMON=(
    --train-data data_15ccs/simulation_pk2hk_15CCs
    --test-data data_15ccs/simulation_15CCs
    --shared "$ROOT/shared"
    --classifier models/checkpoint_9000.pth.tar
    --cache-dir "$ROOT/cache"
    --batch-size 36
    --k-epochs 10
    --max-steps 7
    --decay 0.5
    --threshold 0.7
)

for SEED in 3402 3403 3404; do

    # 原子化领取整个种子的任务，避免重复训练
    if ! mkdir "$ROOT/full/seed-$SEED" 2>/dev/null; then
        continue
    fi

    echo "$GPU" > "$ROOT/full/seed-$SEED/gpu.txt"

    for MODE in baseline no_mask fusion_mask mef; do

        RUN="$ROOT/full/seed-$SEED/$MODE"
        TEST="$ROOT/full/seed-$SEED/test/$MODE"

        mkdir -p "$RUN" "$TEST"

        echo "========== GPU $GPU | SEED $SEED | $MODE =========="
        echo "TRAIN START: $(date)"

        python -u mef_experiment.py \
            "${COMMON[@]}" \
            --phase train \
            --mode "$MODE" \
            --seed "$SEED" \
            --epochs 5 \
            --out "$RUN" \
            > "$RUN/console.log" 2>&1 \
            || { tail -50 "$RUN/console.log"; exit 1; }

        grep -E '\[CACHE\]|\[MEF\] (EPOCH|ALL PASS)' \
            "$RUN/console.log"

        test -f "$RUN/epoch-5.pt"

        echo "TEST START: $(date)"

        python -u mef_experiment.py \
            "${COMMON[@]}" \
            --phase test \
            --mode "$MODE" \
            --seed "$SEED" \
            --checkpoint "$RUN/epoch-5.pt" \
            --indices 0 1 2 3 4 5 6 7 8 9 \
            --out "$TEST" \
            > "$TEST/console.log" 2>&1 \
            || { tail -50 "$TEST/console.log"; exit 1; }

        grep -E '\[CACHE\]|\[MEF\] (FINAL|ALL PASS)' \
            "$TEST/console.log"

        echo "COMPLETED: GPU=$GPU SEED=$SEED MODE=$MODE"
        echo "FINISH TIME: $(date)"

    done

    touch "$ROOT/full/seed-$SEED/COMPLETE"

done

echo "========== GPU $GPU WORKER FINISHED =========="
