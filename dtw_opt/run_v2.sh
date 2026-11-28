#!/usr/bin/env bash
# Usage: bash dtw_opt/run_v2.sh 12 0 /absolute/results/root
set -euo pipefail
K="$1"
GPU="$2"
ROOT="$3"
SEED="${4:-3402}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/profile_generator_${K}ccs"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
case "$K" in
    12)
        TRAIN="data_12ccs/simulation_hk2jp_1201to1203"
        TEST="data_12ccs/simulation_pk2hk_12_20"
        CLASSIFIER="models/best_model_accuracy.pth.tar"
        INDICES=(0 1 2 3 4 5 6 7 8)
        ;;
    15)
        TRAIN="data_15ccs/simulation_pk2hk_15CCs"
        TEST="data_15ccs/simulation_15CCs"
        CLASSIFIER="models/checkpoint_9000.pth.tar"
        INDICES=(0 1 2 3 4 5 6 7 8 9)
        ;;
    *) exit 1 ;;
esac
test -d "$TRAIN" && test -d "$TEST" && test -f "$CLASSIFIER"
DIR="$ROOT/${K}ccs"
mkdir -p "$DIR"
if [ ! -f "$DIR/shared/dtw_metrics.json" ]; then
    # Shared lock avoids simultaneous DTW timing measurements.
    flock "$ROOT/dtw-benchmark.lock" python -u "$REPO/dtw_opt/prepare_v2.py" \
        --classes "$K" --train "$TRAIN" --test "$TEST" --out "$DIR/shared" \
        > "$DIR/prepare.log" 2>&1
fi
cat "$DIR/shared/dtw_metrics.json"
COMMON=(--train-data "$TRAIN" --test-data "$TEST" --classifier "$CLASSIFIER" --batch-size 36)
if [ ! -f "$DIR/cache/build_summary.json" ]; then
    python -u dtw_observation_cache.py "${COMMON[@]}" --shared "$DIR/shared/original" \
        --cache-dir "$DIR/cache" > "$DIR/cache.log" 2>&1
fi
for MODE in original cband; do
    RUN="$DIR/seed-$SEED/$MODE"
    test ! -e "$RUN"
    mkdir -p "$RUN/train" "$RUN/test"
    ARGS=("${COMMON[@]}" --shared "$DIR/shared/$MODE" --cache-dir "$DIR/cache"
          --mode baseline --seed "$SEED" --epochs 5 --k-epochs 10 --max-steps 7 --decay 0.5 --threshold 0.7)
    echo "TRAIN $K $MODE $(date)"
    python -u dtw_experiment.py "${ARGS[@]}" --phase train --out "$RUN/train" \
        > "$RUN/train/console.log" 2>&1
    test -f "$RUN/train/epoch-5.pt"
    echo "TEST $K $MODE $(date)"
    python -u dtw_experiment.py "${ARGS[@]}" --phase test --checkpoint "$RUN/train/epoch-5.pt" \
        --indices "${INDICES[@]}" --out "$RUN/test" > "$RUN/test/console.log" 2>&1
    grep '^\[MEF\] FINAL' "$RUN/test/console.log"
done
python - "$DIR" "$SEED" <<'PY'
import csv, json, sys
from pathlib import Path
d,seed=Path(sys.argv[1]),sys.argv[2]
rows=[]
for mode in ("original","cband"):
    p=d/f"seed-{seed}"/mode/"test"/"test_metrics.json"
    m=json.loads(p.read_text())["aggregate"]
    assert m["n"] == (3456 if d.name=="12ccs" else 2400)
    rows.append(dict(mode=mode,**m))
    print(mode, json.dumps(m))
with (d/f"seed-{seed}"/"comparison.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader(); w.writerows(rows)
for key in ("accuracy","macro_f1","multi_accuracy","mean_steps","wrong_elimination_rate"):
    print(f"{key}: cband - original = {rows[1][key]-rows[0][key]:+.6f}")
print("ALL DTW V2 RUNS PASS")
PY
