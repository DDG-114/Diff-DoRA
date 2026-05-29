#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
MODE=${3:-all}
SEED=${4:-42}
MAX_TRAIN_ITEMS=${5:-4000}
WINDOW_STRIDE=${WINDOW_STRIDE:-6}
NORMALIZATION_SOURCE=${NORMALIZATION_SOURCE:-train_only}
FEWSHOT_RATIOS=${FEWSHOT_RATIOS:-0.05,0.10,0.20,0.40,1.00}
ZEROSHOT_SOURCE_RATIOS=${ZEROSHOT_SOURCE_RATIOS:-0.20,0.40,0.60,0.80}
ZEROSHOT_HALF_TRAIN_RATIOS=${ZEROSHOT_HALF_TRAIN_RATIOS:-0.60,0.80}
ZEROSHOT_TEST_WINDOW_DIVISOR=${ZEROSHOT_TEST_WINDOW_DIVISOR:-10}

source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

SKIP_FEWSHOT_ARGS=()
SKIP_ZEROSHOT_ARGS=()
case "${MODE}" in
  all)
    ;;
  fewshot)
    SKIP_ZEROSHOT_ARGS+=(--skip_zeroshot)
    ;;
  zeroshot)
    SKIP_FEWSHOT_ARGS+=(--skip_fewshot)
    ;;
  *)
    echo "Unsupported mode: ${MODE}. Expected one of: all, fewshot, zeroshot." >&2
    exit 1
    ;;
esac

python -m src.data.build_shot_manifests \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --window_stride "${WINDOW_STRIDE}" \
  --seed "${SEED}" \
  --max_train_items "${MAX_TRAIN_ITEMS}" \
  --normalization_source "${NORMALIZATION_SOURCE}" \
  --fewshot_ratios "${FEWSHOT_RATIOS}" \
  --source_ratios "${ZEROSHOT_SOURCE_RATIOS}" \
  --zeroshot_half_train_ratios "${ZEROSHOT_HALF_TRAIN_RATIOS}" \
  --zeroshot_test_window_divisor "${ZEROSHOT_TEST_WINDOW_DIVISOR}" \
  "${SKIP_FEWSHOT_ARGS[@]}" \
  "${SKIP_ZEROSHOT_ARGS[@]}"

echo "Shot preprocessing complete."
echo "  dataset: ${DATASET}"
echo "  horizon: ${HORIZON}"
echo "  mode: ${MODE}"
echo "  window_stride: ${WINDOW_STRIDE}"
echo "  normalization_source: ${NORMALIZATION_SOURCE}"
echo "  zeroshot_half_train_ratios: ${ZEROSHOT_HALF_TRAIN_RATIOS}"
echo "  zeroshot_test_window_divisor: ${ZEROSHOT_TEST_WINDOW_DIVISOR}"
