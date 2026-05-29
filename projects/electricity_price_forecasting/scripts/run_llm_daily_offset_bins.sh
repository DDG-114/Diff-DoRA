#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

OUT_DIR="${OUT_DIR:-outputs/gs_price_2025_llm_daily_offset_bins}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUT_DIR}_eval.json}"
MAX_DAYS="${MAX_DAYS:-10}"
MAX_TRAIN_DAYS="${MAX_TRAIN_DAYS:-120}"
NUM_BINS="${NUM_BINS:-5}"

.venv/bin/python projects/electricity_price_forecasting/scripts/train_llm_daily_offset_bins.py \
  --source_csv "data/GS(1).csv" \
  --candidate_csv "outputs/gs_price_2025_llm_candidate_map.csv" \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-2}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --grad_accum "${GRAD_ACCUM:-4}" \
  --lr "${LR:-1e-4}" \
  --max_length 2048 \
  --max_train_days "${MAX_TRAIN_DAYS}" \
  --num_bins "${NUM_BINS}"

.venv/bin/python projects/electricity_price_forecasting/scripts/eval_llm_daily_offset_bins.py \
  --source_csv "data/GS(1).csv" \
  --candidate_csv "outputs/gs_price_2025_llm_candidate_map.csv" \
  --adapter_dir "${OUT_DIR}/adapter" \
  --summary_json "${OUT_DIR}/summary.json" \
  --max_days "${MAX_DAYS}" \
  --max_new_tokens 32 \
  --output "${EVAL_OUTPUT}"
