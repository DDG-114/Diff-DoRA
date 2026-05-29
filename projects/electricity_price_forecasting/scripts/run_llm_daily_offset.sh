#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

OUT_DIR="${OUT_DIR:-outputs/gs_price_2025_llm_daily_offset}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUT_DIR}_eval.json}"
MAX_DAYS="${MAX_DAYS:-10}"
MAX_TRAIN_DAYS="${MAX_TRAIN_DAYS:-0}"

.venv/bin/python projects/electricity_price_forecasting/scripts/train_llm_daily_offset.py \
  --source_csv "data/GS(1).csv" \
  --candidate_csv "outputs/gs_price_2025_llm_candidate_map.csv" \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --grad_accum "${GRAD_ACCUM:-8}" \
  --lr "${LR:-1e-4}" \
  --max_length 2048 \
  --max_train_days "${MAX_TRAIN_DAYS}"

.venv/bin/python projects/electricity_price_forecasting/scripts/eval_llm_daily_offset.py \
  --source_csv "data/GS(1).csv" \
  --candidate_csv "outputs/gs_price_2025_llm_candidate_map.csv" \
  --adapter_dir "${OUT_DIR}/adapter" \
  --max_days "${MAX_DAYS}" \
  --max_new_tokens 32 \
  --output "${EVAL_OUTPUT}"
