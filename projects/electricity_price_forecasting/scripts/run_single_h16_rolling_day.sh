#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# Short-horizon adapter plus autoregressive rolling-day evaluation.
# Frequency is 15 minutes:
#   horizon=16              -> 4 hours per generation
#   6 chunks * 16 points    -> 96 points = 1 day
#   history_len=96          -> last day point-by-point
#   context_history_len=1344 -> last 14 days summarized

OUT_DIR="${OUT_DIR:-outputs/gs_price_2025_single_h16_2week_direct_v1}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUT_DIR}_rolling_day96.json}"

.venv/bin/python -m src.train.train_shared_adapter \
  --dataset gs_price_2025 \
  --horizon 16 \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --grad_accum "${GRAD_ACCUM:-8}" \
  --lr "${LR:-1e-4}" \
  --history_len 96 \
  --context_history_len 1344 \
  --neighbor_k 0 \
  --window_stride 16 \
  --use_dora \
  --use_rag \
  --use_diff_dora \
  --prompt_style direct_physical \
  --target_style numeric_only \
  --item_sampling shuffled_pairs \
  --active_selection price_dynamic \
  --active_budget_ratio "${ACTIVE_BUDGET_RATIO:-0.8}" \
  --max_train_items "${MAX_TRAIN_ITEMS:-0}" \
  --max_eval 0 \
  --max_length 4096 \
  --max_new_tokens 256 \
  --retrieval_device auto

.venv/bin/python projects/electricity_price_forecasting/scripts/eval_gs_price_rolling_day.py \
  --dataset gs_price_2025 \
  --adapter_dir "${OUT_DIR}/adapter" \
  --node_id Price \
  --horizon 16 \
  --day_horizon 96 \
  --history_len 96 \
  --context_history_len 1344 \
  --neighbor_k 0 \
  --day_stride 96 \
  --retrieval_stride 16 \
  --use_rag \
  --use_diff_dora \
  --prompt_style direct_physical \
  --max_new_tokens 256 \
  --output "${EVAL_OUTPUT}"
