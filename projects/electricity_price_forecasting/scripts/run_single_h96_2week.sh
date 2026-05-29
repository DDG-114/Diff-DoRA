#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# Single-adapter electricity-price fine-tuning.
# Frequency is 15 minutes, so:
#   horizon=96              -> 1 day ahead
#   history_len=96          -> last day, shown point-by-point in the prompt
#   context_history_len=1344 -> last 14 days, summarized as long-range context

OUT_DIR="${OUT_DIR:-outputs/gs_price_2025_single_h96_2week_direct_v1}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUT_DIR}_eval_full.json}"

.venv/bin/python -m src.train.train_shared_adapter \
  --dataset gs_price_2025 \
  --horizon 96 \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --grad_accum "${GRAD_ACCUM:-8}" \
  --lr "${LR:-1e-4}" \
  --history_len 96 \
  --context_history_len 1344 \
  --neighbor_k 0 \
  --window_stride 24 \
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
  --max_new_tokens 1024 \
  --retrieval_device auto

.venv/bin/python -m src.eval.eval_saved_adapter \
  --dataset gs_price_2025 \
  --horizon 96 \
  --adapter_dir "${OUT_DIR}/adapter" \
  --node_id Price \
  --history_len 96 \
  --context_history_len 1344 \
  --neighbor_k 0 \
  --window_stride 96 \
  --use_rag \
  --use_diff_dora \
  --prompt_style direct_physical \
  --max_eval 0 \
  --sampling head \
  --max_new_tokens 1024 \
  --output "${EVAL_OUTPUT}"
