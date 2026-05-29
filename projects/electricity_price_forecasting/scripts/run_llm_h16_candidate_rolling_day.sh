#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# LLM short-horizon refinement with a candidate day-ahead skeleton.
# The candidate curve currently comes from the strongest non-LLM baseline and is
# injected into the prompt as a refinement target.

OUT_DIR="${OUT_DIR:-outputs/gs_price_2025_llm_h16_candidate_v1}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUT_DIR}_rolling_day96.json}"
CANDIDATE_CSV="${CANDIDATE_CSV:-outputs/gs_price_2025_llm_candidate_map.csv}"
MAX_DAYS="${MAX_DAYS:-0}"
ROLL_MAX_NEW_TOKENS="${ROLL_MAX_NEW_TOKENS:-128}"
RESIDUAL_CLIP="${RESIDUAL_CLIP:-80}"

.venv/bin/python -m src.train.train_shared_adapter \
  --dataset gs_price_2025 \
  --horizon 16 \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-2}" \
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
  --candidate_prediction_csv "${CANDIDATE_CSV}" \
  --target_style candidate_chunk_offset \
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
  --max_new_tokens "${ROLL_MAX_NEW_TOKENS}" \
  --allow_short_padding \
  --max_days "${MAX_DAYS}" \
  --candidate_prediction_csv "${CANDIDATE_CSV}" \
  --candidate_mode chunk_offset \
  --candidate_residual_clip "${RESIDUAL_CLIP}" \
  --candidate_value_min 0 \
  --candidate_value_max 1000 \
  --output "${EVAL_OUTPUT}"
