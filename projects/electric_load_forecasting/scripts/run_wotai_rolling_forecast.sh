#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-wotai_evcdp}
HORIZON=${HORIZON:-6}
SPLIT=${SPLIT:-test}
WINDOW_STRIDE=${WINDOW_STRIDE:-6}
HISTORY_LEN=${HISTORY_LEN:-12}
NEIGHBOR_K=${NEIGHBOR_K:-7}
NODES_PER_DOMAIN=${NODES_PER_DOMAIN:-2}
MAX_WINDOWS=${MAX_WINDOWS:-96}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-192}
PROMPT_STYLE=${PROMPT_STYLE:-direct_physical}
START_TIME=${START_TIME:-}
END_TIME=${END_TIME:-}

RUN_ROOT=${RUN_ROOT:-outputs/wotai_rolling_forecast_$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_JSON="${RUN_ROOT}/rolling_forecast.json"
LOG_PATH="${RUN_ROOT}/rolling_forecast.log"

EXPERT_0_DIR=${EXPERT_0_DIR:-outputs/full_repro_st_evcdp_h6_bs48/full/expert_0/adapter}
EXPERT_1_DIR=${EXPERT_1_DIR:-outputs/full_repro_st_evcdp_h6_bs48/full/expert_1/adapter}
RETRIEVAL_CACHE=${RETRIEVAL_CACHE:-data/retrieval_cache/${DATASET}_h${HORIZON}_step${WINDOW_STRIDE}.pkl}

mkdir -p "${RUN_ROOT}"
source .venv/bin/activate

ARGS=(
  --dataset "${DATASET}"
  --horizon "${HORIZON}"
  --split "${SPLIT}"
  --history_len "${HISTORY_LEN}"
  --neighbor_k "${NEIGHBOR_K}"
  --window_stride "${WINDOW_STRIDE}"
  --nodes_per_domain "${NODES_PER_DOMAIN}"
  --max_windows "${MAX_WINDOWS}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --prompt_style "${PROMPT_STYLE}"
  --expert_0_dir "${EXPERT_0_DIR}"
  --expert_1_dir "${EXPERT_1_DIR}"
  --retrieval_cache "${RETRIEVAL_CACHE}"
  --output "${OUTPUT_JSON}"
)

if [ -n "${START_TIME}" ]; then
  ARGS+=(--start_time "${START_TIME}")
fi
if [ -n "${END_TIME}" ]; then
  ARGS+=(--end_time "${END_TIME}")
fi

echo "Starting Wotai rolling forecast."
echo "Run root: ${RUN_ROOT}"
echo "Output: ${OUTPUT_JSON}"
echo "Log: ${LOG_PATH}"
echo "Existing tasks are not stopped by this launcher."

nohup python -m src.eval.eval_wotai_rolling_forecast "${ARGS[@]}" > "${LOG_PATH}" 2>&1 &
PID=$!
echo "${PID}" > "${RUN_ROOT}/pid"
echo "Started PID ${PID}"
