#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-renewable_solar}
HORIZON=${HORIZON:-16}
HISTORY_LEN=${HISTORY_LEN:-16}
CONTEXT_HISTORY_LEN=${CONTEXT_HISTORY_LEN:-672}
NEIGHBOR_K=${NEIGHBOR_K:-7}
WINDOW_STRIDE=${WINDOW_STRIDE:-16}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-2e-4}
MAX_LENGTH=${MAX_LENGTH:-2560}
PROMPT_STYLE=${PROMPT_STYLE:-cot}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/renewable_solar_shared_h16_hist16_ctx672}
RETRIEVAL_DEVICE=${RETRIEVAL_DEVICE:-auto}
RETRIEVAL_QUERY_BATCH_SIZE=${RETRIEVAL_QUERY_BATCH_SIZE:-128}
RETRIEVAL_CORPUS_CHUNK_SIZE=${RETRIEVAL_CORPUS_CHUNK_SIZE:-65536}

cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m src.train.train_shared_adapter \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --max_length "${MAX_LENGTH}" \
  --history_len "${HISTORY_LEN}" \
  --context_history_len "${CONTEXT_HISTORY_LEN}" \
  --neighbor_k "${NEIGHBOR_K}" \
  --window_stride "${WINDOW_STRIDE}" \
  --use_dora \
  --use_rag \
  --use_diff_dora \
  --prompt_style "${PROMPT_STYLE}" \
  --item_sampling shuffled_pairs \
  --max_train_items 0 \
  --max_eval 0 \
  --retrieval_device "${RETRIEVAL_DEVICE}" \
  --retrieval_query_batch_size "${RETRIEVAL_QUERY_BATCH_SIZE}" \
  --retrieval_corpus_chunk_size "${RETRIEVAL_CORPUS_CHUNK_SIZE}"