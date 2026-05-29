#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
SAMPLE_CAP=${3:-0}
RETR_BANK_CAP=${4:-$SAMPLE_CAP}
NUM_WORKERS=${5:-2}
MATERIALIZE_CHUNK_SIZE=${6:-5000}
TOKENIZED_CACHE_SHARD_SIZE=${7:-25000}
RETR_QUERY_BATCH_SIZE=${8:-128}
RETR_CORPUS_CHUNK_SIZE=${9:-65536}
DISABLE_CACHE_SHARDING=${DIFFDORA_DISABLE_CACHE_SHARDING:-1}
HISTORY_LEN=12
NEIGHBOR_K=7
WINDOW_STRIDE=${WINDOW_STRIDE:-$HORIZON}
MAX_LENGTH=2560
PROMPT_STYLE=cot
CACHE_ROOT=${DIFFDORA_CACHE_ROOT:-/root/autodl-tmp/Diff-DoRA-cache}

if [ "$SAMPLE_CAP" -le 0 ] 2>/dev/null; then
  SAMPLE_TAG="samplesfull"
else
  SAMPLE_TAG="samples${SAMPLE_CAP}"
fi
if [ "$RETR_BANK_CAP" -le 0 ] 2>/dev/null; then
  BANK_TAG="bankfull"
else
  BANK_TAG="bank${RETR_BANK_CAP}"
fi
TOKENIZED_CACHE_ROOT="${CACHE_ROOT}/tokenized_cache/train_experts_${DATASET}_h${HORIZON}_hist${HISTORY_LEN}_nbr${NEIGHBOR_K}_step${WINDOW_STRIDE}_len${MAX_LENGTH}_${PROMPT_STYLE}_${SAMPLE_TAG}_${BANK_TAG}"
RETRIEVAL_CACHE_ROOT="${CACHE_ROOT}/retrieval_result_cache/train_experts_${DATASET}_h${HORIZON}_hist${HISTORY_LEN}_nbr${NEIGHBOR_K}_step${WINDOW_STRIDE}_${PROMPT_STYLE}_${SAMPLE_TAG}_${BANK_TAG}"
SAMPLE_CACHE_PATH="${CACHE_ROOT}/sample_cache/train_experts_${DATASET}_h${HORIZON}_hist${HISTORY_LEN}_nbr${NEIGHBOR_K}_step${WINDOW_STRIDE}.pkl"

source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

export DIFFDORA_CACHE_ROOT="${CACHE_ROOT}"

RETRIEVAL_DEVICE_FULL="cpu"
RETRIEVAL_DEVICE_WO="cpu"
EFFECTIVE_RETR_QUERY_BATCH_SIZE_FULL="${RETR_QUERY_BATCH_SIZE}"
EFFECTIVE_RETR_QUERY_BATCH_SIZE_WO="${RETR_QUERY_BATCH_SIZE}"
GPU_RETRIEVAL_ENABLED=${DIFFDORA_ENABLE_GPU_RETRIEVAL:-1}
if [ "${GPU_RETRIEVAL_ENABLED}" != "0" ] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  if [ "${GPU_COUNT}" -ge 2 ]; then
    RETRIEVAL_DEVICE_FULL="cuda:0"
    RETRIEVAL_DEVICE_WO="cuda:1"
    EFFECTIVE_RETR_QUERY_BATCH_SIZE_FULL=$(( RETR_QUERY_BATCH_SIZE * 2 ))
    EFFECTIVE_RETR_QUERY_BATCH_SIZE_WO=$(( RETR_QUERY_BATCH_SIZE * 2 ))
  elif [ "${GPU_COUNT}" -ge 1 ] && [ "${DIFFDORA_ALLOW_SINGLE_GPU_RETRIEVAL:-0}" = "1" ]; then
    RETRIEVAL_DEVICE_FULL="cuda:0"
    RETRIEVAL_DEVICE_WO="cuda:0"
    EFFECTIVE_RETR_QUERY_BATCH_SIZE_FULL=$(( RETR_QUERY_BATCH_SIZE * 2 ))
    EFFECTIVE_RETR_QUERY_BATCH_SIZE_WO=$(( RETR_QUERY_BATCH_SIZE * 2 ))
  fi
fi

echo "Retrieval devices:"
echo "  full: ${RETRIEVAL_DEVICE_FULL}"
echo "  wo_diffdora: ${RETRIEVAL_DEVICE_WO}"
echo "Retrieval batch settings:"
echo "  requested_query_batch_size: ${RETR_QUERY_BATCH_SIZE}"
echo "  effective_query_batch_size_full: ${EFFECTIVE_RETR_QUERY_BATCH_SIZE_FULL}"
echo "  effective_query_batch_size_wo_diffdora: ${EFFECTIVE_RETR_QUERY_BATCH_SIZE_WO}"
echo "  corpus_chunk_size: ${RETR_CORPUS_CHUNK_SIZE}"
echo "Cache sharding:"
echo "  disabled: ${DISABLE_CACHE_SHARDING}"

DISABLE_CACHE_SHARDING_ARGS=()
if [ "${DISABLE_CACHE_SHARDING}" = "1" ]; then
  DISABLE_CACHE_SHARDING_ARGS+=(--disable_cache_sharding)
fi

python -m src.data.build_sample_cache \
  --datasets "${DATASET}" \
  --horizons "${HORIZON}" \
  --history_len "${HISTORY_LEN}" \
  --neighbor_k "${NEIGHBOR_K}" \
  --window_stride "${WINDOW_STRIDE}" \
  --output_path "${SAMPLE_CACHE_PATH}"

python -m src.train.build_tokenized_expert_cache \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --history_len "${HISTORY_LEN}" \
  --neighbor_k "${NEIGHBOR_K}" \
  --window_stride "${WINDOW_STRIDE}" \
  --max_length "${MAX_LENGTH}" \
  --prompt_style "${PROMPT_STYLE}" \
  --variant full \
  --sample_cache "${SAMPLE_CACHE_PATH}" \
  --retrieval_cache_dir "${RETRIEVAL_CACHE_ROOT}/full" \
  --max_samples_per_expert "${SAMPLE_CAP}" \
  --retrieval_bank_max_samples_per_expert "${RETR_BANK_CAP}" \
  --retrieval_device "${RETRIEVAL_DEVICE_FULL}" \
  --retrieval_query_batch_size "${EFFECTIVE_RETR_QUERY_BATCH_SIZE_FULL}" \
  --retrieval_corpus_chunk_size "${RETR_CORPUS_CHUNK_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --materialize_chunk_size "${MATERIALIZE_CHUNK_SIZE}" \
  --tokenized_cache_shard_size "${TOKENIZED_CACHE_SHARD_SIZE}" \
  "${DISABLE_CACHE_SHARDING_ARGS[@]}" \
  --output_dir "${TOKENIZED_CACHE_ROOT}/full" &
PID_FULL=$!

python -m src.train.build_tokenized_expert_cache \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --history_len "${HISTORY_LEN}" \
  --neighbor_k "${NEIGHBOR_K}" \
  --window_stride "${WINDOW_STRIDE}" \
  --max_length "${MAX_LENGTH}" \
  --prompt_style "${PROMPT_STYLE}" \
  --variant wo_diffdora \
  --sample_cache "${SAMPLE_CACHE_PATH}" \
  --retrieval_cache_dir "${RETRIEVAL_CACHE_ROOT}/wo_diffdora" \
  --max_samples_per_expert "${SAMPLE_CAP}" \
  --retrieval_bank_max_samples_per_expert "${RETR_BANK_CAP}" \
  --retrieval_device "${RETRIEVAL_DEVICE_WO}" \
  --retrieval_query_batch_size "${EFFECTIVE_RETR_QUERY_BATCH_SIZE_WO}" \
  --retrieval_corpus_chunk_size "${RETR_CORPUS_CHUNK_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --materialize_chunk_size "${MATERIALIZE_CHUNK_SIZE}" \
  --tokenized_cache_shard_size "${TOKENIZED_CACHE_SHARD_SIZE}" \
  "${DISABLE_CACHE_SHARDING_ARGS[@]}" \
  --output_dir "${TOKENIZED_CACHE_ROOT}/wo_diffdora" &
PID_WO=$!

wait "${PID_FULL}"
wait "${PID_WO}"

echo "Pretokenized expert caches ready:"
echo "  ${TOKENIZED_CACHE_ROOT}/full"
echo "  ${TOKENIZED_CACHE_ROOT}/wo_diffdora"
echo "Retrieval-result caches ready:"
echo "  ${RETRIEVAL_CACHE_ROOT}/full"
echo "  ${RETRIEVAL_CACHE_ROOT}/wo_diffdora"
