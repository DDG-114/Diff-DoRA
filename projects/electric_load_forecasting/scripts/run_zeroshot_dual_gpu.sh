#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
STAMP=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT="outputs/zeroshot_moe_${DATASET}_h${HORIZON}_${STAMP}"

source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
BATCH_SIZE=${BATCH_SIZE:-48}
RETRIEVAL_DEVICE=${RETRIEVAL_DEVICE:-auto}
RETRIEVAL_QUERY_BATCH_SIZE=${RETRIEVAL_QUERY_BATCH_SIZE:-256}
RETRIEVAL_CORPUS_CHUNK_SIZE=${RETRIEVAL_CORPUS_CHUNK_SIZE:-65536}

mkdir -p "${RUN_ROOT}/logs"

run_ratio() {
  local gpu="$1"
  local ratio="$2"
  local outdir="${RUN_ROOT}/ratio_${ratio}"
  local logfile="${RUN_ROOT}/logs/ratio_${ratio}_gpu${gpu}.log"

  echo "[gpu${gpu}] starting ratio ${ratio}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m src.eval.eval_zeroshot \
    --strict_protocol \
    --dataset "${DATASET}" \
    --horizon "${HORIZON}" \
    --source_ratios "${ratio}" \
    --output_dir "${outdir}" \
    --batch_size "${BATCH_SIZE}" \
    --max_train_items 4000 \
    --use_dora \
    --use_diff_dora \
    --use_rag \
    --prompt_style cot \
    --retrieval_device "${RETRIEVAL_DEVICE}" \
    --retrieval_query_batch_size "${RETRIEVAL_QUERY_BATCH_SIZE}" \
    --retrieval_corpus_chunk_size "${RETRIEVAL_CORPUS_CHUNK_SIZE}" \
    > "${logfile}" 2>&1
  echo "[gpu${gpu}] finished ratio ${ratio}"
}

(
  run_ratio 0 0.20
  run_ratio 0 0.60
) &
PID_GPU0=$!

(
  run_ratio 1 0.40
  run_ratio 1 0.80
) &
PID_GPU1=$!

echo "Zero-shot MoE dual-GPU queues started."
echo "Run root: ${RUN_ROOT}"
echo "GPU0 queue PID: ${PID_GPU0}  ratios: 0.20 -> 0.60"
echo "GPU1 queue PID: ${PID_GPU1}  ratios: 0.40 -> 0.80"
echo "Logs:"
echo "  ${RUN_ROOT}/logs/ratio_0.20_gpu0.log"
echo "  ${RUN_ROOT}/logs/ratio_0.60_gpu0.log"
echo "  ${RUN_ROOT}/logs/ratio_0.40_gpu1.log"
echo "  ${RUN_ROOT}/logs/ratio_0.80_gpu1.log"
echo "Batch size: ${BATCH_SIZE}"
echo "Retrieval backend: ${RETRIEVAL_DEVICE} (query_batch_size=${RETRIEVAL_QUERY_BATCH_SIZE}, corpus_chunk_size=${RETRIEVAL_CORPUS_CHUNK_SIZE})"

wait "${PID_GPU0}"
wait "${PID_GPU1}"

echo "All zero-shot MoE queues completed."
