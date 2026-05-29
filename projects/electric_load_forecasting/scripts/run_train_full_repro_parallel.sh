#!/usr/bin/env bash
# Full-data dual-GPU training launcher:
# - GPU 0 trains full (prompt-only Diff-DoRA)
# - GPU 1 trains wo_diffdora
# - no mid-train validation
# - no automatic evaluation after training
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
BATCH_SIZE=${3:-16}
SAMPLE_CAP=${4:-0}
RETR_BANK_CAP=${5:-$SAMPLE_CAP}
HISTORY_LEN=12
NEIGHBOR_K=7
WINDOW_STRIDE=${WINDOW_STRIDE:-$HORIZON}
MAX_LENGTH=2560
PROMPT_STYLE=cot
CACHE_ROOT=${DIFFDORA_CACHE_ROOT:-/root/autodl-tmp/Diff-DoRA-cache}

if [ "$SAMPLE_CAP" -le 0 ] 2>/dev/null; then
  OUTDIR="outputs/full_repro_${DATASET}_h${HORIZON}_bs${BATCH_SIZE}"
else
  OUTDIR="outputs/full_repro_${DATASET}_h${HORIZON}_bs${BATCH_SIZE}_cap${SAMPLE_CAP}"
fi
CACHE_PATH="data/retrieval_cache/${DATASET}_h${HORIZON}_step${WINDOW_STRIDE}.pkl"
SAMPLE_CACHE_PATH="${CACHE_ROOT}/sample_cache/train_experts_${DATASET}_h${HORIZON}_hist${HISTORY_LEN}_nbr${NEIGHBOR_K}_step${WINDOW_STRIDE}.pkl"
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
TOKENIZED_CACHE_DIR_FULL="${TOKENIZED_CACHE_ROOT}/full"
TOKENIZED_CACHE_DIR_WO="${TOKENIZED_CACHE_ROOT}/wo_diffdora"

source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFDORA_CACHE_ROOT="${CACHE_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; this launcher requires two visible NVIDIA GPUs." >&2
  exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [ "${GPU_COUNT}" -lt 2 ]; then
  echo "Expected at least 2 visible GPUs, found ${GPU_COUNT}." >&2
  exit 1
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "Detected active GPU compute processes. Please free both GPUs before starting full-data training." >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >&2 || true
  exit 1
fi

if [ ! -f "${CACHE_PATH}" ]; then
  echo "Retrieval cache not found: ${CACHE_PATH}" >&2
  echo "Build it first or run a cache-building script before this launcher." >&2
  exit 1
fi

if [ -d "${OUTDIR}" ] && find "${OUTDIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Output directory already exists and is non-empty: ${OUTDIR}" >&2
  echo "Refusing to overwrite existing full-data run artifacts." >&2
  exit 1
fi

mkdir -p "${OUTDIR}/logs"

COMMON_TRAIN_ARGS=(
  --dataset "${DATASET}"
  --horizon "${HORIZON}"
  --epochs 2
  --batch_size "${BATCH_SIZE}"
  --lr 2e-4
  --max_length "${MAX_LENGTH}"
  --lora_rank 32
  --lora_alpha 32
  --history_len "${HISTORY_LEN}"
  --neighbor_k "${NEIGHBOR_K}"
  --window_stride "${WINDOW_STRIDE}"
  --use_dora
  --use_rag
  --prompt_style "${PROMPT_STYLE}"
  --retrieval_cache "${CACHE_PATH}"
  --sample_cache "${SAMPLE_CACHE_PATH}"
  --max_samples_per_expert "${SAMPLE_CAP}"
  --retrieval_bank_max_samples_per_expert "${RETR_BANK_CAP}"
  --eval_steps 0
  --eval_max_samples 0
  --eval_nodes_per_expert 0
)

python -m src.data.build_sample_cache \
  --datasets "${DATASET}" \
  --horizons "${HORIZON}" \
  --history_len "${HISTORY_LEN}" \
  --neighbor_k "${NEIGHBOR_K}" \
  --window_stride "${WINDOW_STRIDE}" \
  --output_path "${SAMPLE_CACHE_PATH}"

echo "Starting full-data dual-GPU training."
echo "Output root: ${OUTDIR}"
echo "Batch size per device: ${BATCH_SIZE}"
echo "Sample cap per expert: ${SAMPLE_CAP}"
echo "Retrieval-bank cap per expert: ${RETR_BANK_CAP}"
echo "Window stride: ${WINDOW_STRIDE}"
echo "Cache: ${CACHE_PATH}"
echo "Sample cache: ${SAMPLE_CACHE_PATH}"
echo "Cache root: ${CACHE_ROOT}"

FULL_EXTRA_ARGS=()
WO_EXTRA_ARGS=()
if [ -f "${TOKENIZED_CACHE_DIR_FULL}/expert_0_train.pkl" ] && \
   [ -f "${TOKENIZED_CACHE_DIR_FULL}/expert_0_val.pkl" ] && \
   [ -f "${TOKENIZED_CACHE_DIR_FULL}/expert_1_train.pkl" ] && \
   [ -f "${TOKENIZED_CACHE_DIR_FULL}/expert_1_val.pkl" ]; then
  FULL_EXTRA_ARGS=(--tokenized_cache_dir "${TOKENIZED_CACHE_DIR_FULL}")
  echo "Using pretokenized cache for full: ${TOKENIZED_CACHE_DIR_FULL}"
fi
if [ -f "${TOKENIZED_CACHE_DIR_WO}/expert_0_train.pkl" ] && \
   [ -f "${TOKENIZED_CACHE_DIR_WO}/expert_0_val.pkl" ] && \
   [ -f "${TOKENIZED_CACHE_DIR_WO}/expert_1_train.pkl" ] && \
   [ -f "${TOKENIZED_CACHE_DIR_WO}/expert_1_val.pkl" ]; then
  WO_EXTRA_ARGS=(--tokenized_cache_dir "${TOKENIZED_CACHE_DIR_WO}")
  echo "Using pretokenized cache for wo_diffdora: ${TOKENIZED_CACHE_DIR_WO}"
fi

CUDA_VISIBLE_DEVICES=0 python -m src.train.train_experts \
  "${COMMON_TRAIN_ARGS[@]}" \
  "${FULL_EXTRA_ARGS[@]}" \
  --use_diff_dora \
  --output_dir "${OUTDIR}/full" \
  > "${OUTDIR}/logs/full_gpu0.log" 2>&1 &
PID_FULL=$!

CUDA_VISIBLE_DEVICES=1 python -m src.train.train_experts \
  "${COMMON_TRAIN_ARGS[@]}" \
  "${WO_EXTRA_ARGS[@]}" \
  --output_dir "${OUTDIR}/wo_diffdora" \
  > "${OUTDIR}/logs/wo_diffdora_gpu1.log" 2>&1 &
PID_WO=$!

echo "full PID: ${PID_FULL} (GPU 0)"
echo "wo_diffdora PID: ${PID_WO} (GPU 1)"
echo "Logs:"
echo "  ${OUTDIR}/logs/full_gpu0.log"
echo "  ${OUTDIR}/logs/wo_diffdora_gpu1.log"

STATUS=0

if ! wait "${PID_FULL}"; then
  STATUS=1
  echo "full training failed. Tail of log:" >&2
  tail -n 50 "${OUTDIR}/logs/full_gpu0.log" >&2 || true
fi

if ! wait "${PID_WO}"; then
  STATUS=1
  echo "wo_diffdora training failed. Tail of log:" >&2
  tail -n 50 "${OUTDIR}/logs/wo_diffdora_gpu1.log" >&2 || true
fi

if [ "${STATUS}" -ne 0 ]; then
  exit "${STATUS}"
fi

echo "Full-data training complete."
echo "Artifacts:"
echo "  ${OUTDIR}/full"
echo "  ${OUTDIR}/wo_diffdora"
