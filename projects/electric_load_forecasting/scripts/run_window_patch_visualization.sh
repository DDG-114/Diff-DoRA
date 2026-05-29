#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
RUN_ROOT=${3:-"outputs/full_repro_${DATASET}_h${HORIZON}_bs16_cap10000"}
OUTPUT_DIR="${RUN_ROOT}/window_patches"
ABLATION_JSON="${RUN_ROOT}/expert_ablation_eval_max60.json"
RETRIEVAL_CACHE="data/retrieval_cache/${DATASET}_h${HORIZON}.pkl"

source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

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
  echo "Detected active GPU compute processes. Please free both GPUs before starting patch visualization." >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >&2 || true
  exit 1
fi

if [ ! -f "${ABLATION_JSON}" ]; then
  echo "Missing ablation json: ${ABLATION_JSON}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

SELECTION_JSON="${OUTPUT_DIR}/window_patch_selection.json"
FULL_JSON="${OUTPUT_DIR}/full_window_cases_gpu0.json"
WO_JSON="${OUTPUT_DIR}/wo_window_cases_gpu1.json"

python -m src.analysis.prepare_window_case_selection \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --ablation-json "${ABLATION_JSON}" \
  --output "${SELECTION_JSON}"

CUDA_VISIBLE_DEVICES=0 python -m src.analysis.eval_window_patches_variant \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --use_rag \
  --use_diff_dora \
  --prompt_style cot \
  --retrieval_cache "${RETRIEVAL_CACHE}" \
  --expert_0_dir "${RUN_ROOT}/full/expert_0/adapter" \
  --expert_1_dir "${RUN_ROOT}/full/expert_1/adapter" \
  --selection-json "${SELECTION_JSON}" \
  --infer_batch_size 64 \
  --max_new_tokens 512 \
  --output "${FULL_JSON}" &
PID_FULL=$!

CUDA_VISIBLE_DEVICES=1 python -m src.analysis.eval_window_patches_variant \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --use_rag \
  --prompt_style cot \
  --retrieval_cache "${RETRIEVAL_CACHE}" \
  --expert_0_dir "${RUN_ROOT}/wo_diffdora/expert_0/adapter" \
  --expert_1_dir "${RUN_ROOT}/wo_diffdora/expert_1/adapter" \
  --selection-json "${SELECTION_JSON}" \
  --infer_batch_size 64 \
  --max_new_tokens 512 \
  --output "${WO_JSON}" &
PID_WO=$!

echo "Selection: ${SELECTION_JSON}"
echo "full PID: ${PID_FULL} (GPU 0)"
echo "wo_diffdora PID: ${PID_WO} (GPU 1)"

wait "${PID_FULL}"
wait "${PID_WO}"

python -m src.analysis.visualize_window_patches \
  --selection-json "${SELECTION_JSON}" \
  --full-json "${FULL_JSON}" \
  --wo-json "${WO_JSON}" \
  --output-dir "${OUTPUT_DIR}"

echo "Window patch visualization complete:"
echo "  ${OUTPUT_DIR}/cbd_scheme_a.svg"
echo "  ${OUTPUT_DIR}/cbd_scheme_b.svg"
echo "  ${OUTPUT_DIR}/residential_scheme_a.svg"
echo "  ${OUTPUT_DIR}/residential_scheme_b.svg"
echo "  ${OUTPUT_DIR}/window_patch_summary.json"
