#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
RUN_ROOT=${3:-"outputs/full_repro_${DATASET}_h${HORIZON}_bs16_cap10000"}
OUTPUT_DIR="${RUN_ROOT}/daily_curves"
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
  echo "Detected active GPU compute processes. Please free both GPUs before starting weekly curve visualization." >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >&2 || true
  exit 1
fi

if [ ! -d "${RUN_ROOT}/full/expert_0/adapter" ] || [ ! -d "${RUN_ROOT}/full/expert_1/adapter" ]; then
  echo "Missing full expert adapters under ${RUN_ROOT}/full" >&2
  exit 1
fi
if [ ! -d "${RUN_ROOT}/wo_diffdora/expert_0/adapter" ] || [ ! -d "${RUN_ROOT}/wo_diffdora/expert_1/adapter" ]; then
  echo "Missing wo_diffdora expert adapters under ${RUN_ROOT}/wo_diffdora" >&2
  exit 1
fi
if [ ! -f "${RETRIEVAL_CACHE}" ]; then
  echo "Missing retrieval cache: ${RETRIEVAL_CACHE}" >&2
  exit 1
fi

read -r CBD_NODES RES_NODES < <(
  DATASET_FOR_SELECT="${DATASET}" \
  python - <<'PY'
import os
import io
from contextlib import redirect_stdout
from src.analysis.visualize_weekly_curves import _select_nodes
buf = io.StringIO()
with redirect_stdout(buf):
    sel = _select_nodes(os.environ["DATASET_FOR_SELECT"])
print(",".join(str(v) for v in sel["CBD"]), ",".join(str(v) for v in sel["Residential"]))
PY
)

mkdir -p "${OUTPUT_DIR}"

FULL_JSON="${OUTPUT_DIR}/full_weekly_gpu0.json"
WO_JSON="${OUTPUT_DIR}/wo_diffdora_weekly_gpu1.json"

CUDA_VISIBLE_DEVICES=0 python -m src.analysis.eval_weekly_curve_variant \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --use_rag \
  --use_diff_dora \
  --prompt_style cot \
  --retrieval_cache "${RETRIEVAL_CACHE}" \
  --expert_0_dir "${RUN_ROOT}/full/expert_0/adapter" \
  --expert_1_dir "${RUN_ROOT}/full/expert_1/adapter" \
  --cbd_nodes "${CBD_NODES}" \
  --res_nodes "${RES_NODES}" \
  --max_points 288 \
  --infer_batch_size 32 \
  --max_new_tokens 512 \
  --output "${FULL_JSON}" &
PID_FULL=$!

CUDA_VISIBLE_DEVICES=1 python -m src.analysis.eval_weekly_curve_variant \
  --dataset "${DATASET}" \
  --horizon "${HORIZON}" \
  --use_rag \
  --prompt_style cot \
  --retrieval_cache "${RETRIEVAL_CACHE}" \
  --expert_0_dir "${RUN_ROOT}/wo_diffdora/expert_0/adapter" \
  --expert_1_dir "${RUN_ROOT}/wo_diffdora/expert_1/adapter" \
  --cbd_nodes "${CBD_NODES}" \
  --res_nodes "${RES_NODES}" \
  --max_points 288 \
  --infer_batch_size 32 \
  --max_new_tokens 512 \
  --output "${WO_JSON}" &
PID_WO=$!

echo "Selected nodes: CBD=${CBD_NODES} Residential=${RES_NODES}"
echo "full PID: ${PID_FULL} (GPU 0)"
echo "wo_diffdora PID: ${PID_WO} (GPU 1)"

wait "${PID_FULL}"
wait "${PID_WO}"

python -m src.analysis.visualize_weekly_curves \
  --dataset "${DATASET}" \
  --full-json "${FULL_JSON}" \
  --wo-json "${WO_JSON}" \
  --output-dir "${OUTPUT_DIR}"

echo "Weekly curve visualization complete:"
echo "  ${OUTPUT_DIR}/cbd_weekly_curve.svg"
echo "  ${OUTPUT_DIR}/residential_weekly_curve.svg"
echo "  ${OUTPUT_DIR}/weekly_curve_summary.json"
