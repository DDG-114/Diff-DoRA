#!/usr/bin/env bash
# Strict reproduction launcher:
# - full expert training pools
# - full expert-local retrieval banks
# - full test split / all-node evaluation
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
BATCH_SIZE=${3:-16}
SAMPLE_CAP=${4:-0}
RETR_BANK_CAP=${5:-$SAMPLE_CAP}

if [ "$SAMPLE_CAP" -le 0 ] 2>/dev/null; then
  OUTDIR="outputs/strict_repro_${DATASET}_h${HORIZON}"
else
  OUTDIR="outputs/strict_repro_${DATASET}_h${HORIZON}_cap${SAMPLE_CAP}"
fi
CACHE_PATH="data/retrieval_cache/${DATASET}_h${HORIZON}.pkl"

source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTDIR"

if [ ! -f "$CACHE_PATH" ]; then
  echo "[INFO] Retrieval cache not found. Building: $CACHE_PATH"
  python -m src.retrieval.build_cache \
    --datasets "$DATASET" \
    --horizons "$HORIZON" \
    --output_path "$CACHE_PATH"
fi

COMMON_TRAIN_ARGS=(
  --dataset "$DATASET"
  --horizon "$HORIZON"
  --epochs 2
  --batch_size "$BATCH_SIZE"
  --lr 2e-4
  --max_length 2560
  --lora_rank 32
  --lora_alpha 32
  --history_len 12
  --neighbor_k 7
  --use_dora
  --use_rag
  --prompt_style cot
  --retrieval_cache "$CACHE_PATH"
  --max_samples_per_expert "$SAMPLE_CAP"
  --retrieval_bank_max_samples_per_expert "$RETR_BANK_CAP"
  --eval_max_samples 0
  --eval_nodes_per_expert 0
)

python -m src.train.train_experts \
  "${COMMON_TRAIN_ARGS[@]}" \
  --use_diff_dora \
  --output_dir "$OUTDIR/full"

python -m src.train.train_experts \
  "${COMMON_TRAIN_ARGS[@]}" \
  --output_dir "$OUTDIR/wo_diffdora"

bash scripts/run_eval_diffdora_strict.sh "$DATASET" "$HORIZON" "$OUTDIR"

echo "Strict train + eval complete."
echo "Train dir: $OUTDIR/full"
echo "Train dir: $OUTDIR/wo_diffdora"
echo "Eval json: $OUTDIR/expert_ablation_full_test.json"
