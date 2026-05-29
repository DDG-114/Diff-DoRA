#!/usr/bin/env bash
# Legacy baseline launcher:
# - quick expert training with per-expert sample caps
# - expert-local retrieval-bank truncation
# - quick ablation eval with max_eval defaulting to 6
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
MAX_EVAL=${3:-6}

OUTDIR="outputs/train_eval_diffdora_${DATASET}_h${HORIZON}"
CACHE_PATH="data/retrieval_cache/${DATASET}_h${HORIZON}.pkl"

source .venv/bin/activate

# Avoid libgomp warnings from invalid container defaults.
if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

# Reduce fragmentation risk during large-sequence training.
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
  --batch_size 8
  --lr 2e-4
  --max_length 2560
  --lora_rank 32
  --lora_alpha 32
  --history_len 12
  --neighbor_k 7
  --use_dora
  --use_diff_dora
  --use_rag
  --prompt_style cot
  --retrieval_cache "$CACHE_PATH"
  --max_samples_per_expert 1000
  --eval_max_samples 0
  --eval_nodes_per_expert 0
)

python -m src.train.train_experts \
  "${COMMON_TRAIN_ARGS[@]}" \
  --output_dir "$OUTDIR/full"

python -m src.train.train_experts \
  --dataset "$DATASET" \
  --horizon "$HORIZON" \
  --epochs 2 \
  --batch_size 8 \
  --lr 2e-4 \
  --max_length 2560 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --history_len 12 \
  --neighbor_k 7 \
  --use_dora \
  --use_rag \
  --prompt_style cot \
  --retrieval_cache "$CACHE_PATH" \
  --max_samples_per_expert 1000 \
  --eval_max_samples 0 \
  --eval_nodes_per_expert 0 \
  --output_dir "$OUTDIR/wo_diffdora"

python -m src.eval.eval_paper_ablation \
  --dataset "$DATASET" \
  --horizon "$HORIZON" \
  --use_rag \
  --retrieval_cache "$CACHE_PATH" \
  --max_eval "$MAX_EVAL" \
  --sampling random \
  --seed 42 \
  --node_sampling balanced_random \
  --max_nodes_per_domain 12 \
  --max_new_tokens 512 \
  --infer_batch_size 12 \
  --skip_base_model \
  --full_expert_0_dir "$OUTDIR/full/expert_0/adapter" \
  --full_expert_1_dir "$OUTDIR/full/expert_1/adapter" \
  --wo_diffdora_expert_0_dir "$OUTDIR/wo_diffdora/expert_0/adapter" \
  --wo_diffdora_expert_1_dir "$OUTDIR/wo_diffdora/expert_1/adapter" \
  --output "$OUTDIR/expert_ablation_eval.json"

echo "Legacy baseline train + eval complete."
echo "Train dir: $OUTDIR/full"
echo "Train dir: $OUTDIR/wo_diffdora"
echo "Eval json: $OUTDIR/expert_ablation_eval.json"
