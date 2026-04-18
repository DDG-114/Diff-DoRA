#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}

OUTDIR="outputs/ablation_inference_track_${DATASET}_h${HORIZON}_shapeaware"
FIXED_DIR="outputs/ablation_inference_track_${DATASET}_h${HORIZON}_fixed"
CACHE_PATH="data/retrieval_cache/${DATASET}_h${HORIZON}_shapeaware.pkl"

source .venv/bin/activate

mkdir -p "$OUTDIR"

python -m src.retrieval.build_cache \
  --datasets "$DATASET" \
  --horizons "$HORIZON" \
  --output_path "$CACHE_PATH"

COMMON_TRAIN_ARGS=(
  --dataset "$DATASET"
  --horizon "$HORIZON"
  --epochs 2
  --batch_size 64
  --lr 2e-4
  --max_length 2560
  --lora_rank 32
  --lora_alpha 32
  --history_len 12
  --neighbor_k 7
  --use_rag
  --retrieval_cache "$CACHE_PATH"
  --max_samples_per_expert 1000
  --eval_max_samples 0
  --eval_nodes_per_expert 0
)

python -m src.train.train_experts \
  "${COMMON_TRAIN_ARGS[@]}" \
  --output_dir "$OUTDIR/full" \
  --use_dora \
  --use_diff_dora \
  --prompt_style cot

python -m src.train.train_experts \
  "${COMMON_TRAIN_ARGS[@]}" \
  --output_dir "$OUTDIR/wo_diffdora" \
  --use_dora \
  --prompt_style cot

run_eval() {
  local run_name=$1
  local max_eval=$2
  local seed=$3

  mkdir -p "$OUTDIR/$run_name"
  python -m src.eval.eval_paper_ablation \
    --dataset "$DATASET" \
    --horizon "$HORIZON" \
    --use_rag \
    --retrieval_cache "$CACHE_PATH" \
    --max_eval "$max_eval" \
    --sampling random \
    --seed "$seed" \
    --node_sampling balanced_random \
    --max_nodes_per_domain 12 \
    --max_new_tokens 512 \
    --infer_batch_size 12 \
    --skip_base_model \
    --full_expert_0_dir "$OUTDIR/full/expert_0/adapter" \
    --full_expert_1_dir "$OUTDIR/full/expert_1/adapter" \
    --wo_diffdora_expert_0_dir "$OUTDIR/wo_diffdora/expert_0/adapter" \
    --wo_diffdora_expert_1_dir "$OUTDIR/wo_diffdora/expert_1/adapter" \
    --wo_cot_expert_0_dir "$OUTDIR/full/expert_0/adapter" \
    --wo_cot_expert_1_dir "$OUTDIR/full/expert_1/adapter" \
    --wo_rag_expert_0_dir "$OUTDIR/full/expert_0/adapter" \
    --wo_rag_expert_1_dir "$OUTDIR/full/expert_1/adapter" \
    --output "$OUTDIR/$run_name/expert_ablation.json"
}

run_eval quick_seed42 6 42
run_eval official_seed42 60 42
run_eval official_seed43 60 43
run_eval official_seed44 60 44

if [ -f "$FIXED_DIR/quick_seed42/expert_ablation.json" ]; then
  python -m src.analysis.compare_inference_track \
    --baseline_root "$FIXED_DIR" \
    --shapeaware_root "$OUTDIR" \
    --output_json "$OUTDIR/comparison_summary.json"
else
  echo "[WARN] Fixed baseline quick/official outputs not found under: $FIXED_DIR" >&2
  echo "[WARN] Skipping old-vs-new comparison summary." >&2
fi

echo "Inference-track ablation complete: $OUTDIR"
