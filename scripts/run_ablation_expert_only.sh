#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
OUTDIR="outputs/ablation_expert_${DATASET}_h${HORIZON}_fixed"
CACHE_PATH="data/retrieval_cache/${DATASET}_h${HORIZON}.pkl"

source .venv/bin/activate

if [ ! -f "$CACHE_PATH" ]; then
  echo "Retrieval cache not found: $CACHE_PATH" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

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

python -m src.train.train_experts   "${COMMON_TRAIN_ARGS[@]}"   --output_dir "$OUTDIR/full"   --use_dora   --use_diff_dora   --prompt_style cot

python -m src.train.train_experts   "${COMMON_TRAIN_ARGS[@]}"   --output_dir "$OUTDIR/wo_cot"   --use_dora   --use_diff_dora   --prompt_style direct_physical

python -m src.train.train_experts   "${COMMON_TRAIN_ARGS[@]}"   --output_dir "$OUTDIR/wo_dora"   --prompt_style cot

python -m src.eval.eval_paper_ablation   --dataset "$DATASET"   --horizon "$HORIZON"   --use_rag   --retrieval_cache "$CACHE_PATH"   --max_eval 6   --sampling random   --seed 42   --node_sampling balanced_random   --max_nodes_per_domain 12   --max_new_tokens 512   --infer_batch_size 12   --skip_base_model   --full_expert_0_dir "$OUTDIR/full/expert_0/adapter"   --full_expert_1_dir "$OUTDIR/full/expert_1/adapter"   --wo_cot_expert_0_dir "$OUTDIR/wo_cot/expert_0/adapter"   --wo_cot_expert_1_dir "$OUTDIR/wo_cot/expert_1/adapter"   --wo_dora_expert_0_dir "$OUTDIR/wo_dora/expert_0/adapter"   --wo_dora_expert_1_dir "$OUTDIR/wo_dora/expert_1/adapter"   --output "$OUTDIR/expert_ablation.json"

python -m src.analysis.visualize_expert_ablation   --ablation-json "$OUTDIR/expert_ablation.json"   --output-dir "$OUTDIR/figures"

echo "Ablation run complete: $OUTDIR"
