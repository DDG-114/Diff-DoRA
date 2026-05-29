#!/usr/bin/env bash
# Strict evaluation launcher:
# - full test split
# - all routed nodes
# - no quick max_eval/node subsampling
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
OUTDIR=${3:-"outputs/strict_repro_${DATASET}_h${HORIZON}"}
CACHE_PATH="data/retrieval_cache/${DATASET}_h${HORIZON}.pkl"

source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

mkdir -p "$OUTDIR"

python -m src.eval.eval_paper_ablation \
  --dataset "$DATASET" \
  --horizon "$HORIZON" \
  --use_rag \
  --retrieval_cache "$CACHE_PATH" \
  --max_eval 0 \
  --sampling head \
  --seed 42 \
  --node_sampling all \
  --max_new_tokens 512 \
  --infer_batch_size 12 \
  --skip_base_model \
  --full_expert_0_dir "$OUTDIR/full/expert_0/adapter" \
  --full_expert_1_dir "$OUTDIR/full/expert_1/adapter" \
  --wo_diffdora_expert_0_dir "$OUTDIR/wo_diffdora/expert_0/adapter" \
  --wo_diffdora_expert_1_dir "$OUTDIR/wo_diffdora/expert_1/adapter" \
  --output "$OUTDIR/expert_ablation_full_test.json"

echo "Strict full-test evaluation complete."
echo "Eval json: $OUTDIR/expert_ablation_full_test.json"
