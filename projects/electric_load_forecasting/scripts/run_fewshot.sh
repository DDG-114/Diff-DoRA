#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
OUTDIR="outputs/fewshot_${DATASET}_h${HORIZON}"

source .venv/bin/activate

if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS:-0}" -le 0 ] 2>/dev/null; then
  export OMP_NUM_THREADS=1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m src.eval.eval_fewshot \
  --dataset "$DATASET" \
  --horizon "$HORIZON" \
  --output_dir "$OUTDIR" \
  --max_train_items 4000 \
  --use_dora \
  --use_diff_dora \
  --use_rag \
  --prompt_style cot

echo "Few-shot run complete: $OUTDIR"
