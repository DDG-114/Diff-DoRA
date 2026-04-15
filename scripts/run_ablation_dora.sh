#!/usr/bin/env bash
# scripts/run_ablation_dora.sh
# Legacy engineering ablation: LoRA vs DoRA vs Diff-DoRA on single adapters.
# For paper-style w/o-MoE / w/o-CoT / w/o-DoRA, use scripts/run_ablation_paper.sh.

set -e
DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
OUTDIR="outputs/ablation_dora_${DATASET}_h${HORIZON}"

source .venv/bin/activate

python -m src.eval.eval_ablation \
    --dataset    "$DATASET" \
    --horizon    "$HORIZON" \
    --vanilla    "outputs/single_lora_h${HORIZON}/adapter" \
    --rag        "outputs/single_dora_h${HORIZON}/adapter" \
    --diff_dora  "outputs/single_diff_dora_h${HORIZON}/adapter" \
    --output     "${OUTDIR}/ablation_dora.json"

echo "Done. Results: ${OUTDIR}/ablation_dora.json"
