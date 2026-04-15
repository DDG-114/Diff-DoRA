#!/usr/bin/env bash
# scripts/run_ablation_cot.sh
# Legacy engineering ablation: vanilla vs RAG vs RAG+CoT.
# For paper-style ablations, use scripts/run_ablation_paper.sh.

set -e
DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
OUTDIR="outputs/ablation_cot_${DATASET}_h${HORIZON}"

source .venv/bin/activate

python -m src.eval.eval_ablation \
    --dataset    "$DATASET" \
    --horizon    "$HORIZON" \
    --vanilla    "outputs/single_lora_h${HORIZON}/adapter" \
    --rag        "outputs/single_rag_h${HORIZON}/adapter" \
    --rag_cot    "outputs/single_rag_cot_h${HORIZON}/adapter" \
    --output     "${OUTDIR}/ablation_cot.json"

echo "Done. Results: ${OUTDIR}/ablation_cot.json"
