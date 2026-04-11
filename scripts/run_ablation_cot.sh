#!/usr/bin/env bash
# scripts/run_ablation_cot.sh
# Ablation: vanilla vs RAG vs RAG+CoT
# Requires adapters to have been trained first.

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
