#!/usr/bin/env bash
# scripts/run_ablation_paper.sh
# Paper-oriented ablation runner: full / w-o-MoE / w-o-CoT / w-o-DoRA / base model.

set -e
DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
OUTDIR="outputs/paper_ablation_${DATASET}_h${HORIZON}"

source .venv/bin/activate

python -m src.eval.eval_paper_ablation \
    --dataset "$DATASET" \
    --horizon "$HORIZON" \
    --use_rag \
    --full_expert_0_dir "outputs/${DATASET}_moe_diffdora_h${HORIZON}/expert_0/adapter" \
    --full_expert_1_dir "outputs/${DATASET}_moe_diffdora_h${HORIZON}/expert_1/adapter" \
    --wo_moe_dir "outputs/${DATASET}_single_rag_cot_h${HORIZON}/adapter" \
    --wo_cot_expert_0_dir "outputs/${DATASET}_moe_nocot_h${HORIZON}/expert_0/adapter" \
    --wo_cot_expert_1_dir "outputs/${DATASET}_moe_nocot_h${HORIZON}/expert_1/adapter" \
    --wo_dora_expert_0_dir "outputs/${DATASET}_moe_dora_h${HORIZON}/expert_0/adapter" \
    --wo_dora_expert_1_dir "outputs/${DATASET}_moe_dora_h${HORIZON}/expert_1/adapter" \
    --output "${OUTDIR}/paper_ablation.json"

echo "Done. Results: ${OUTDIR}/paper_ablation.json"
