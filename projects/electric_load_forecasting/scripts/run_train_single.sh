#!/usr/bin/env bash
# scripts/run_train_single.sh
# Paper-aligned launcher for single-expert LoRA / DoRA training.

set -e
DATASET=${1:-st_evcdp}
HORIZON=${2:-6}
USE_DORA=${3:-""}

source .venv/bin/activate

DORA_FLAG=""
OUTNAME="single_lora"
if [ "$USE_DORA" = "dora" ]; then
    DORA_FLAG="--use_dora"
    OUTNAME="single_dora"
fi

python -m src.train.train_single \
    --dataset    "$DATASET" \
    --horizon    "$HORIZON" \
    --output_dir "outputs/${OUTNAME}_h${HORIZON}" \
    $DORA_FLAG

echo "Training complete."
