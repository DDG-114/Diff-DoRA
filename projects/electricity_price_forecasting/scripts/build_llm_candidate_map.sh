#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

.venv/bin/python projects/electricity_price_forecasting/scripts/build_llm_candidate_map.py \
  --source_csv "data/GS(1).csv" \
  --fused_val_csv "outputs/gs_price_2025_supply_demand_fused/val_predictions.csv" \
  --fused_test_csv "outputs/gs_price_2025_supply_demand_fused/test_predictions.csv" \
  --output_csv "outputs/gs_price_2025_llm_candidate_map.csv"
