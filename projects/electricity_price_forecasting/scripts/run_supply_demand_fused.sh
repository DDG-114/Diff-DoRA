#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

.venv/bin/python projects/electricity_price_forecasting/scripts/fuse_supply_demand_predictions.py \
  --base_val "outputs/gs_price_2025_supply_demand_baseline/val_predictions.csv" \
  --base_test "outputs/gs_price_2025_supply_demand_baseline/test_predictions.csv" \
  --seq_val "outputs/gs_price_2025_supply_demand_seq/val_predictions.csv" \
  --seq_test "outputs/gs_price_2025_supply_demand_seq/test_predictions.csv" \
  --output_dir "outputs/gs_price_2025_supply_demand_fused"
