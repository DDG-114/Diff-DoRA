#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

.venv/bin/python projects/electricity_price_forecasting/scripts/train_point_offset_calibrator.py \
  --candidate_val_csv "outputs/gs_price_2025_supply_demand_baseline/val_predictions.csv" \
  --candidate_test_csv "outputs/gs_price_2025_supply_demand_baseline/test_predictions.csv" \
  --output_dir "outputs/gs_price_2025_point_offset_calibrated" \
  --group_sizes "1,4,8,12,24,48" \
  --shrink_grid "0,0.25,0.5,0.75,1.0" \
  --primary_metric "mean_relative_accuracy" \
  --target_metric 0.8 \
  --clip_offset 120.0
