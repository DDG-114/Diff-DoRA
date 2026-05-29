#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

.venv/bin/python projects/electricity_price_forecasting/scripts/train_supply_demand_day_ahead_seq.py \
  --source_csv "data/GS(1).csv" \
  --timeline_csv "supply_demand_data/features/shaanxi_2025_power_timeline_daily_asof.csv" \
  --policy_csv "supply_demand_data/features/supply_demand_policy_calendar_daily.csv" \
  --output_dir "outputs/gs_price_2025_supply_demand_seq"
