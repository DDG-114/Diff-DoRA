#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

.venv/bin/python projects/electricity_price_forecasting/scripts/prepare_gs_price.py \
  --source_csv "data/GS(1).csv" \
  --output_dir data/raw/gs_price \
  --processed_path data/processed/gs_price.pkl \
  --split_mode 2025_to_2026

.venv/bin/python projects/electricity_price_forecasting/scripts/prepare_gs_price.py \
  --source_csv "data/GS(1).csv" \
  --output_dir data/raw/gs_price_2025 \
  --processed_path data/processed/gs_price_2025.pkl \
  --split_mode 2025_internal

.venv/bin/python projects/electricity_price_forecasting/scripts/prepare_gs_market.py \
  --source_csv "data/GS(1).csv" \
  --output_dir data/raw/gs_market \
  --processed_path data/processed/gs_market.pkl \
  --split_mode 2025_to_2026

.venv/bin/python projects/electricity_price_forecasting/scripts/prepare_gs_market.py \
  --source_csv "data/GS(1).csv" \
  --output_dir data/raw/gs_market_2025 \
  --processed_path data/processed/gs_market_2025.pkl \
  --split_mode 2025_internal
