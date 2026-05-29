# 2025 Supply-Demand Day-Ahead Baseline

This baseline predicts Shaanxi day-ahead 96-point electricity spot prices using:

- `data/GS(1).csv`
- `supply_demand_data/features/shaanxi_2025_power_timeline_daily_asof.csv`
- `supply_demand_data/features/supply_demand_policy_calendar_daily.csv`
- `supply_demand_data/features/priority_generation_2025_monthly_features.csv`

## Run

```bash
bash projects/electricity_price_forecasting/scripts/run_supply_demand_day_ahead.sh
```

Outputs are written to:

- `outputs/gs_price_2025_supply_demand_baseline/summary.json`
- `outputs/gs_price_2025_supply_demand_baseline/daily_metrics.csv`
- `outputs/gs_price_2025_supply_demand_baseline/test_predictions.csv`
- `outputs/gs_price_2025_supply_demand_baseline/regime_reports.csv`

## Method

- Uses only `2025` data.
- Builds day-level train/val/test splits after a 7-day history warmup.
- Trains separate `HistGradientBoostingRegressor` models for four price regimes:
  - `floor`
  - `solar`
  - `mid`
  - `evening`
- Avoids same-day price leakage:
  - historical price features come only from previous days
  - target-day exogenous curves come from forecast / plan columns
  - slow supply-demand and policy features come from `supply_demand_data`

## Metrics

Two different daily accuracy definitions are reported.

- `relative_mape_accuracy = max(0, 1 - mean(|pred-true| / max(true, 40)))`
  - This is strict and heavily penalizes errors during floor-price periods.
- `market_range_accuracy = max(0, 1 - daily_mae / (1000 - 40))`
  - This is range-normalized using the observed market floor and cap.

For the current run:

- `mean_market_range_accuracy = 0.8929`
- `share_days_market_range_accuracy_ge_0_8 = 1.0`
- `mean_relative_mape_accuracy = 0.2368`

Interpretation:

- If the acceptance target is "daily accuracy >= 80%" under market-range normalization, this baseline already satisfies it.
- If the acceptance target is "daily pointwise relative accuracy >= 80%", this baseline does not satisfy it and the current project needs a stronger model or a different target definition.
