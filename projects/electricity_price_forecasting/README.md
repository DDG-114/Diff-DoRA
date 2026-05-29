# Electricity Price Forecasting Project

This workspace is for the new GS electricity-price and market-signal forecasting experiments.

## Current Data

Source CSV:

```text
data/GS(1).csv
```

Prepared datasets:

- `gs_price`: price-only target with auxiliary market/system covariates.
- `gs_price_2025`: price-only, 2025 internal train/val/test split.
- `gs_market`: seven-node market-system dataset, 2025 train to 2026 test.
- `gs_market_2025`: seven-node market-system dataset, 2025 internal train/val/test split.

The latest multi-expert 1-day-ahead smoke experiment used `gs_market_2025`:

```text
horizon = 96
history_len = 96
window_stride = 96
frequency = 15 minutes
```

Nodes:

```text
Price
发电总出力预测
竞价空间
统一负荷预测
抽蓄
统一新能源预测
联络线计划
```

Hard-routing experts:

```text
Expert 0: Price, 发电总出力预测, 竞价空间, 统一新能源预测
Expert 1: 统一负荷预测, 抽蓄, 联络线计划
```

## Current Outputs

- `outputs/gs_market_2025_moe_h96_smoke/`: latest multi-expert 1-day-ahead smoke run.
- `outputs/gs_price_2025_h96_*`: earlier price-only 1-day-ahead adapter runs.
- `outputs/gs_2025_quality/` and `outputs/gs_2026_quality/`: data-quality checks.

Top-level `outputs/...` paths are symlinks for compatibility.

## Important Result So Far

The pipeline can train and evaluate 1-day-ahead models, but long-form 96-value generation is still unstable:

- the price-only model often collapses to the price floor;
- the multi-expert CoT run had parse failures on long outputs, especially for Expert 1 nodes.

The next useful direction is to keep the 1-day-ahead target but improve output format and supervision, for example by using direct JSON-only prompts, segmented horizons, or a numeric prediction head.
