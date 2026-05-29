# Advisor Brief: LLM Electricity Spot Price Forecasting

## Objective

Use LLM-based methods to forecast Shaanxi day-ahead electricity spot prices on `2025` data with a `96`-step horizon (`15 min x 96 = 1 day`).

## Core conclusion

Directly asking an LLM to generate all `96` future prices is not realistic in the current setup.

The repo now supports a more practical LLM route:

- short-horizon prediction (`h16`)
- rolling composition to `96` steps
- retrieval augmentation
- candidate skeleton curve
- LLM residual refinement

## Why the route changed

Observed failure modes from current repo:

- direct `h96` generation drifts badly in later steps
- naive `h16 rolling` collapses to near-floor constants

This matches recent literature:

- `Time-LLM`
- `Large Language Models Are Zero-Shot Time Series Forecasters`
- `Forecasting Time Series with LLMs via Patch-Based Prompting and Decomposition`
- `FLAIRR-TS`
- `Day-Ahead Electricity Price Forecasting for Volatile Markets Using Foundation Models with Regularization Strategy`
- `Regression Models Meet Foundation Models`

## Implemented technical route

1. Build a candidate day-level skeleton
   - from the strongest non-LLM baseline
   - stored in:
     - [outputs/gs_price_2025_llm_candidate_map.csv](/root/Diff-DoRA/outputs/gs_price_2025_llm_candidate_map.csv:1)

2. Train the LLM to predict residual corrections, not absolute prices
   - candidate and labels now share the same normalized scale
   - residual targets are clipped during early training

3. Roll `h16` predictions forward to reconstruct `96` points

## Code status

Implemented files:

- [projects/electricity_price_forecasting/LLM_EP_PLAN.md](/root/Diff-DoRA/projects/electricity_price_forecasting/LLM_EP_PLAN.md:1)
- [projects/electricity_price_forecasting/scripts/build_llm_candidate_map.py](/root/Diff-DoRA/projects/electricity_price_forecasting/scripts/build_llm_candidate_map.py:1)
- [projects/electricity_price_forecasting/scripts/run_llm_h16_candidate_rolling_day.sh](/root/Diff-DoRA/projects/electricity_price_forecasting/scripts/run_llm_h16_candidate_rolling_day.sh:1)
- [src/utils/price_candidate.py](/root/Diff-DoRA/src/utils/price_candidate.py:1)
- [src/train/train_single.py](/root/Diff-DoRA/src/train/train_single.py:1)
- [src/train/train_shared_adapter.py](/root/Diff-DoRA/src/train/train_shared_adapter.py:1)
- [src/prompts/prompt_vanilla.py](/root/Diff-DoRA/src/prompts/prompt_vanilla.py:1)
- [projects/electricity_price_forecasting/scripts/eval_gs_price_rolling_day.py](/root/Diff-DoRA/projects/electricity_price_forecasting/scripts/eval_gs_price_rolling_day.py:1)

## Experimental status

### Candidate baseline on the same 10 days

- `mean_daily_mean_accuracy ≈ 0.7488`

### LLM residual refinement on 10 rolling days

File:

- [outputs/gs_price_2025_llm_h16_candidate_residual_normfix_10day_rolling_day96.json](/root/Diff-DoRA/outputs/gs_price_2025_llm_h16_candidate_residual_normfix_10day_rolling_day96.json:1)

Key metrics:

- `parse_success_rate = 1.0`
- `mean_day_mae ≈ 94.80`
- `mean_day_rmse ≈ 171.14`
- `mean_daily_mean_accuracy ≈ 0.6897`

### Interpretation

- The LLM route is now technically stable.
- It no longer collapses to constants.
- It no longer explodes numerically after the normalization fix.
- However, it still underperforms the candidate baseline on average.

## Current research conclusion

At this stage, the LLM is better viewed as a:

- local refiner
- retrieval-aware corrector
- candidate-editing module

rather than a full day-curve generator.

## Recommended next step

The next step should not be larger direct full-curve rewriting.

The best next research move is:

- `selective refinement`
  - only let the LLM modify parts of the candidate curve that are likely unreliable
  - preserve the candidate elsewhere

## Practical summary

Current status is:

- feasible LLM route: `yes`
- stable rolling-day generation: `yes`
- already reaches 80% on current evaluated LLM setting: `no`
- most promising next iteration: `selective local refinement`
