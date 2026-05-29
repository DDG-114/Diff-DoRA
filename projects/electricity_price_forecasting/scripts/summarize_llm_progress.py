#!/usr/bin/env python3
"""Summarize current LLM electricity-price forecasting progress."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _daily_mean_accuracy_from_records(records: list[dict]) -> tuple[float, float, float]:
    scores = []
    for row in records:
        if row.get("parsed_prediction") is None:
            continue
        pred = np.asarray(row["parsed_prediction"], dtype=np.float32) * 1000.0
        true = np.asarray(row["target"], dtype=np.float32) * 1000.0
        score = max(0.0, 1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), 40.0))
        scores.append(score)
    arr = np.asarray(scores, dtype=np.float32)
    return float(arr.mean()), float(np.median(arr)), float((arr >= 0.8).mean())


def main() -> None:
    rows = []

    base = _load_json("outputs/gs_price_2025_llm_h16_candidate_smoke_rolling_day96.json")
    rows.append(
        {
            "run": "llm_h16_candidate_absolute_smoke",
            "days": int(base["parsed_days"]),
            "parse_success_rate": float(base["parse_success_rate"]),
            "mean_day_mae": float(base["metrics"]["mean_day_mae"]),
            "mean_day_rmse": float(base["metrics"]["mean_day_rmse"]),
            "mean_prediction_range": float(base["metrics"]["mean_prediction_range"]),
            "mean_daily_mean_accuracy": float("nan"),
            "median_daily_mean_accuracy": float("nan"),
            "share_days_daily_mean_accuracy_ge_0_8": float("nan"),
        }
    )

    residual_3day = _load_json("outputs/gs_price_2025_llm_h16_candidate_residual_normfix_3day_rolling_day96.json")
    mean_acc, median_acc, share_acc = _daily_mean_accuracy_from_records(residual_3day["records"])
    rows.append(
        {
            "run": "llm_h16_candidate_residual_normfix_3day",
            "days": int(residual_3day["parsed_days"]),
            "parse_success_rate": float(residual_3day["parse_success_rate"]),
            "mean_day_mae": float(residual_3day["metrics"]["mean_day_mae"]),
            "mean_day_rmse": float(residual_3day["metrics"]["mean_day_rmse"]),
            "mean_prediction_range": float(residual_3day["metrics"]["mean_prediction_range"]),
            "mean_daily_mean_accuracy": mean_acc,
            "median_daily_mean_accuracy": median_acc,
            "share_days_daily_mean_accuracy_ge_0_8": share_acc,
        }
    )

    residual_10day = _load_json("outputs/gs_price_2025_llm_h16_candidate_residual_normfix_10day_rolling_day96.json")
    mean_acc, median_acc, share_acc = _daily_mean_accuracy_from_records(residual_10day["records"])
    rows.append(
        {
            "run": "llm_h16_candidate_residual_normfix_10day",
            "days": int(residual_10day["parsed_days"]),
            "parse_success_rate": float(residual_10day["parse_success_rate"]),
            "mean_day_mae": float(residual_10day["metrics"]["mean_day_mae"]),
            "mean_day_rmse": float(residual_10day["metrics"]["mean_day_rmse"]),
            "mean_prediction_range": float(residual_10day["metrics"]["mean_prediction_range"]),
            "mean_daily_mean_accuracy": mean_acc,
            "median_daily_mean_accuracy": median_acc,
            "share_days_daily_mean_accuracy_ge_0_8": share_acc,
        }
    )

    daily_offset = _load_json("outputs/gs_price_2025_llm_daily_offset_smoke_eval_fixed_v2.json")
    rows.append(
        {
            "run": "llm_daily_offset_10day",
            "days": int(daily_offset["days"]),
            "parse_success_rate": float(daily_offset["parse_success_rate"]),
            "mean_day_mae": float(daily_offset["metrics"]["mean_day_mae"]),
            "mean_day_rmse": float(daily_offset["metrics"]["mean_day_rmse"]),
            "mean_prediction_range": float("nan"),
            "mean_daily_mean_accuracy": float(daily_offset["metrics"]["mean_daily_mean_accuracy"]),
            "median_daily_mean_accuracy": float("nan"),
            "share_days_daily_mean_accuracy_ge_0_8": float(daily_offset["metrics"]["share_days_daily_mean_accuracy_ge_0_8"]),
        }
    )

    out_dir = Path("outputs/gs_price_2025_llm_progress")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_table.csv", index=False)

    payload = {
        "rows": rows,
        "best_run_by_mean_day_mae": min(rows, key=lambda row: row["mean_day_mae"]),
        "best_run_by_daily_mean_accuracy": max(
            [row for row in rows if not np.isnan(row["mean_daily_mean_accuracy"])],
            key=lambda row: row["mean_daily_mean_accuracy"],
        ),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
