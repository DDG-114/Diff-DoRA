#!/usr/bin/env python3
"""Fuse tree and sequence day-ahead predictions using validation-selected weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _daily_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    acc = []
    mae = []
    mean_acc = []
    rows = []
    for day, group in df.groupby("day", sort=True):
        pred = group[pred_col].to_numpy(dtype=np.float32)
        true = group["target"].to_numpy(dtype=np.float32)
        day_mae = float(np.mean(np.abs(pred - true)))
        relative_mape = float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 40.0)))
        day_acc = max(0.0, 1.0 - relative_mape)
        day_mean_acc = max(0.0, 1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), 40.0))
        rows.append(
            {
                "day": day,
                "mae": day_mae,
                "relative_mape_accuracy": day_acc,
                "daily_mean_accuracy": day_mean_acc,
            }
        )
        mae.append(day_mae)
        acc.append(day_acc)
        mean_acc.append(day_mean_acc)
    return {
        "daily_rows": rows,
        "metrics": {
            "mean_day_mae": float(np.mean(mae)),
            "mean_relative_mape_accuracy": float(np.mean(acc)),
            "median_relative_mape_accuracy": float(np.median(acc)),
            "share_days_relative_mape_accuracy_ge_0_8": float((np.asarray(acc) >= 0.8).mean()),
            "mean_daily_mean_accuracy": float(np.mean(mean_acc)),
        },
    }


def _load_pair(base_path: Path, seq_path: Path) -> pd.DataFrame:
    base = pd.read_csv(base_path)
    seq = pd.read_csv(seq_path)
    base = base.groupby(["day", "slot"], as_index=False).agg({"prediction": "mean", "target": "mean"})
    seq = seq.groupby(["day", "slot"], as_index=False).agg({"prediction": "mean", "target": "mean"})
    merged = base.merge(seq, on=["day", "slot"], suffixes=("_base", "_seq"))
    target = merged["target_base"].to_numpy(dtype=np.float32)
    if not np.allclose(target, merged["target_seq"].to_numpy(dtype=np.float32)):
        raise ValueError("Base and sequence targets do not align.")
    merged["target"] = target
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_val", default="outputs/gs_price_2025_supply_demand_baseline/val_predictions.csv")
    parser.add_argument("--base_test", default="outputs/gs_price_2025_supply_demand_baseline/test_predictions.csv")
    parser.add_argument("--seq_val", default="outputs/gs_price_2025_supply_demand_seq/val_predictions.csv")
    parser.add_argument("--seq_test", default="outputs/gs_price_2025_supply_demand_seq/test_predictions.csv")
    parser.add_argument("--output_dir", default="outputs/gs_price_2025_supply_demand_fused")
    args = parser.parse_args()

    val = _load_pair(Path(args.base_val), Path(args.seq_val))
    test = _load_pair(Path(args.base_test), Path(args.seq_test))

    best_weight = 1.0
    best_score = float("-inf")
    val_candidates: list[dict] = []
    for weight in np.linspace(0.0, 1.0, num=11):
        val["prediction"] = weight * val["prediction_base"] + (1.0 - weight) * val["prediction_seq"]
        result = _daily_metrics(val, "prediction")
        score = result["metrics"]["mean_relative_mape_accuracy"]
        val_candidates.append({"base_weight": float(weight), **result["metrics"]})
        if score > best_score:
            best_score = score
            best_weight = float(weight)

    val["prediction"] = best_weight * val["prediction_base"] + (1.0 - best_weight) * val["prediction_seq"]
    test["prediction"] = best_weight * test["prediction_base"] + (1.0 - best_weight) * test["prediction_seq"]
    fused_eval = _daily_metrics(test, "prediction")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(val_candidates).to_csv(output_dir / "val_weight_search.csv", index=False)
    val[["day", "slot", "prediction", "target"]].to_csv(output_dir / "val_predictions.csv", index=False)
    pd.DataFrame(fused_eval["daily_rows"]).to_csv(output_dir / "daily_metrics.csv", index=False)
    test[["day", "slot", "prediction", "target"]].to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "dataset": "gs_price_2025_supply_demand_fused",
        "base_weight": best_weight,
        "seq_weight": 1.0 - best_weight,
        "validation_best_mean_relative_mape_accuracy": best_score,
        "metrics": fused_eval["metrics"],
        "objective_audit": {
            "metric_name": "mean_daily_mean_accuracy",
            "threshold": 0.8,
            "value": float(fused_eval["metrics"]["mean_daily_mean_accuracy"]),
            "passed": bool(fused_eval["metrics"]["mean_daily_mean_accuracy"] >= 0.8),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
