#!/usr/bin/env python3
"""Point/segment offset calibration for 96-slot day-ahead price curves.

This is the conservative curve-level route:
  final(slot) = candidate(slot) + shrink * learned_offset(slot_group)

The offsets are learned only from the validation residuals. The test set is
used only once for final reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_prediction_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    required = {"day", "slot", "prediction", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns: {sorted(required)}")
    out = df[["day", "slot", "prediction", "target"]].copy()
    out["day"] = out["day"].astype(str)
    out["slot"] = pd.to_numeric(out["slot"], errors="raise").astype(int)
    for col in ("prediction", "target"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
    return out.sort_values(["day", "slot"]).reset_index(drop=True)


def _parse_csv_ints(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _parse_csv_floats(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("Expected at least one float.")
    return out


def _validate_group_size(group_size: int) -> None:
    if group_size <= 0 or 96 % group_size != 0:
        raise ValueError(f"slot group size must divide 96, got {group_size}")


def _with_offsets(
    df: pd.DataFrame,
    offsets: pd.Series,
    *,
    group_size: int,
    shrink: float,
    price_floor: float,
    price_cap: float,
) -> pd.DataFrame:
    out = df.copy()
    out["slot_group"] = out["slot"] // group_size
    out["point_offset"] = out["slot_group"].map(offsets).fillna(0.0).astype(float) * float(shrink)
    out["candidate_prediction"] = out["prediction"].astype(float)
    out["prediction"] = np.clip(out["candidate_prediction"] + out["point_offset"], price_floor, price_cap)
    out["residual"] = out["target"] - out["candidate_prediction"]
    out["calibrated_error"] = out["prediction"] - out["target"]
    return out


def _daily_metrics(df: pd.DataFrame, *, pred_col: str = "prediction", price_floor: float, price_cap: float) -> dict:
    rows: list[dict] = []
    for day, group in df.groupby("day", sort=True):
        pred = group[pred_col].to_numpy(dtype=np.float64)
        true = group["target"].to_numpy(dtype=np.float64)
        if len(pred) != 96:
            raise ValueError(f"{day} has {len(pred)} rows, expected 96")
        err = pred - true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        relative_mape = float(np.mean(np.abs(err) / np.maximum(np.abs(true), price_floor)))
        daily_mean_accuracy = max(
            0.0,
            1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), price_floor),
        )
        rows.append(
            {
                "day": day,
                "mae": mae,
                "rmse": rmse,
                "relative_accuracy": max(0.0, 1.0 - relative_mape),
                "daily_mean_accuracy": daily_mean_accuracy,
                "pred_mean": float(pred.mean()),
                "true_mean": float(true.mean()),
                "pred_max": float(pred.max()),
                "true_max": float(true.max()),
                "peak_slot_hit": int(int(np.argmax(pred)) == int(np.argmax(true))),
            }
        )

    daily = pd.DataFrame(rows)
    metrics = {
        "mean_day_mae": float(daily["mae"].mean()),
        "mean_day_rmse": float(daily["rmse"].mean()),
        "mean_relative_accuracy": float(daily["relative_accuracy"].mean()),
        "median_relative_accuracy": float(daily["relative_accuracy"].median()),
        "share_days_relative_accuracy_ge_0_8": float((daily["relative_accuracy"] >= 0.8).mean()),
        "mean_daily_mean_accuracy": float(daily["daily_mean_accuracy"].mean()),
        "median_daily_mean_accuracy": float(daily["daily_mean_accuracy"].median()),
        "share_days_daily_mean_accuracy_ge_0_8": float((daily["daily_mean_accuracy"] >= 0.8).mean()),
        "peak_slot_hit_rate": float(daily["peak_slot_hit"].mean()),
    }
    return {"metrics": metrics, "daily_rows": rows}


def _learn_offsets(val_df: pd.DataFrame, *, group_size: int, clip_offset: float) -> pd.Series:
    _validate_group_size(group_size)
    train = val_df.copy()
    train["slot_group"] = train["slot"] // group_size
    train["residual"] = train["target"] - train["prediction"]
    offsets = train.groupby("slot_group")["residual"].mean()
    return offsets.clip(-float(clip_offset), float(clip_offset))


def _candidate_score(row: dict, *, primary_metric: str) -> tuple:
    met = row["validation_metrics"]
    # Select by validation-only primary metric. Other terms are tie-breakers.
    return (
        float(met[primary_metric]),
        -float(met["mean_day_mae"]),
        -int(row["group_size"]),
        -float(row["shrink"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_val_csv", default="outputs/gs_price_2025_supply_demand_baseline/val_predictions.csv")
    parser.add_argument("--candidate_test_csv", default="outputs/gs_price_2025_supply_demand_baseline/test_predictions.csv")
    parser.add_argument("--output_dir", default="outputs/gs_price_2025_point_offset_calibrated")
    parser.add_argument("--group_sizes", default="1,4,8,12,24,48", help="Slot group sizes to search; 1 means 96 point-level offsets.")
    parser.add_argument("--shrink_grid", default="0,0.25,0.5,0.75,1.0", help="Shrink values for learned offsets.")
    parser.add_argument(
        "--primary_metric",
        choices=["mean_relative_accuracy", "mean_daily_mean_accuracy"],
        default="mean_relative_accuracy",
    )
    parser.add_argument("--target_metric", type=float, default=0.8)
    parser.add_argument("--target_daily_accuracy", type=float, default=None, help="Deprecated alias for daily-accuracy runs.")
    parser.add_argument("--clip_offset", type=float, default=120.0)
    parser.add_argument("--price_floor", type=float, default=40.0)
    parser.add_argument("--price_cap", type=float, default=1000.0)
    args = parser.parse_args()
    if args.target_daily_accuracy is not None:
        args.primary_metric = "mean_daily_mean_accuracy"
        args.target_metric = float(args.target_daily_accuracy)

    val_df = _read_prediction_csv(args.candidate_val_csv)
    test_df = _read_prediction_csv(args.candidate_test_csv)
    group_sizes = _parse_csv_ints(args.group_sizes)
    shrink_grid = _parse_csv_floats(args.shrink_grid)
    for group_size in group_sizes:
        _validate_group_size(group_size)

    candidate_val = val_df.rename(columns={"prediction": "candidate_prediction"}).copy()
    candidate_val["prediction"] = candidate_val["candidate_prediction"]
    candidate_test = test_df.rename(columns={"prediction": "candidate_prediction"}).copy()
    candidate_test["prediction"] = candidate_test["candidate_prediction"]
    candidate_val_metrics = _daily_metrics(candidate_val, price_floor=args.price_floor, price_cap=args.price_cap)
    candidate_test_metrics = _daily_metrics(candidate_test, price_floor=args.price_floor, price_cap=args.price_cap)

    search_rows = []
    for group_size in group_sizes:
        offsets = _learn_offsets(val_df, group_size=group_size, clip_offset=args.clip_offset)
        for shrink in shrink_grid:
            calibrated_val = _with_offsets(
                val_df,
                offsets,
                group_size=group_size,
                shrink=shrink,
                price_floor=args.price_floor,
                price_cap=args.price_cap,
            )
            val_metrics = _daily_metrics(calibrated_val, price_floor=args.price_floor, price_cap=args.price_cap)["metrics"]
            search_rows.append(
                {
                    "group_size": int(group_size),
                    "offset_count": int(96 // group_size),
                    "shrink": float(shrink),
                    "validation_metrics": val_metrics,
                }
            )

    selected = max(search_rows, key=lambda row: _candidate_score(row, primary_metric=args.primary_metric))
    selected_group_size = int(selected["group_size"])
    selected_shrink = float(selected["shrink"])
    selected_offsets = _learn_offsets(val_df, group_size=selected_group_size, clip_offset=args.clip_offset)
    calibrated_test = _with_offsets(
        test_df,
        selected_offsets,
        group_size=selected_group_size,
        shrink=selected_shrink,
        price_floor=args.price_floor,
        price_cap=args.price_cap,
    )
    calibrated_val = _with_offsets(
        val_df,
        selected_offsets,
        group_size=selected_group_size,
        shrink=selected_shrink,
        price_floor=args.price_floor,
        price_cap=args.price_cap,
    )
    val_eval = _daily_metrics(calibrated_val, price_floor=args.price_floor, price_cap=args.price_cap)
    test_eval = _daily_metrics(calibrated_test, price_floor=args.price_floor, price_cap=args.price_cap)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_val.to_csv(output_dir / "val_predictions.csv", index=False)
    calibrated_test.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(test_eval["daily_rows"]).to_csv(output_dir / "daily_metrics.csv", index=False)
    pd.DataFrame(search_rows).to_json(output_dir / "validation_search.json", orient="records", indent=2, force_ascii=False)
    pd.DataFrame(
        {
            "slot_group": selected_offsets.index.astype(int),
            "slot_start": selected_offsets.index.astype(int) * selected_group_size,
            "slot_end_exclusive": (selected_offsets.index.astype(int) + 1) * selected_group_size,
            "raw_offset": selected_offsets.to_numpy(dtype=float),
            "applied_offset": selected_offsets.to_numpy(dtype=float) * selected_shrink,
        }
    ).to_csv(output_dir / "point_offsets.csv", index=False)

    summary = {
        "route": "supply_demand_candidate_plus_validated_point_offset",
        "candidate_val_csv": args.candidate_val_csv,
        "candidate_test_csv": args.candidate_test_csv,
        "selection_policy": {
            "description": "Validation-only search; maximize the configured primary metric. The target is used for audit reporting only.",
            "primary_metric": args.primary_metric,
            "target_metric": float(args.target_metric),
            "group_sizes": group_sizes,
            "shrink_grid": shrink_grid,
            "clip_offset": float(args.clip_offset),
        },
        "selected": {
            "group_size": selected_group_size,
            "offset_count": int(96 // selected_group_size),
            "shrink": selected_shrink,
            "validation_metrics": val_eval["metrics"],
        },
        "candidate_metrics": {
            "validation": candidate_val_metrics["metrics"],
            "test": candidate_test_metrics["metrics"],
        },
        "metrics": test_eval["metrics"],
        "objective_audit": {
            "metric_name": args.primary_metric,
            "threshold": float(args.target_metric),
            "value": float(test_eval["metrics"][args.primary_metric]),
            "passed": bool(test_eval["metrics"][args.primary_metric] >= args.target_metric),
        },
        "outputs": {
            "val_predictions": str(output_dir / "val_predictions.csv"),
            "test_predictions": str(output_dir / "test_predictions.csv"),
            "daily_metrics": str(output_dir / "daily_metrics.csv"),
            "point_offsets": str(output_dir / "point_offsets.csv"),
            "validation_search": str(output_dir / "validation_search.json"),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
