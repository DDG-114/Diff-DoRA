#!/usr/bin/env python3
"""Compare candidate baseline and LLM rolling forecasts on aligned days."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_truth(source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").interpolate().ffill().bfill()
    df = df.loc[df["Date"].dt.year == 2025, ["Date", "Price"]].copy()
    df["day"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["slot"] = df["Date"].dt.hour * 4 + df["Date"].dt.minute // 15
    df["truth"] = df["Price"] / 1000.0
    return df[["day", "slot", "truth"]]


def _curve_metrics(curves: list[dict], pred_key: str) -> pd.DataFrame:
    rows = []
    for item in curves:
        pred = np.asarray(item[pred_key], dtype=np.float32) * 1000.0
        true = np.asarray(item["truth"], dtype=np.float32) * 1000.0
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mean_acc = max(0.0, 1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), 40.0))
        rel_acc = max(0.0, 1.0 - float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 40.0))))
        rows.append(
            {
                "day": item["window_start"],
                "mae": mae,
                "rmse": rmse,
                "daily_mean_accuracy": mean_acc,
                "relative_accuracy": rel_acc,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--llm_json", default="outputs/gs_price_2025_llm_h16_candidate_residual_normfix_10day_rolling_day96.json")
    parser.add_argument("--output_dir", default="outputs/gs_price_2025_candidate_vs_llm")
    args = parser.parse_args()

    truth = _load_truth(Path(args.source_csv))
    candidate = pd.read_csv(args.candidate_csv)
    candidate.columns = [str(col).lstrip("\ufeff").strip() for col in candidate.columns]
    candidate_map = {
        (str(row.day), int(row.slot)): float(row.prediction)
        for row in candidate.itertuples(index=False)
    }
    llm_payload = json.loads(Path(args.llm_json).read_text())
    curves = []
    for record in llm_payload["records"]:
        start_ts = pd.Timestamp(record["day_start_timestamp"])
        true = np.asarray(record["target"], dtype=np.float32)
        llm_pred = np.asarray(record["parsed_prediction"], dtype=np.float32)
        candidate_curve = []
        for idx in range(len(true)):
            ts = start_ts + pd.Timedelta(minutes=15 * idx)
            day = str(ts.normalize().date())
            slot = int(ts.hour * 4 + ts.minute // 15)
            candidate_curve.append(float(candidate_map[(day, slot)]))
        curves.append(
            {
                "window_start": str(start_ts),
                "truth": true,
                "candidate_prediction": np.asarray(candidate_curve, dtype=np.float32),
                "llm_prediction": llm_pred,
            }
        )

    cand_daily = _curve_metrics(curves, "candidate_prediction")
    llm_daily = _curve_metrics(curves, "llm_prediction")
    compare = cand_daily.merge(llm_daily, on="day", suffixes=("_candidate", "_llm"))

    summary = {
        "candidate": {
            "mean_day_mae": float(compare["mae_candidate"].mean()),
            "mean_day_rmse": float(compare["rmse_candidate"].mean()),
            "mean_daily_mean_accuracy": float(compare["daily_mean_accuracy_candidate"].mean()),
            "share_days_daily_mean_accuracy_ge_0_8": float((compare["daily_mean_accuracy_candidate"] >= 0.8).mean()),
            "mean_relative_accuracy": float(compare["relative_accuracy_candidate"].mean()),
        },
        "llm": {
            "mean_day_mae": float(compare["mae_llm"].mean()),
            "mean_day_rmse": float(compare["rmse_llm"].mean()),
            "mean_daily_mean_accuracy": float(compare["daily_mean_accuracy_llm"].mean()),
            "share_days_daily_mean_accuracy_ge_0_8": float((compare["daily_mean_accuracy_llm"] >= 0.8).mean()),
            "mean_relative_accuracy": float(compare["relative_accuracy_llm"].mean()),
        },
        "days": int(len(compare)),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compare.to_csv(out_dir / "daily_comparison.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
