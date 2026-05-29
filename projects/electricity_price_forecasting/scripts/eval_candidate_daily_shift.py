#!/usr/bin/env python3
"""Evaluate an oracle-style daily-shift correction upper bound for candidate forecasts.

This is not a deployable predictor by itself; it quantifies how much of the
remaining error is explainable by a day-level mean offset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--start_day", default="2025-11-26")
    parser.add_argument("--end_day", default="2025-12-31")
    parser.add_argument("--output", default="outputs/gs_price_2025_candidate_daily_shift_eval.json")
    args = parser.parse_args()

    cand = pd.read_csv(args.candidate_csv)
    cand.columns = [str(col).lstrip("\ufeff").strip() for col in cand.columns]
    cand_map = {(str(row.day), int(row.slot)): float(row.prediction) for row in cand.itertuples(index=False)}

    raw = pd.read_csv(args.source_csv)
    raw.columns = [str(col).lstrip("\ufeff").strip() for col in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", format="mixed")
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce").interpolate().ffill().bfill() / 1000.0
    raw = raw.loc[raw["Date"].dt.year == 2025].copy()
    raw["day"] = raw["Date"].dt.strftime("%Y-%m-%d")
    raw["slot"] = raw["Date"].dt.hour * 4 + raw["Date"].dt.minute // 15

    days = sorted(day for day in raw["day"].unique() if args.start_day <= day <= args.end_day)
    rows = []
    for day in days:
        g = raw.loc[raw["day"] == day].sort_values("slot")
        true = g["Price"].to_numpy(dtype=np.float32) * 1000.0
        candidate = np.asarray([cand_map[(day, int(slot))] for slot in g["slot"]], dtype=np.float32) * 1000.0
        shift = float(true.mean() - candidate.mean())
        pred = np.clip(candidate + shift, 40.0, 1000.0)
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mean_acc = float(max(0.0, 1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), 40.0)))
        rel_acc = float(max(0.0, 1.0 - np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 40.0))))
        rows.append(
            {
                "day": day,
                "shift": shift,
                "mae": mae,
                "rmse": rmse,
                "daily_mean_accuracy": mean_acc,
                "relative_accuracy": rel_acc,
            }
        )

    payload = {
        "days": len(rows),
        "metrics": {
            "mean_day_mae": float(np.mean([row["mae"] for row in rows])),
            "mean_day_rmse": float(np.mean([row["rmse"] for row in rows])),
            "mean_daily_mean_accuracy": float(np.mean([row["daily_mean_accuracy"] for row in rows])),
            "share_days_daily_mean_accuracy_ge_0_8": float(
                (np.asarray([row["daily_mean_accuracy"] for row in rows]) >= 0.8).mean()
            ),
            "mean_relative_accuracy": float(np.mean([row["relative_accuracy"] for row in rows])),
        },
        "records": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
