#!/usr/bin/env python3
"""Build a strict candidate curve map for LLM refinement experiments.

Policy:
  - train range: use previous-day true price as the candidate
  - val range:   use fused val prediction if available, else previous-day true price
  - test range:  use fused test prediction if available, else previous-day true price
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_market(source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.loc[df["Date"].notna(), ["Date", "Price"]].copy()
    df["Price"] = df["Price"].interpolate(limit_direction="both").ffill().bfill()
    df = df.loc[df["Date"].dt.year == 2025].reset_index(drop=True)
    df["day"] = df["Date"].dt.normalize()
    df["slot"] = df["Date"].dt.hour * 4 + df["Date"].dt.minute // 15
    return df


def _load_prediction_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    required = {"day", "slot", "prediction"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {sorted(required)}")
    return df[["day", "slot", "prediction"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--fused_val_csv", default="outputs/gs_price_2025_supply_demand_fused/val_predictions.csv")
    parser.add_argument("--fused_test_csv", default="outputs/gs_price_2025_supply_demand_fused/test_predictions.csv")
    parser.add_argument("--output_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--norm_min", type=float, default=0.0)
    parser.add_argument("--norm_max", type=float, default=1000.0)
    args = parser.parse_args()

    market = _load_market(Path(args.source_csv))
    truth = market[["day", "slot", "Price"]].rename(columns={"Price": "truth"})
    truth["day"] = truth["day"].dt.strftime("%Y-%m-%d")

    prev_day = truth.copy()
    prev_day["day"] = (pd.to_datetime(prev_day["day"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    prev_day = prev_day.rename(columns={"truth": "prediction"})

    fused_val = _load_prediction_csv(args.fused_val_csv) if Path(args.fused_val_csv).exists() else pd.DataFrame(columns=["day", "slot", "prediction"])
    fused_test = _load_prediction_csv(args.fused_test_csv) if Path(args.fused_test_csv).exists() else pd.DataFrame(columns=["day", "slot", "prediction"])
    overrides = pd.concat([fused_val, fused_test], ignore_index=True)
    overrides = overrides.drop_duplicates(subset=["day", "slot"], keep="last")

    candidate = prev_day.merge(overrides, on=["day", "slot"], how="left", suffixes=("_base", "_override"))
    candidate["prediction"] = candidate["prediction_override"].fillna(candidate["prediction_base"])
    denom = float(args.norm_max - args.norm_min)
    if denom <= 0:
        raise ValueError("--norm_max must be greater than --norm_min")
    candidate["prediction"] = (candidate["prediction"] - float(args.norm_min)) / denom
    candidate = candidate[["day", "slot", "prediction"]].sort_values(["day", "slot"]).reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_csv, index=False)

    summary = {
        "rows": int(len(candidate)),
        "days": int(candidate["day"].nunique()),
        "output_csv": str(output_csv),
        "used_fused_val": bool(len(fused_val) > 0),
        "used_fused_test": bool(len(fused_test) > 0),
        "normalization": {
            "norm_min": float(args.norm_min),
            "norm_max": float(args.norm_max),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
