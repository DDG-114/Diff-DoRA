#!/usr/bin/env python3
"""Evaluate the daily-offset LLM route on natural-day windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").exists() and (parent / "data").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qwen_peft import generate, load_model_and_tokenizer, load_peft_model
from src.prompts.parser import parse_output


EXOG_COLS = [
    "发电总出力预测",
    "竞价空间",
    "统一负荷预测",
    "抽蓄",
    "统一新能源预测",
    "联络线计划",
]


def _build_rows(source_csv: Path, candidate_csv: Path):
    raw = pd.read_csv(source_csv)
    raw.columns = [str(col).lstrip("\ufeff").strip() for col in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", format="mixed")
    for col in ["Price", *EXOG_COLS]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").interpolate().ffill().bfill()
    raw = raw.loc[raw["Date"].dt.year == 2025].copy()
    raw["day"] = raw["Date"].dt.normalize()
    raw["slot"] = raw["Date"].dt.hour * 4 + raw["Date"].dt.minute // 15
    price = raw.pivot(index="day", columns="slot", values="Price").sort_index()
    exog = raw.pivot_table(index="day", columns="slot", values=EXOG_COLS).sort_index()

    cand = pd.read_csv(candidate_csv)
    cand.columns = [str(col).lstrip("\ufeff").strip() for col in cand.columns]
    cand_map = {(str(row.day), int(row.slot)): float(row.prediction) for row in cand.itertuples(index=False)}

    rows = []
    for i, day in enumerate(price.index):
        day_str = str(day.date())
        if (day_str, 0) not in cand_map:
            continue
        truth = price.loc[day].to_numpy(dtype=np.float32)
        candidate_norm = np.asarray([cand_map[(day_str, slot)] for slot in range(96)], dtype=np.float32)
        candidate = candidate_norm * 1000.0
        prev_norm = price.iloc[i - 1].to_numpy(dtype=np.float32) / 1000.0 if i > 0 else candidate_norm
        prev7_norm = price.iloc[max(i - 7, 0)].to_numpy(dtype=np.float32) / 1000.0
        exog_parts = []
        for col in EXOG_COLS:
            series = exog[col].loc[day].to_numpy(dtype=np.float32)
            exog_parts.append(f"{col}: mean={series.mean():.1f}, std={series.std():.1f}, max={series.max():.1f}")
        rows.append(
            {
                "day": day_str,
                "truth": truth,
                "candidate": candidate,
                "prompt": (
                    "You are an expert electricity price forecasting assistant. "
                    "A candidate day-ahead curve is given. Output one scalar offset in a JSON list like [offset].",
                    f"Day: {day_str}\n"
                    f"Candidate day-ahead mean price: {candidate_norm.mean():.3f}\n"
                    f"Candidate min/max: {candidate_norm.min():.3f} / {candidate_norm.max():.3f}\n"
                    f"Previous-day mean price: {prev_norm.mean():.3f}\n"
                    f"Previous 7-day mean price: {prev7_norm.mean():.3f}\n"
                    f"Known exogenous summary: {'; '.join(exog_parts)}\n"
                    "Output only one JSON list containing one float: [offset].\n"
                    "Numerical Prediction:",
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--max_days", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--offset_clip", type=float, default=0.2)
    parser.add_argument("--output", default="outputs/gs_price_2025_llm_daily_offset_eval.json")
    args = parser.parse_args()

    rows = _build_rows(Path(args.source_csv), Path(args.candidate_csv))
    test_rows = [row for row in rows if row["day"] > "2025-11-25"]
    if args.max_days > 0:
        test_rows = test_rows[: args.max_days]

    base_model, tokenizer = load_model_and_tokenizer()
    model = load_peft_model(base_model, args.adapter_dir)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    records = []
    for row in test_rows:
        sys_msg, usr_msg = row["prompt"]
        raw_output = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=args.max_new_tokens)
        parsed = parse_output(raw_output, expected_len=1)
        parse_ok = parsed is not None and len(parsed) >= 1
        offset = float(parsed[0]) if parse_ok else 0.0
        offset = float(np.clip(offset, -args.offset_clip, args.offset_clip))
        pred = np.clip(row["candidate"] + offset * 1000.0, 40.0, 1000.0)
        true = row["truth"]
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mean_acc = float(max(0.0, 1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), 40.0)))
        rel_acc = float(max(0.0, 1.0 - np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 40.0))))
        records.append(
            {
                "day": row["day"],
                "parse_ok": parse_ok,
                "offset": offset,
                "mae": mae,
                "rmse": rmse,
                "daily_mean_accuracy": mean_acc,
                "relative_accuracy": rel_acc,
                "raw_generation": raw_output,
            }
        )

    payload = {
        "days": len(records),
        "parse_success_rate": float(np.mean([row["parse_ok"] for row in records])) if records else 0.0,
        "metrics": {
            "mean_day_mae": float(np.mean([row["mae"] for row in records])) if records else None,
            "mean_day_rmse": float(np.mean([row["rmse"] for row in records])) if records else None,
            "mean_daily_mean_accuracy": float(np.mean([row["daily_mean_accuracy"] for row in records])) if records else None,
            "share_days_daily_mean_accuracy_ge_0_8": float(
                (np.asarray([row["daily_mean_accuracy"] for row in records]) >= 0.8).mean()
            ) if records else None,
            "mean_relative_accuracy": float(np.mean([row["relative_accuracy"] for row in records])) if records else None,
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
