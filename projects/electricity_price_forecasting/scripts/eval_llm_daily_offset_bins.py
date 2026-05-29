#!/usr/bin/env python3
"""Evaluate the daily-offset-bin LLM route on natural-day windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").exists() and (parent / "data").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qwen_peft import generate, load_model_and_tokenizer, load_peft_model
from src.prompts.parser import parse_output
from projects.electricity_price_forecasting.scripts.eval_llm_daily_offset import _build_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--summary_json", required=True, help="summary.json from train_llm_daily_offset_bins.py")
    parser.add_argument("--max_days", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output", default="outputs/gs_price_2025_llm_daily_offset_bins_eval.json")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text())
    centers = np.asarray(summary["bin_centers"], dtype=np.float32)
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
        pred_bin = int(round(float(parsed[0]))) if parse_ok else int(np.argmin(np.abs(centers)))
        pred_bin = int(np.clip(pred_bin, 0, len(centers) - 1))
        offset = float(centers[pred_bin])
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
                "pred_bin": pred_bin,
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
