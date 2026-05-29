#!/usr/bin/env python3
"""Calibrate daily-offset LLM outputs using the 2025 validation split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").exists() and (parent / "data").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qwen_peft import generate, load_model_and_tokenizer, load_peft_model
from src.prompts.parser import parse_output
from projects.electricity_price_forecasting.scripts.eval_llm_daily_offset import _build_rows


def _eval_rows(rows: list[dict], offsets: np.ndarray) -> dict:
    metrics = []
    for row, offset in zip(rows, offsets):
        pred = np.clip(row["candidate"] + offset * 1000.0, 40.0, 1000.0)
        true = row["truth"]
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mean_acc = float(max(0.0, 1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), 40.0)))
        rel_acc = float(max(0.0, 1.0 - np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 40.0))))
        metrics.append({"day": row["day"], "offset": float(offset), "mae": mae, "rmse": rmse, "daily_mean_accuracy": mean_acc, "relative_accuracy": rel_acc})
    return {
        "days": len(metrics),
        "metrics": {
            "mean_day_mae": float(np.mean([m["mae"] for m in metrics])) if metrics else None,
            "mean_day_rmse": float(np.mean([m["rmse"] for m in metrics])) if metrics else None,
            "mean_daily_mean_accuracy": float(np.mean([m["daily_mean_accuracy"] for m in metrics])) if metrics else None,
            "share_days_daily_mean_accuracy_ge_0_8": float((np.asarray([m["daily_mean_accuracy"] for m in metrics]) >= 0.8).mean()) if metrics else None,
            "mean_relative_accuracy": float(np.mean([m["relative_accuracy"] for m in metrics])) if metrics else None,
        },
        "records": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--offset_clip", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output", default="outputs/gs_price_2025_llm_daily_offset_calibrated_eval.json")
    args = parser.parse_args()

    rows = _build_rows(Path(args.source_csv), Path(args.candidate_csv))
    val_rows = [row for row in rows if "2025-10-21" <= row["day"] <= "2025-11-25"]
    test_rows = [row for row in rows if row["day"] > "2025-11-25"]

    base_model, tokenizer = load_model_and_tokenizer()
    model = load_peft_model(base_model, args.adapter_dir)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    def infer(rows_: list[dict]) -> np.ndarray:
        outputs = []
        for row in rows_:
            sys_msg, usr_msg = row["prompt"]
            raw_output = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=args.max_new_tokens)
            parsed = parse_output(raw_output, expected_len=1)
            offset = float(parsed[0]) if parsed is not None and len(parsed) >= 1 else 0.0
            offset = float(np.clip(offset, -args.offset_clip, args.offset_clip))
            outputs.append(offset)
        return np.asarray(outputs, dtype=np.float32)

    val_pred = infer(val_rows)
    val_true = np.asarray([row["truth"].mean() - row["candidate"].mean() for row in val_rows], dtype=np.float32) / 1000.0
    reg = LinearRegression().fit(val_pred.reshape(-1, 1), val_true)
    test_pred = infer(test_rows)
    test_cal = reg.predict(test_pred.reshape(-1, 1)).astype(np.float32)

    payload = {
        "calibration": {
            "slope": float(reg.coef_[0]),
            "intercept": float(reg.intercept_),
        },
        "raw_test": _eval_rows(test_rows, test_pred),
        "calibrated_test": _eval_rows(test_rows, test_cal),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload["calibrated_test"]["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
