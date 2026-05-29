#!/usr/bin/env python3
"""Train a lightweight LLM adapter to predict daily price offsets for candidate curves."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").exists() and (parent / "data").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qwen_peft import get_lora_model, load_model_and_tokenizer


EXOG_COLS = [
    "发电总出力预测",
    "竞价空间",
    "统一负荷预测",
    "抽蓄",
    "统一新能源预测",
    "联络线计划",
]


def _read_market(source_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return price, exog


def _load_candidate_map(candidate_csv: Path) -> dict[tuple[str, int], float]:
    df = pd.read_csv(candidate_csv)
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return {(str(row.day), int(row.slot)): float(row.prediction) for row in df.itertuples(index=False)}


class DailyOffsetDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = [self._encode(row) for row in rows]

    def _encode(self, row: dict) -> dict:
        system = (
            "You are an expert electricity price forecasting assistant. "
            "A candidate day-ahead price curve is provided. "
            "Your task is to output exactly one scalar offset that should be added to the whole candidate curve."
        )
        user = (
            f"Day: {row['day']}\n"
            f"Candidate day-ahead mean price: {row['candidate_mean']:.3f}\n"
            f"Candidate min/max: {row['candidate_min']:.3f} / {row['candidate_max']:.3f}\n"
            f"Previous-day mean price: {row['prev_mean']:.3f}\n"
            f"Previous 7-day mean price: {row['prev7_mean']:.3f}\n"
            f"Known exogenous summary: {row['exog_summary']}\n"
            "Output only one JSON list containing one float: [offset].\n"
            "Numerical Prediction:"
        )
        target = f" [{row['offset']:.3f}]"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(full_ids) > self.max_length:
            full_ids = full_ids[-self.max_length:]
            labels = labels[-self.max_length:]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return {key: torch.tensor(value) for key, value in item.items()}


def _build_rows(price: pd.DataFrame, exog: pd.DataFrame, candidate_map: dict[tuple[str, int], float]) -> list[dict]:
    rows = []
    days = list(price.index)
    for i, day in enumerate(days):
        day_str = str(day.date())
        if (day_str, 0) not in candidate_map:
            continue
        truth = price.iloc[i].to_numpy(dtype=np.float32) / 1000.0
        candidate = np.asarray([candidate_map[(day_str, slot)] for slot in range(96)], dtype=np.float32)
        prev = price.iloc[i - 1].to_numpy(dtype=np.float32) / 1000.0 if i > 0 else candidate
        prev7 = price.iloc[max(i - 7, 0)].to_numpy(dtype=np.float32) / 1000.0
        offset = float(np.mean(truth - candidate))
        exog_parts = []
        for col in EXOG_COLS:
            series = exog[col].iloc[i].to_numpy(dtype=np.float32)
            exog_parts.append(f"{col}: mean={series.mean():.1f}, std={series.std():.1f}, max={series.max():.1f}")
        rows.append(
            {
                "day": day_str,
                "candidate_mean": float(candidate.mean()),
                "candidate_min": float(candidate.min()),
                "candidate_max": float(candidate.max()),
                "prev_mean": float(prev.mean()),
                "prev7_mean": float(prev7.mean()),
                "exog_summary": "; ".join(exog_parts),
                "offset": float(np.clip(offset, -0.2, 0.2)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--output_dir", default="outputs/gs_price_2025_llm_daily_offset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_train_days", type=int, default=0)
    args = parser.parse_args()

    price, exog = _read_market(Path(args.source_csv))
    candidate_map = _load_candidate_map(Path(args.candidate_csv))
    rows = _build_rows(price, exog, candidate_map)
    train_rows = [row for row in rows if row["day"] <= "2025-10-20"]
    val_rows = [row for row in rows if "2025-10-21" <= row["day"] <= "2025-11-25"]
    test_rows = [row for row in rows if row["day"] > "2025-11-25"]
    if args.max_train_days > 0:
        train_rows = train_rows[: args.max_train_days]

    model, tokenizer = load_model_and_tokenizer()
    model = get_lora_model(model, use_dora=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_ds = DailyOffsetDataset(train_rows, tokenizer, args.max_length)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.label_names = ["labels"]
    trainer.train()

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "train_days": len(train_rows),
        "val_days": len(val_rows),
        "test_days": len(test_rows),
        "train_loss": float(getattr(trainer.state, "log_history", [{}])[-1].get("loss", 0.0)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
