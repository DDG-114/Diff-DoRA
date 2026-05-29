#!/usr/bin/env python3
"""Train an LLM adapter to classify daily offset bins for candidate curves."""
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
from projects.electricity_price_forecasting.scripts.train_llm_daily_offset import (
    EXOG_COLS,
    _build_rows,
    _load_candidate_map,
    _read_market,
)


def _fit_bins(train_rows: list[dict], num_bins: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray([row["offset"] for row in train_rows], dtype=np.float32)
    edges = np.quantile(offsets, np.linspace(0.0, 1.0, num_bins + 1))
    centers = []
    for idx in range(num_bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == num_bins - 1:
            mask = (offsets >= lo) & (offsets <= hi)
        else:
            mask = (offsets >= lo) & (offsets < hi)
        values = offsets[mask]
        centers.append(float(values.mean()) if len(values) else float((lo + hi) / 2.0))
    return edges.astype(np.float32), np.asarray(centers, dtype=np.float32)


def _assign_bin(offset: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges[1:-1], offset, side="right"))
    return idx


class DailyOffsetBinDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int, edges: np.ndarray) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.edges = edges
        self.items = [self._encode(row) for row in rows]

    def _encode(self, row: dict) -> dict:
        bin_id = _assign_bin(float(row["offset"]), self.edges)
        system = (
            "You are an expert electricity price forecasting assistant. "
            "A candidate day-ahead curve is provided. "
            "Your task is to choose the most suitable daily offset bin."
        )
        user = (
            f"Day: {row['day']}\n"
            f"Candidate day-ahead mean price: {row['candidate_mean']:.3f}\n"
            f"Candidate min/max: {row['candidate_min']:.3f} / {row['candidate_max']:.3f}\n"
            f"Previous-day mean price: {row['prev_mean']:.3f}\n"
            f"Previous 7-day mean price: {row['prev7_mean']:.3f}\n"
            f"Known exogenous summary: {row['exog_summary']}\n"
            "Output only one JSON list containing one integer offset-bin id: [bin_id].\n"
            "Numerical Prediction:"
        )
        target = f" [{bin_id}]"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--output_dir", default="outputs/gs_price_2025_llm_daily_offset_bins")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_train_days", type=int, default=0)
    parser.add_argument("--num_bins", type=int, default=5)
    args = parser.parse_args()

    price, exog = _read_market(Path(args.source_csv))
    candidate_map = _load_candidate_map(Path(args.candidate_csv))
    rows = _build_rows(price, exog, candidate_map)
    train_rows = [row for row in rows if row["day"] <= "2025-10-20"]
    val_rows = [row for row in rows if "2025-10-21" <= row["day"] <= "2025-11-25"]
    test_rows = [row for row in rows if row["day"] > "2025-11-25"]
    if args.max_train_days > 0:
        train_rows = train_rows[: args.max_train_days]

    edges, centers = _fit_bins(train_rows, args.num_bins)

    model, tokenizer = load_model_and_tokenizer()
    model = get_lora_model(model, use_dora=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_ds = DailyOffsetBinDataset(train_rows, tokenizer, args.max_length, edges)
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
        "num_bins": int(args.num_bins),
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
