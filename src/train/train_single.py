"""
src/train/train_single.py
--------------------------
Fine-tune Qwen2.5-1.5B-Instruct with LoRA on a single dataset split.

Usage:
  python -m src.train.train_single \
      --dataset st_evcdp \
      --horizon 6 \
      --output_dir outputs/single_lora_h6 \
      --epochs 3

The script:
1. Loads and splits the dataset
2. Builds training samples (horizon-specific)
3. Tokenises each sample as an instruction-tuning example
4. Fine-tunes with the Hugging Face Trainer
5. Saves checkpoint + metrics JSON
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev  import load_urbanev
from src.data.build_splits  import build_splits
from src.data.build_samples import build_samples
from src.eval.metrics       import per_horizon_metrics
from src.models.qwen_peft   import load_model_and_tokenizer, get_lora_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.parser     import parse_output


# ─── Dataset wrapper ──────────────────────────────────────────────────────────

class EVDataset(Dataset):
    """
    Each item is one (prompt, target) pair for a single node/horizon.
    The model is trained to produce only the target tokens.
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        horizon: int,
        max_length: int = 512,
        node_idx: int = 0,
    ):
        self.tokenizer = tokenizer
        self.horizon   = horizon
        self.max_length = max_length
        self.node_idx  = node_idx
        self.items = self._build(samples)

    def _build(self, samples: list[dict]) -> list[dict]:
        from src.prompts.prompt_vanilla import build_vanilla_prompt, SYSTEM_MSG
        import json as _json
        items = []
        for s in samples:
            sys_msg, usr_msg = build_vanilla_prompt(s, self.node_idx, self.horizon)
            # Target: the answer line
            y = s["y"][:self.horizon, self.node_idx]
            target = " [" + ", ".join(f"{v:.3f}" for v in y) + "]"

            messages = [
                {"role": "system",    "content": sys_msg},
                {"role": "user",      "content": usr_msg},
                {"role": "assistant", "content": target},
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            enc = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
            )
            # Mask prompt tokens so loss is only on target
            input_ids = enc["input_ids"]
            # Find where assistant response starts
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[:self.max_length]

            items.append({
                "input_ids": input_ids,
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            })
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {k: torch.tensor(v) for k, v in item.items()}


# ─── Evaluation helper ────────────────────────────────────────────────────────

def run_inference(model, tokenizer, test_samples, horizon, node_idx, max_samples=200):
    preds, trues = [], []
    subset = test_samples[:max_samples]
    for s in subset:
        sys_msg, usr_msg = build_vanilla_prompt(s, node_idx, horizon)
        out = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=128)
        arr = parse_output(out, expected_len=horizon)
        if arr is not None and len(arr) == horizon:
            preds.append(arr)
            trues.append(s["y"][:horizon, node_idx])
    return preds, trues


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon",     type=int, default=6)
    parser.add_argument("--node_idx",    type=int, default=0)
    parser.add_argument("--output_dir",  default="outputs/single_lora_h6")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--use_dora",    action="store_true")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Cap training samples (for quick tests)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading dataset …")
    if args.dataset == "st_evcdp":
        raw = load_st_evcdp()
    else:
        raw = load_urbanev()
    splits = build_splits(raw, args.dataset)

    # 2. Build samples
    print("Building samples …")
    train_samples_map = build_samples(
        splits["train"], splits["timestamps_train"],
        adj=splits.get("adj"), horizons=[args.horizon]
    )
    test_samples_map = build_samples(
        splits["test"], splits["timestamps_test"],
        adj=splits.get("adj"), horizons=[args.horizon]
    )
    train_samples = train_samples_map[args.horizon][:args.max_samples]
    test_samples  = test_samples_map[args.horizon]

    # 3. Load model
    print("Loading model …")
    model, tokenizer = load_model_and_tokenizer()
    model = get_lora_model(model, use_dora=args.use_dora)

    # 4. Tokenise
    train_ds = EVDataset(train_samples, tokenizer, args.horizon, node_idx=args.node_idx)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    # 5. Train
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
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
    print("Training …")
    trainer.train()
    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    print("Adapter saved.")

    # 6. Evaluate
    print("Evaluating …")
    model.eval()
    preds, trues = run_inference(model, tokenizer, test_samples, args.horizon, args.node_idx)
    if preds:
        metrics = per_horizon_metrics(preds, trues, args.horizon,
                                      splits["norm_min"], splits["norm_max"])
        print(json.dumps(metrics, indent=2))
        result = {
            "run_id":    out_dir.name,
            "dataset":   args.dataset,
            "horizon":   args.horizon,
            "node_idx":  args.node_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics":   metrics,
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_dir}/metrics.json")
    else:
        print("No parseable predictions.")


if __name__ == "__main__":
    main()
