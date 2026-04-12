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
from src.prompts.prompt_cot import build_cot_prompt, build_cot_target
from src.prompts.parser     import parse_output
from src.retrieval.knn_retriever import KNNRetriever
from src.retrieval.diff_features import compute_diff_features


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
        use_rag: bool = False,
        retriever: KNNRetriever | None = None,
        weather=None,
        price=None,
    ):
        self.tokenizer = tokenizer
        self.horizon   = horizon
        self.max_length = max_length
        self.node_idx  = node_idx
        self.use_rag = use_rag
        self.retriever = retriever
        self.weather = weather
        self.price = price
        self.items = self._build(samples)

    def _weather_at(self, t_start: int) -> dict | None:
        if self.weather is None or getattr(self.weather, "empty", True):
            return None
        idx = min(max(int(t_start) + 11, 0), len(self.weather) - 1)
        row = self.weather.iloc[idx]
        if hasattr(row, "to_dict"):
            d = row.to_dict()
            # normalize common keys
            for k in list(d.keys()):
                lk = str(k).lower()
                if "temp" in lk and "temperature" not in d:
                    d["temperature"] = d[k]
                if "humid" in lk and "humidity" not in d:
                    d["humidity"] = d[k]
            return d
        return None

    def _price_at(self, t_start: int) -> float | None:
        if self.price is None or getattr(self.price, "empty", True):
            return None
        idx = min(max(int(t_start) + 11, 0), len(self.price) - 1)
        row = self.price.iloc[idx]
        if np.isscalar(row):
            return float(row)
        if hasattr(row, "to_dict"):
            d = row.to_dict()
            # Prefer exact node column, fallback to mean across columns
            if self.node_idx in d:
                return float(d[self.node_idx])
            if str(self.node_idx) in d:
                return float(d[str(self.node_idx)])
            vals = [float(v) for v in d.values() if v is not None]
            if vals:
                return float(np.mean(vals))
        return None

    def _build(self, samples: list[dict]) -> list[dict]:
        items = []
        dropped_no_supervision = 0
        for s in samples:
            sample_node_idx = int(s.get("node_idx", self.node_idx))
            if self.use_rag and self.retriever is not None:
                retrieved = self.retriever.query(s, exclude_t_start=s.get("t_start"))
                weather_current = self._weather_at(s.get("t_start", 0))
                weather_retrieved = [self._weather_at(rs.get("t_start", 0)) for rs in retrieved]
                price_current = self._price_at(s.get("t_start", 0))
                price_retrieved = [self._price_at(rs.get("t_start", 0)) for rs in retrieved]
                diff = compute_diff_features(
                    query_sample=s,
                    retrieved_samples=retrieved,
                    weather_current=weather_current,
                    weather_retrieved=weather_retrieved,
                    price_current=price_current,
                    price_retrieved=price_retrieved,
                )
                sys_msg, usr_msg = build_cot_prompt(
                    s, retrieved, diff, node_idx=sample_node_idx, horizon=self.horizon
                )
                target = "\n" + build_cot_target(
                    s, retrieved, diff, node_idx=sample_node_idx, horizon=self.horizon
                )
            else:
                sys_msg, usr_msg = build_vanilla_prompt(s, sample_node_idx, self.horizon)
                # Target: the answer line
                y = s["y"][:self.horizon, sample_node_idx]
                target = " [" + ", ".join(f"{v:.3f}" for v in y) + "]"

            messages = [
                {"role": "system",    "content": sys_msg},
                {"role": "user",      "content": usr_msg},
                {"role": "assistant", "content": target},
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # Find where assistant response starts
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

            # Mask prompt tokens so loss is only on assistant target tokens.
            labels_full = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            # IMPORTANT: keep tail when truncating so assistant target is preserved.
            if len(full_ids) > self.max_length:
                input_ids = full_ids[-self.max_length:]
                labels = labels_full[-self.max_length:]
            else:
                input_ids = full_ids
                labels = labels_full

            attention_mask = [1] * len(input_ids)
            supervised_tokens = sum(1 for v in labels if v != -100)
            if supervised_tokens == 0:
                dropped_no_supervision += 1
                continue

            items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        if not items:
            raise ValueError(
                "All training samples have zero supervised tokens after truncation. "
                "Increase --max_length (e.g., 768/1024) or shorten prompt content."
            )
        if dropped_no_supervision > 0:
            print(
                f"[EVDataset] dropped {dropped_no_supervision}/{len(samples)} samples "
                f"with zero supervised tokens (max_length={self.max_length})."
            )
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {k: torch.tensor(v) for k, v in item.items()}


# ─── Evaluation helper ────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    test_samples,
    horizon,
    node_idx,
    max_samples=200,
    max_new_tokens: int = 128,
    use_rag: bool = False,
    retriever: KNNRetriever | None = None,
):
    preds, trues = [], []
    subset = test_samples[:max_samples]
    for s in subset:
        sample_node_idx = int(s.get("node_idx", node_idx))
        if use_rag and retriever is not None:
            retrieved = retriever.query(s, exclude_t_start=s.get("t_start"))
            diff = compute_diff_features(query_sample=s, retrieved_samples=retrieved)
            sys_msg, usr_msg = build_cot_prompt(s, retrieved, diff, sample_node_idx, horizon)
        else:
            sys_msg, usr_msg = build_vanilla_prompt(s, sample_node_idx, horizon)
        out = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=max_new_tokens)
        arr = parse_output(out, expected_len=horizon)
        if arr is not None and len(arr) == horizon:
            preds.append(arr)
            trues.append(s["y"][:horizon, sample_node_idx])
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
    parser.add_argument("--max_length",  type=int, default=384,
                        help="Tokenization max length (reduce for lower VRAM)")
    parser.add_argument("--use_dora",    action="store_true")
    parser.add_argument("--use_rag",     action="store_true",
                        help="Enable retrieval-augmented CoT prompts")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=True,
                        help="Enable gradient checkpointing to reduce VRAM (default: enabled)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false",
                        help="Disable gradient checkpointing")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/{dataset}_h{horizon}.pkl")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Cap training samples (for quick tests)")
    parser.add_argument("--eval_max_samples", type=int, default=200,
                        help="Cap evaluation samples for speed")
    parser.add_argument("--eval_max_new_tokens", type=int, default=128,
                        help="Generation max_new_tokens during evaluation")
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
    train_pool = train_samples_map[args.horizon]
    train_samples = train_pool[:args.max_samples]
    test_samples  = test_samples_map[args.horizon]

    # Optional RAG retriever
    retriever = None
    if args.use_rag:
        if args.retrieval_cache:
            cache_path = Path(args.retrieval_cache)
        else:
            cache_path = Path(f"data/retrieval_cache/{args.dataset}_h{args.horizon}.pkl")
        if cache_path.exists():
            retriever = KNNRetriever.load(cache_path)
            print(f"Loaded retrieval cache: {cache_path}")
        else:
            print(f"Retrieval cache not found, building in-memory retriever from train pool: {len(train_pool)}")
            retriever = KNNRetriever(train_pool, top_k=2)

    # 3. Load model
    print("Loading model …")
    model, tokenizer = load_model_and_tokenizer()
    model = get_lora_model(model, use_dora=args.use_dora)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # DoRA + RAG has notably higher memory footprint due longer prompts.
    if args.use_dora and args.use_rag and args.batch_size > 1:
        print("[MemoryGuard] Warning: DoRA+RAG with batch_size>1 may OOM on 16GB GPUs.")

    # 4. Tokenise
    train_ds = EVDataset(
        train_samples,
        tokenizer,
        args.horizon,
        max_length=args.max_length,
        node_idx=args.node_idx,
        use_rag=args.use_rag,
        retriever=retriever,
        weather=splits.get("weather"),
        price=splits.get("price"),
    )
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
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    preds, trues = run_inference(
        model,
        tokenizer,
        test_samples,
        args.horizon,
        args.node_idx,
        max_samples=args.eval_max_samples,
        max_new_tokens=args.eval_max_new_tokens,
        use_rag=args.use_rag,
        retriever=retriever,
    )
    if preds:
        metrics = per_horizon_metrics(preds, trues, args.horizon,
                                      splits["norm_min"], splits["norm_max"])
        print(json.dumps(metrics, indent=2))
        result = {
            "run_id":    out_dir.name,
            "dataset":   args.dataset,
            "horizon":   args.horizon,
            "node_idx":  args.node_idx,
            "use_rag":   args.use_rag,
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
