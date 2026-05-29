"""
src/train/train_single.py
--------------------------
Fine-tune Qwen2.5-1.5B-Instruct with LoRA on a single dataset split.

Usage:
  python -m src.train.train_single \
      --dataset st_evcdp \
      --horizon 6 \
      --output_dir outputs/single_lora_h6

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

from src.data.loaders import DATASET_CHOICES, load_dataset
from src.data.build_splits  import build_splits
from src.data.build_samples import build_samples
from src.utils.history_window import price_at_history_end, sample_history_end_index, weather_at_history_end
from src.data.windowing import default_retrieval_cache_path, resolve_window_stride
from src.utils.node_context import (
    extract_node_static_context,
    normalise_domain_label,
    resolve_node_id,
)
from src.eval.metrics       import per_horizon_metrics
from src.models.qwen_peft   import load_model_and_tokenizer, get_lora_model, generate
from src.prompts.prompt_vanilla import build_direct_physical_prompt, build_vanilla_prompt
from src.prompts.prompt_cot import build_cot_prompt, build_cot_target
from src.prompts.parser     import parse_output
from src.retrieval.knn_retriever import KNNRetriever
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.retrieval_result_cache import retrieval_result_cache_key
from src.utils.price_candidate import combine_candidate_prediction


# ─── Dataset wrapper ──────────────────────────────────────────────────────────

class EVDataset(Dataset):
    """
    Each item is one (prompt, target) pair for a single node/horizon.
    The model is trained to produce only the target tokens.
    """

    PROMPT_STYLES = {"auto", "cot", "direct_physical", "vanilla"}

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
        node_meta=None,
        node_ids: list[str] | None = None,
        poi=None,
        include_env_diff: bool = False,
        prompt_style: str = "auto",
        target_style: str = "auto",
        retrieval_cache_entries: dict[str, dict] | None = None,
        precomputed_retrieval_entries: dict[str, dict] | None = None,
    ):
        self.tokenizer = tokenizer
        self.horizon = horizon
        self.max_length = max_length
        self.node_idx = node_idx
        self.use_rag = use_rag
        self.retriever = retriever
        self.weather = weather
        self.price = price
        self.node_meta = node_meta
        self.node_ids = node_ids
        self.poi = poi
        self.include_env_diff = include_env_diff
        self.retrieval_cache_entries = retrieval_cache_entries
        self.precomputed_retrieval_entries = precomputed_retrieval_entries
        if prompt_style not in self.PROMPT_STYLES:
            raise ValueError(
                f"Unsupported prompt_style={prompt_style!r}; expected one of {sorted(self.PROMPT_STYLES)}"
            )
        if prompt_style in {"cot", "direct_physical"} and not use_rag:
            raise ValueError(f"prompt_style={prompt_style!r} requires use_rag=True")
        if include_env_diff and not use_rag:
            raise ValueError("include_env_diff=True requires use_rag=True")
        self.prompt_style = prompt_style
        if target_style not in {"auto", "numeric_only", "candidate_residual", "candidate_selective_residual", "candidate_chunk_offset"}:
            raise ValueError("target_style must be one of: auto, numeric_only")
        self.target_style = target_style
        self.items = self._build(samples)

    def _history_end_index(self, sample: dict, frame) -> int:
        return sample_history_end_index(sample, frame)

    def _weather_at(self, sample: dict) -> dict | None:
        return weather_at_history_end(self.weather, sample)

    def _node_id(self, sample_node_idx: int) -> str | int:
        return resolve_node_id(sample_node_idx, node_ids=self.node_ids, node_meta=self.node_meta)

    def _static_context(self, sample_node_idx: int) -> dict:
        return extract_node_static_context(
            sample_node_idx,
            node_ids=self.node_ids,
            node_meta=self.node_meta,
        )

    def _price_at(self, sample: dict, sample_node_idx: int) -> float | None:
        return price_at_history_end(
            self.price,
            sample,
            sample_node_idx,
            node_ids=self.node_ids,
            node_meta=self.node_meta,
        )

    def _numeric_target(self, y: np.ndarray) -> str:
        return " [" + ", ".join(f"{v:.3f}" for v in y) + "]"

    def _effective_prompt_style(self, has_retrieval: bool) -> str:
        if self.prompt_style == "auto":
            return "cot" if has_retrieval else "vanilla"
        if self.prompt_style in {"cot", "direct_physical"} and not has_retrieval:
            raise ValueError(
                f"prompt_style={self.prompt_style!r} requires retrieval context, but no retriever was available."
            )
        return self.prompt_style

    def _build(self, samples: list[dict]) -> list[dict]:
        items = []
        dropped_no_supervision = 0
        for s in samples:
            sample_node_idx = int(s.get("node_idx", self.node_idx))
            static_context = self._static_context(sample_node_idx)
            domain_label = normalise_domain_label(static_context.get("zone_type"))
            retrieved = []
            diff = None
            if self.use_rag and self.retriever is not None:
                cache_key = retrieval_result_cache_key(s, sample_node_idx)
                cached = None
                if self.precomputed_retrieval_entries is not None:
                    cached = self.precomputed_retrieval_entries.get(cache_key)
                if cached is None and self.retrieval_cache_entries is not None:
                    cached = self.retrieval_cache_entries.get(cache_key)
                if cached is not None:
                    retrieved_idx = cached.get("retrieved_indices", [])
                    retrieved = [self.retriever.pool[int(i)] for i in retrieved_idx]
                    diff = cached.get("diff")
                else:
                    retrieved_idx = self.retriever.query_indices(s, exclude_t_start=s.get("t_start"))
                    retrieved = [self.retriever.pool[int(i)] for i in retrieved_idx]
                    weather_current = self._weather_at(s)
                    weather_retrieved = [self._weather_at(rs) for rs in retrieved]
                    price_current = self._price_at(s, sample_node_idx)
                    price_retrieved = [self._price_at(rs, sample_node_idx) for rs in retrieved]
                    diff = compute_diff_features(
                        query_sample=s,
                        retrieved_samples=retrieved,
                        weather_current=weather_current,
                        weather_retrieved=weather_retrieved,
                        price_current=price_current,
                        price_retrieved=price_retrieved,
                        node_idx=sample_node_idx,
                    )
                    if self.retrieval_cache_entries is not None:
                        self.retrieval_cache_entries[cache_key] = {
                            "retrieved_indices": [int(i) for i in retrieved_idx],
                            "diff": diff,
                        }

            prompt_style = self._effective_prompt_style(bool(retrieved) and diff is not None)
            include_env_diff = self.include_env_diff and prompt_style != "vanilla"
            if prompt_style == "cot":
                sys_msg, usr_msg = build_cot_prompt(
                    s,
                    retrieved,
                    diff,
                    node_idx=sample_node_idx,
                    horizon=self.horizon,
                    domain_label=domain_label,
                    static_context=static_context,
                    include_env_diff=include_env_diff,
                )
                y = s["y"][:self.horizon, sample_node_idx]
                if self.target_style == "numeric_only":
                    # The COT structure remains in the prompt; the supervised
                    # response focuses the loss on the final forecast sequence.
                    target = self._numeric_target(y)
                else:
                    target = "\n" + build_cot_target(
                        s,
                        retrieved,
                        diff,
                        node_idx=sample_node_idx,
                        horizon=self.horizon,
                        static_context=static_context,
                        include_env_diff=include_env_diff,
                    )
            elif prompt_style == "direct_physical":
                sys_msg, usr_msg = build_direct_physical_prompt(
                    s,
                    retrieved,
                    diff,
                    node_idx=sample_node_idx,
                    horizon=self.horizon,
                    domain_label=domain_label,
                    static_context=static_context,
                    include_env_diff=include_env_diff,
                    target_mode=(
                        "chunk_offset"
                        if self.target_style == "candidate_chunk_offset"
                        else (
                        "selective_residual"
                        if self.target_style == "candidate_selective_residual"
                        else ("residual" if self.target_style == "candidate_residual" else "absolute")
                        )
                    ),
                )
                y = s["y"][:self.horizon, sample_node_idx]
                if self.target_style in {"candidate_residual", "candidate_selective_residual", "candidate_chunk_offset"}:
                    candidate = s.get("candidate_future")
                    if candidate is None:
                        raise ValueError("candidate residual targets require candidate_future in samples.")
                    residual = np.asarray(y, dtype=np.float32) - np.asarray(candidate[: self.horizon], dtype=np.float32)
                    if self.target_style == "candidate_chunk_offset":
                        residual = np.full_like(residual, float(np.mean(residual)))
                    elif self.target_style == "candidate_selective_residual":
                        mask = s.get("candidate_refine_mask")
                        if mask is None:
                            raise ValueError("candidate_selective_residual requires candidate_refine_mask in samples.")
                        residual = residual * np.asarray(mask[: self.horizon], dtype=np.float32)
                    residual = np.clip(residual, -0.2, 0.2)
                    target = self._numeric_target(residual)
                else:
                    target = self._numeric_target(y)
            else:
                sys_msg, usr_msg = build_vanilla_prompt(
                    s,
                    sample_node_idx,
                    self.horizon,
                    domain_label=domain_label,
                    static_context=static_context,
                    target_mode=(
                        "chunk_offset"
                        if self.target_style == "candidate_chunk_offset"
                        else (
                        "selective_residual"
                        if self.target_style == "candidate_selective_residual"
                        else ("residual" if self.target_style == "candidate_residual" else "absolute")
                        )
                    ),
                )
                y = s["y"][:self.horizon, sample_node_idx]
                if self.target_style in {"candidate_residual", "candidate_selective_residual", "candidate_chunk_offset"}:
                    candidate = s.get("candidate_future")
                    if candidate is None:
                        raise ValueError("candidate residual targets require candidate_future in samples.")
                    residual = np.asarray(y, dtype=np.float32) - np.asarray(candidate[: self.horizon], dtype=np.float32)
                    if self.target_style == "candidate_chunk_offset":
                        residual = np.full_like(residual, float(np.mean(residual)))
                    elif self.target_style == "candidate_selective_residual":
                        mask = s.get("candidate_refine_mask")
                        if mask is None:
                            raise ValueError("candidate_selective_residual requires candidate_refine_mask in samples.")
                        residual = residual * np.asarray(mask[: self.horizon], dtype=np.float32)
                    residual = np.clip(residual, -0.2, 0.2)
                    target = self._numeric_target(residual)
                else:
                    target = self._numeric_target(y)

            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
                {"role": "assistant", "content": target},
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

            labels_full = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

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
    node_meta=None,
    node_ids: list[str] | None = None,
    sampling: str = "head",
    seed: int = 42,
    include_env_diff: bool = False,
):
    preds, trues = [], []
    if sampling == "random":
        rng = random.Random(seed)
        k = min(max_samples, len(test_samples))
        subset = rng.sample(test_samples, k)
    else:
        subset = test_samples[:max_samples]
    for s in subset:
        sample_node_idx = int(s.get("node_idx", node_idx))
        static_context = extract_node_static_context(
            sample_node_idx,
            node_ids=node_ids,
            node_meta=node_meta,
        )
        domain_label = normalise_domain_label(static_context.get("zone_type"))
        if use_rag and retriever is not None:
            retrieved = retriever.query(s, exclude_t_start=None)
            diff = compute_diff_features(
                query_sample=s,
                retrieved_samples=retrieved,
                node_idx=sample_node_idx,
            )
            sys_msg, usr_msg = build_cot_prompt(
                s,
                retrieved,
                diff,
                sample_node_idx,
                horizon,
                domain_label=domain_label,
                static_context=static_context,
                include_env_diff=include_env_diff,
            )
        else:
            sys_msg, usr_msg = build_vanilla_prompt(
                s,
                sample_node_idx,
                horizon,
                domain_label=domain_label,
                static_context=static_context,
            )
        out = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=max_new_tokens)
        arr = parse_output(out, expected_len=horizon)
        if arr is not None and len(arr) == horizon:
            preds.append(arr)
            trues.append(s["y"][:horizon, sample_node_idx])
    return preds, trues


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="st_evcdp", choices=DATASET_CHOICES)
    parser.add_argument("--horizon",     type=int, default=6)
    parser.add_argument("--node_idx",    type=int, default=0)
    parser.add_argument("--output_dir",  default="outputs/single_lora_h6")
    parser.add_argument("--epochs",      type=int, default=2)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max_length",  type=int, default=2560,
                        help="Tokenization max length; the paper uses 2560.")
    parser.add_argument("--lora_rank",   type=int, default=32,
                        help="LoRA rank; the paper uses 32.")
    parser.add_argument("--lora_alpha",  type=int, default=32,
                        help="LoRA alpha (scaling factor); the paper uses 32.")
    parser.add_argument("--history_len", type=int, default=12,
                        help="Historical observation window; the paper uses 12.")
    parser.add_argument("--neighbor_k",  type=int, default=7,
                        help="Neighbour top-k used for spatial context; the paper uses 7.")
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help="Window step size. `0` means use `horizon` (non-overlapping targets); `1` keeps classic overlapping sliding windows.",
    )
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
    parser.add_argument("--eval_sampling", choices=["head", "random"], default="head",
                        help="Evaluation sample selection: head=first N, random=random N")
    parser.add_argument("--eval_seed", type=int, default=42,
                        help="Random seed for evaluation sampling when eval_sampling=random")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_window_stride = resolve_window_stride(args.window_stride, horizon=args.horizon)

    # 1. Load data
    print("Loading dataset …")
    raw = load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)

    # 2. Build samples
    print("Building samples …")
    train_samples_map = build_samples(
        splits["train"], splits["timestamps_train"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
        window_stride=effective_window_stride,
    )
    test_samples_map = build_samples(
        splits["test"], splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
        window_stride=effective_window_stride,
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
            cache_path = default_retrieval_cache_path(args.dataset, args.horizon, effective_window_stride)
        if cache_path.exists():
            retriever = KNNRetriever.load(cache_path)
            print(f"Loaded retrieval cache: {cache_path}")
        else:
            print(f"Retrieval cache not found, building in-memory retriever from train pool: {len(train_pool)}")
            retriever = KNNRetriever(train_pool, top_k=2)

    # 3. Load model
    print("Loading model …")
    model, tokenizer = load_model_and_tokenizer()
    model = get_lora_model(model, use_dora=args.use_dora,
                           r=args.lora_rank, lora_alpha=args.lora_alpha)
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
        node_meta=splits.get("node_meta"),
        node_ids=splits.get("node_ids"),
        poi=splits.get("poi"),
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    # 5. Train
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
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
        node_meta=splits.get("node_meta"),
        node_ids=splits.get("node_ids"),
        sampling=args.eval_sampling,
        seed=args.eval_seed,
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
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_length": args.max_length,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "history_len": args.history_len,
                "neighbor_k": args.neighbor_k,
                "use_dora": args.use_dora,
            },
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
