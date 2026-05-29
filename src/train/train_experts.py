"""
src/train/train_experts.py
---------------------------
Train two independent expert adapters (CBD + Residential) using hard routing.

Usage:
  python -m src.train.train_experts \
      --dataset st_evcdp \
      --horizon 6 \
      --output_dir outputs/moe_experts_h6 \
      --use_dora --use_diff_dora --use_rag
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

from src.data.loaders import DATASET_CHOICES, load_dataset
from src.data.build_splits    import build_splits
from src.data.build_samples   import build_samples
from src.data.sample_cache    import default_expert_sample_cache_path, load_or_build_expert_sample_cache
from src.data.windowing       import default_retrieval_cache_path, resolve_window_stride
from src.utils.history_window import price_at_history_end, weather_at_history_end
from src.utils.node_context   import extract_node_static_context, resolve_node_id
from src.eval.metrics         import per_horizon_metrics
from src.models.qwen_peft     import load_model_and_tokenizer, load_tokenizer, get_lora_model, load_peft_model, generate
from src.prompts.prompt_vanilla import build_direct_physical_prompt, build_vanilla_prompt
from src.prompts.prompt_cot   import build_cot_prompt
from src.prompts.parser       import parse_output
from src.retrieval.knn_retriever import KNNRetriever
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.retrieval_result_cache import (
    can_use_retrieval_result_cache,
    default_expert_retrieval_cache_dir,
    expert_split_retrieval_cache_path,
    load_retrieval_result_cache,
    retrieval_result_cache_metadata,
    save_retrieval_result_cache,
)
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router  import HardRouter
from src.train.train_single   import EVDataset   # re-use dataset class
from src.train.tokenized_cache import (
    TokenizedItemsDataset,
    can_use_tokenized_items_cache,
    expert_split_tokenized_cache_path,
    load_tokenized_items_cache,
    save_tokenized_items_cache,
    tokenized_cache_metadata,
)


class BestExpertSnapshotCallback(TrainerCallback):
    """Persist the best adapter snapshot without full Trainer checkpoints."""

    def __init__(self, *, peft_model, tokenizer, adapter_dir: Path):
        self.peft_model = peft_model
        self.tokenizer = tokenizer
        self.adapter_dir = adapter_dir
        self.best_metric: float | None = None

    @property
    def metadata_path(self) -> Path:
        return self.adapter_dir.parent / "best_snapshot.json"

    def _write_snapshot(self, *, metric: float | None, state, reason: str):
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(self.adapter_dir))
        self.tokenizer.save_pretrained(str(self.adapter_dir))

        payload = {
            "best_eval_loss": metric,
            "global_step": getattr(state, "global_step", None),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.metadata_path, "w") as f:
            json.dump(payload, f, indent=2)
        if metric is not None:
            self.best_metric = metric

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metric = None if metrics is None else metrics.get("eval_loss")
        if metric is None:
            return control
        metric = float(metric)
        if self.best_metric is None or metric < self.best_metric:
            self._write_snapshot(metric=metric, state=state, reason="eval_improved")
        return control

    def ensure_snapshot(self, state, reason: str = "train_end_fallback"):
        if not self.adapter_dir.exists():
            self._write_snapshot(metric=self.best_metric, state=state, reason=reason)

    def on_train_end(self, args, state, control, **kwargs):
        self.ensure_snapshot(state)
        return control


def _resolve_prompt_style(prompt_style: str, has_retrieval: bool) -> str:
    if prompt_style == "auto":
        return "cot" if has_retrieval else "vanilla"
    if prompt_style in {"cot", "direct_physical"} and not has_retrieval:
        raise ValueError(
            f"prompt_style={prompt_style!r} requires retrieval context, but no retriever was available."
        )
    return prompt_style


def _weather_at(weather, sample: dict) -> dict | None:
    return weather_at_history_end(weather, sample)


def _price_at(price, sample: dict, node_idx: int, *, node_ids=None, node_meta=None) -> float | None:
    return price_at_history_end(price, sample, node_idx, node_ids=node_ids, node_meta=node_meta)


def _compute_retrieval_diff(sample: dict, retrieved: list[dict], splits: dict, node_idx: int) -> dict:
    weather_current = _weather_at(splits.get("weather"), sample)
    weather_retrieved = [_weather_at(splits.get("weather"), rs) for rs in retrieved]
    price_current = _price_at(
        splits.get("price"),
        sample,
        node_idx,
        node_ids=splits.get("node_ids"),
        node_meta=splits.get("node_meta"),
    )
    price_retrieved = [
        _price_at(
            splits.get("price"),
            rs,
            node_idx,
            node_ids=splits.get("node_ids"),
            node_meta=splits.get("node_meta"),
        )
        for rs in retrieved
    ]
    return compute_diff_features(
        query_sample=sample,
        retrieved_samples=retrieved,
        weather_current=weather_current,
        weather_retrieved=weather_retrieved,
        price_current=price_current,
        price_retrieved=price_retrieved,
        node_idx=node_idx,
    )


def _warn_legacy_controller_checkpoint(ctrl_path: Path) -> None:
    if ctrl_path.exists():
        print(
            f"[WARN] Ignoring legacy Diff-DoRA controller checkpoint: {ctrl_path} "
            "(paper-style Diff-DoRA is prompt-only now)."
        )


def _is_within_sample_cap(current_len: int, cap: int) -> bool:
    """Interpret non-positive caps as 'no limit' for strict/full-data runs."""
    return cap <= 0 or current_len < cap


def _split_train_val_samples(samples_with_node: list[dict]) -> tuple[list[dict], list[dict]]:
    val_split_idx = int(len(samples_with_node) * 0.85)
    return samples_with_node[:val_split_idx], samples_with_node[val_split_idx:]


def _build_eval_prompt(sample, node_idx: int, splits: dict, args, retriever: KNNRetriever | None, domain_name: str):
    static_context = extract_node_static_context(
        node_idx,
        node_ids=splits.get("node_ids"),
        node_meta=splits.get("node_meta"),
    )
    retrieved = []
    diff = None
    if args.use_rag and retriever is not None:
        retrieved = retriever.query(sample, exclude_t_start=None)
        diff = _compute_retrieval_diff(sample, retrieved, splits, node_idx)

    prompt_style = _resolve_prompt_style(args.prompt_style, bool(retrieved) and diff is not None)
    include_env_diff = args.use_diff_dora and prompt_style != "vanilla"
    if prompt_style == "cot":
        sys_msg, usr_msg = build_cot_prompt(
            sample,
            retrieved,
            diff,
            node_idx,
            args.horizon,
            domain_label=domain_name,
            static_context=static_context,
            include_env_diff=include_env_diff,
        )
    elif prompt_style == "direct_physical":
        sys_msg, usr_msg = build_direct_physical_prompt(
            sample,
            retrieved,
            diff,
            node_idx=node_idx,
            horizon=args.horizon,
            domain_label=domain_name,
            static_context=static_context,
            include_env_diff=include_env_diff,
        )
    else:
        sys_msg, usr_msg = build_vanilla_prompt(
            sample,
            node_idx,
            args.horizon,
            domain_label=domain_name,
            static_context=static_context,
        )
    return sys_msg, usr_msg


def train_one_expert(
    expert_id: int,
    samples_with_node: list[dict] | None,
    out_dir: Path,
    args,
    retriever: KNNRetriever | None = None,
    weather=None,
    price=None,
    node_meta=None,
    node_ids: list[str] | None = None,
    poi=None,
    tokenized_cache_dir: Path | None = None,
    retrieval_cache_dir: Path | None = None,
    variant_name: str = "default",
):
    train_ds = None
    val_ds = None
    train_cache_path = None
    val_cache_path = None
    train_cache_meta = None
    val_cache_meta = None
    retrieval_train_path = None
    retrieval_val_path = None
    retrieval_train_meta = None
    retrieval_val_meta = None

    if tokenized_cache_dir is not None:
        train_cache_path = expert_split_tokenized_cache_path(tokenized_cache_dir, expert_id, "train")
        val_cache_path = expert_split_tokenized_cache_path(tokenized_cache_dir, expert_id, "val")
        train_cache_meta = tokenized_cache_metadata(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            context_history_len=args.context_history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.window_stride,
            max_length=args.max_length,
            prompt_style=args.prompt_style,
            variant=variant_name,
            expert_id=expert_id,
            split="train",
            use_rag=args.use_rag,
            include_env_diff=args.use_diff_dora,
            max_samples_per_expert=args.max_samples_per_expert,
            retrieval_bank_max_samples_per_expert=args.retrieval_bank_max_samples_per_expert,
        )
        val_cache_meta = tokenized_cache_metadata(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            context_history_len=args.context_history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.window_stride,
            max_length=args.max_length,
            prompt_style=args.prompt_style,
            variant=variant_name,
            expert_id=expert_id,
            split="val",
            use_rag=args.use_rag,
            include_env_diff=args.use_diff_dora,
            max_samples_per_expert=args.max_samples_per_expert,
            retrieval_bank_max_samples_per_expert=args.retrieval_bank_max_samples_per_expert,
        )

    if retrieval_cache_dir is not None:
        retrieval_train_path = expert_split_retrieval_cache_path(retrieval_cache_dir, expert_id, "train")
        retrieval_val_path = expert_split_retrieval_cache_path(retrieval_cache_dir, expert_id, "val")
        retrieval_train_meta = retrieval_result_cache_metadata(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            context_history_len=args.context_history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.window_stride,
            prompt_style=args.prompt_style,
            variant=variant_name,
            expert_id=expert_id,
            split="train",
            top_k=(0 if retriever is None else retriever.top_k),
            use_rag=args.use_rag,
            include_env_diff=args.use_diff_dora,
            max_samples_per_expert=args.max_samples_per_expert,
            retrieval_bank_max_samples_per_expert=args.retrieval_bank_max_samples_per_expert,
        )
        retrieval_val_meta = retrieval_result_cache_metadata(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            context_history_len=args.context_history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.window_stride,
            prompt_style=args.prompt_style,
            variant=variant_name,
            expert_id=expert_id,
            split="val",
            top_k=(0 if retriever is None else retriever.top_k),
            use_rag=args.use_rag,
            include_env_diff=args.use_diff_dora,
            max_samples_per_expert=args.max_samples_per_expert,
            retrieval_bank_max_samples_per_expert=args.retrieval_bank_max_samples_per_expert,
        )

        if (
            tokenized_cache_dir is not None
            and train_cache_path is not None
            and val_cache_path is not None
            and train_cache_meta is not None
            and val_cache_meta is not None
            and not args.rebuild_tokenized_cache
            and can_use_tokenized_items_cache(train_cache_path, expected_metadata=train_cache_meta)
            and can_use_tokenized_items_cache(val_cache_path, expected_metadata=val_cache_meta)
        ):
            train_items = load_tokenized_items_cache(train_cache_path, expected_metadata=train_cache_meta)
            val_items = load_tokenized_items_cache(val_cache_path, expected_metadata=val_cache_meta)
            print(
                f"\n=== Training Expert {expert_id} "
                f"(cached train={len(train_items)}, val={len(val_items)}) ==="
            )
            train_ds = TokenizedItemsDataset(train_items)
            val_ds = TokenizedItemsDataset(val_items)

    if train_ds is None or val_ds is None:
        if samples_with_node is None:
            raise ValueError("samples_with_node is required when tokenized caches are unavailable.")
        print(f"\n=== Training Expert {expert_id} ({len(samples_with_node)} samples) ===")
    base_model, tokenizer = load_model_and_tokenizer()
    peft_model = get_lora_model(base_model, use_dora=args.use_dora,
                                r=args.lora_rank, lora_alpha=args.lora_alpha)
    use_gc = args.gradient_checkpointing
    if use_gc and hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()
    elif hasattr(peft_model, "gradient_checkpointing_disable"):
        peft_model.gradient_checkpointing_disable()

    if train_ds is None or val_ds is None:
        train_samples, val_samples = _split_train_val_samples(samples_with_node)
        print(f"  Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

        train_retrieval_entries = {}
        val_retrieval_entries = {}
        if (
            retrieval_cache_dir is not None
            and retrieval_train_path is not None
            and retrieval_val_path is not None
            and not args.rebuild_tokenized_cache
        ):
            retrieval_train_meta = {
                **retrieval_train_meta,
                "sample_count": len(train_samples),
            }
            retrieval_val_meta = {
                **retrieval_val_meta,
                "sample_count": len(val_samples),
            }
            if can_use_retrieval_result_cache(retrieval_train_path, expected_metadata=retrieval_train_meta):
                train_retrieval_entries = load_retrieval_result_cache(
                    retrieval_train_path,
                    expected_metadata=retrieval_train_meta,
                )
                print(f"[retrieval_cache] Loaded {retrieval_train_path} ({len(train_retrieval_entries)} entries)")
            if can_use_retrieval_result_cache(retrieval_val_path, expected_metadata=retrieval_val_meta):
                val_retrieval_entries = load_retrieval_result_cache(
                    retrieval_val_path,
                    expected_metadata=retrieval_val_meta,
                )
                print(f"[retrieval_cache] Loaded {retrieval_val_path} ({len(val_retrieval_entries)} entries)")

        train_ds = EVDataset(
            train_samples,
            tokenizer,
            args.horizon,
            max_length=args.max_length,
            node_idx=0,
            use_rag=args.use_rag,
            retriever=retriever,
            weather=weather,
            price=price,
            node_meta=node_meta,
            node_ids=node_ids,
            poi=poi,
            include_env_diff=args.use_diff_dora,
            prompt_style=args.prompt_style,
            retrieval_cache_entries=train_retrieval_entries,
        )
        
        val_ds = EVDataset(
            val_samples,
            tokenizer,
            args.horizon,
            max_length=args.max_length,
            node_idx=0,
            use_rag=args.use_rag,
            retriever=retriever,
            weather=weather,
            price=price,
            node_meta=node_meta,
            node_ids=node_ids,
            poi=poi,
            include_env_diff=args.use_diff_dora,
            prompt_style=args.prompt_style,
            retrieval_cache_entries=val_retrieval_entries,
        )
        if retrieval_cache_dir is not None and retrieval_train_path is not None and retrieval_val_path is not None:
            save_retrieval_result_cache(train_retrieval_entries, retrieval_train_path, metadata=retrieval_train_meta)
            save_retrieval_result_cache(val_retrieval_entries, retrieval_val_path, metadata=retrieval_val_meta)
        if tokenized_cache_dir is not None and train_cache_path is not None and val_cache_path is not None:
            save_tokenized_items_cache(
                train_ds.items,
                train_cache_path,
                metadata={**train_cache_meta, "sample_count": len(train_samples)},
            )
            save_tokenized_items_cache(
                val_ds.items,
                val_cache_path,
                metadata={**val_cache_meta, "sample_count": len(val_samples)},
            )

    collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model, padding=True, pad_to_multiple_of=8)
    adapter_dir = out_dir / f"expert_{expert_id}" / "adapter"
    best_snapshot_cb = BestExpertSnapshotCallback(
        peft_model=peft_model,
        tokenizer=tokenizer,
        adapter_dir=adapter_dir,
    )
    training_args = TrainingArguments(
        output_dir=str(out_dir / f"expert_{expert_id}" / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=20,
        eval_strategy="no" if args.eval_steps <= 0 else "steps",
        eval_steps=max(1, args.eval_steps),
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[best_snapshot_cb],
    )
    trainer.label_names = ["labels"]
    trainer.train()
    best_snapshot_cb.ensure_snapshot(trainer.state, reason="post_train_fallback")
    save_path = str(adapter_dir)
    if best_snapshot_cb.best_metric is not None:
        print(f"  Best eval_loss: {best_snapshot_cb.best_metric:.5f}")
    print(f"Expert {expert_id} saved → {save_path}")
    del trainer
    del collator
    del train_ds
    del peft_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return save_path


def evaluate_one_expert(
    expert_id: int,
    adapter_path: str,
    tokenizer,
    test_samples: list[dict],
    router: HardRouter,
    splits: dict,
    args,
    retriever: KNNRetriever | None = None,
):
    print(f"\n=== Evaluating Expert {expert_id} ===")
    base_model, _ = load_model_and_tokenizer()
    model = load_peft_model(base_model, adapter_path)
    _warn_legacy_controller_checkpoint(Path(adapter_path).parent / "diff_controller.pt")
    eval_model = model
    eval_model.eval()
    model.eval()

    from tqdm import tqdm
    nodes = router.nodes_for_expert(expert_id)
    preds, trues = [], []
    eval_cap = min(args.eval_max_samples, 50)
    n_nodes = min(args.eval_nodes_per_expert, len(nodes))
    total_calls = eval_cap * n_nodes
    print(f"  Inference: {eval_cap} samples × {n_nodes} nodes = {total_calls} calls")
    domain_name = "CBD" if expert_id == 0 else "Residential"
    bar = tqdm(total=total_calls, desc=f"  Expert {expert_id} ({domain_name})",
               unit="call", ncols=90, leave=True)
    for sample in test_samples[:eval_cap]:
        for node_idx in nodes[:n_nodes]:
            sys_msg, usr_msg = _build_eval_prompt(
                sample,
                node_idx,
                splits,
                args,
                retriever,
                domain_name,
            )
            out = generate(eval_model, tokenizer, sys_msg, usr_msg, max_new_tokens=256)
            arr = parse_output(out, args.horizon)
            ok = arr is not None and len(arr) == args.horizon
            if ok:
                preds.append(arr)
                trues.append(sample["y"][:args.horizon, node_idx])
            bar.update(1)
            bar.set_postfix(parsed=len(preds), node=node_idx, ok="✓" if ok else "✗", refresh=False)
    bar.close()

    del eval_model
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if not preds:
        return None
    return per_horizon_metrics(
        preds,
        trues,
        args.horizon,
        splits["norm_min"],
        splits["norm_max"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="st_evcdp", choices=DATASET_CHOICES)
    parser.add_argument("--horizon",    type=int,   default=6)
    parser.add_argument("--output_dir", default="outputs/moe_experts_h6")
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora_rank",  type=int, default=32,
                        help="LoRA rank; the paper uses 32.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor); the paper uses 32.")
    parser.add_argument("--history_len", type=int, default=12,
                        help="Historical observation window; the paper uses 12.")
    parser.add_argument("--context_history_len", type=int, default=0,
                        help="Optional long-context history window. 0 means use history_len only.")
    parser.add_argument("--neighbor_k", type=int, default=7,
                        help="Neighbour top-k used for spatial context; the paper uses 7.")
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help="Window step size. `0` means use `horizon` (non-overlapping targets); `1` keeps classic overlapping sliding windows.",
    )
    parser.add_argument("--use_dora",   action="store_true")
    parser.add_argument("--use_diff_dora", action="store_true",
                        help="Enable paper-style Diff-DoRA prompt conditioning (requires DoRA + RAG)")
    parser.add_argument("--use_rag",    action="store_true",
                        help="Enable retrieval-augmented CoT prompts for expert training/eval")
    parser.add_argument("--prompt_style", choices=sorted(EVDataset.PROMPT_STYLES), default="auto",
                        help="Prompt supervision style: auto preserves current behavior; direct_physical enables w/o-CoT training.")
    parser.add_argument("--max_length", type=int, default=2560,
                        help="Tokenization max length; the paper uses 2560.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=True,
                        help="Enable gradient checkpointing to reduce VRAM (default: enabled)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false",
                        help="Disable gradient checkpointing")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/{dataset}_h{horizon}.pkl")
    parser.add_argument("--retrieval_device", default="cpu",
                        help="Retrieval query backend: cpu, auto, or a CUDA device like cuda:0.")
    parser.add_argument("--retrieval_query_batch_size", type=int, default=128,
                        help="Number of query samples to score per retrieval batch when GPU retrieval is enabled.")
    parser.add_argument("--retrieval_corpus_chunk_size", type=int, default=65536,
                        help="Number of retrieval-bank vectors to compare per GPU retrieval chunk.")
    parser.add_argument("--sample_cache", default="",
                        help="Path to pre-built expert sample cache pkl; defaults to data/sample_cache/train_experts_{dataset}_h{horizon}_hist{history_len}_nbr{neighbor_k}.pkl")
    parser.add_argument("--rebuild_sample_cache", action="store_true",
                        help="Force rebuilding the sample cache even if it already exists.")
    parser.add_argument("--tokenized_cache_dir", default="",
                        help="Directory containing pre-built expert_{id}_{train|val}.pkl tokenized caches.")
    parser.add_argument("--rebuild_tokenized_cache", action="store_true",
                        help="Force rebuilding tokenized train/val caches even if they already exist.")
    parser.add_argument("--retrieval_results_cache_dir", default="",
                        help="Directory containing expert_{id}_{train|val}.pkl retrieval-result caches.")
    parser.add_argument("--max_samples_per_expert", type=int, default=1000,
                        help="Per-expert training sample cap; <=0 uses the full expert sample pool.")
    parser.add_argument("--retrieval_bank_max_samples_per_expert", type=int, default=800,
                        help="Per-expert retrieval-bank size cap when rebuilding expert-local KNN pools; <=0 uses the full expert pool.")
    parser.add_argument("--eval_max_samples", type=int, default=3,
                        help="Number of test samples per expert after training (default 3)")
    parser.add_argument("--eval_nodes_per_expert", type=int, default=5,
                        help="Number of nodes per expert after training (default 5)")
    parser.add_argument("--eval_steps", type=int, default=25,
                        help="Validation interval during training; use a very large value to effectively disable mid-train eval.")
    args = parser.parse_args()

    if args.use_diff_dora and not args.use_dora:
        raise ValueError("--use_diff_dora requires --use_dora")
    if args.use_diff_dora and not args.use_rag:
        raise ValueError("--use_diff_dora requires --use_rag")
    if args.use_diff_dora and args.prompt_style == "vanilla":
        raise ValueError("--use_diff_dora is invalid with --prompt_style vanilla")

    if args.prompt_style in {"cot", "direct_physical"} and not args.use_rag:
        raise ValueError("--prompt_style cot/direct_physical requires --use_rag")

    args.window_stride = resolve_window_stride(args.window_stride, horizon=args.horizon)
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_name = "full" if args.use_diff_dora else "wo_diffdora"
    tokenized_cache_dir = Path(args.tokenized_cache_dir) if args.tokenized_cache_dir else None
    retrieval_cache_dir = (
        Path(args.retrieval_results_cache_dir)
        if args.retrieval_results_cache_dir
        else default_expert_retrieval_cache_dir(
            args.dataset,
            args.horizon,
            args.history_len,
            args.neighbor_k,
            args.window_stride,
            args.prompt_style,
            variant_name,
            args.max_samples_per_expert,
            args.retrieval_bank_max_samples_per_expert,
            context_history_len=args.context_history_len,
        )
    )

    # 1. Data
    raw    = load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    include_test_samples = args.eval_max_samples > 0 and args.eval_nodes_per_expert > 0
    sample_cache_path = (
        Path(args.sample_cache)
        if args.sample_cache
        else default_expert_sample_cache_path(
            args.dataset,
            args.horizon,
            args.history_len,
            args.neighbor_k,
            args.window_stride,
            context_history_len=args.context_history_len,
            include_test=include_test_samples,
        )
    )
    sample_cache, resolved_sample_cache_path = load_or_build_expert_sample_cache(
        splits=splits,
        dataset=args.dataset,
        horizon=args.horizon,
        history_len=args.history_len,
        context_history_len=args.context_history_len,
        neighbor_k=args.neighbor_k,
        window_stride=args.window_stride,
        cache_path=sample_cache_path,
        include_test=include_test_samples,
        force_rebuild=args.rebuild_sample_cache,
    )

    all_tokenized_cache_ready = False
    if (
        tokenized_cache_dir is not None
        and not args.rebuild_tokenized_cache
        and args.eval_max_samples <= 0
        and args.eval_nodes_per_expert <= 0
    ):
        expected_common = {
            "version": 2,
            "kind": "train_experts_tokenized",
            "dataset": args.dataset,
            "horizon": int(args.horizon),
            "history_len": int(args.history_len),
            "context_history_len": int(args.context_history_len or args.history_len),
            "neighbor_k": int(args.neighbor_k),
            "window_stride": int(args.window_stride),
            "max_length": int(args.max_length),
            "prompt_style": args.prompt_style,
            "variant": variant_name,
            "use_rag": bool(args.use_rag),
            "include_env_diff": bool(args.use_diff_dora),
            "max_samples_per_expert": int(args.max_samples_per_expert),
            "retrieval_bank_max_samples_per_expert": int(args.retrieval_bank_max_samples_per_expert),
        }
        all_tokenized_cache_ready = True
        for expert_id in (0, 1):
            for split in ("train", "val"):
                cache_path = expert_split_tokenized_cache_path(tokenized_cache_dir, expert_id, split)
                expected_meta = {
                    **expected_common,
                    "expert_id": expert_id,
                    "split": split,
                }
                if not can_use_tokenized_items_cache(cache_path, expected_metadata=expected_meta):
                    all_tokenized_cache_ready = False
                    break
            if not all_tokenized_cache_ready:
                break
        if all_tokenized_cache_ready:
            print(f"Using pretokenized caches from {tokenized_cache_dir}; skipping expert expansion and retriever build.")

    # 2. Routing labels
    labels = build_routing_labels(splits["train"], raw.get("node_meta"))
    router = HardRouter(labels)
    N = splits["train"].shape[1]

    tagged: dict[int, list] = {0: [], 1: []}
    raw_samples = sample_cache["train_samples"]
    need_tagged_samples = (not all_tokenized_cache_ready) or (args.eval_max_samples > 0 and args.eval_nodes_per_expert > 0)
    if need_tagged_samples:
        print("Expanding per-node samples …")
        for s in raw_samples:
            for n in range(N):
                eid = router.route(n)
                if _is_within_sample_cap(len(tagged[eid]), args.max_samples_per_expert):
                    s_node = dict(s, node_idx=n)
                    tagged[eid].append(s_node)
        print(f"Expert sample counts: expert_0={len(tagged[0])}, expert_1={len(tagged[1])}")

    tokenizer = load_tokenizer() if args.eval_max_samples > 0 and args.eval_nodes_per_expert > 0 else None

    # Build per-expert retrieval banks for isolation (Issue 2 fix)
    retrievers = {0: None, 1: None}
    need_retrievers = args.use_rag and (not all_tokenized_cache_ready or (args.eval_max_samples > 0 and args.eval_nodes_per_expert > 0))
    if need_retrievers:
        try:
            if args.retrieval_cache:
                cache_path = Path(args.retrieval_cache)
            else:
                cache_path = default_retrieval_cache_path(args.dataset, args.horizon, args.window_stride)
            
            if cache_path.exists():
                global_retriever = KNNRetriever.load(cache_path)
                print(f"Loaded retrieval cache: {cache_path}")
                # Create per-expert retrievers by filtering the global pool
                for eid in (0, 1):
                    if args.retrieval_bank_max_samples_per_expert > 0:
                        expert_samples = tagged[eid][:args.retrieval_bank_max_samples_per_expert]
                    else:
                        expert_samples = tagged[eid]
                    retrievers[eid] = KNNRetriever(
                        expert_samples,
                        top_k=2,
                        query_device=args.retrieval_device,
                        query_batch_size=args.retrieval_query_batch_size,
                        corpus_chunk_size=args.retrieval_corpus_chunk_size,
                    )
                    print(
                        f"Built per-expert retriever for expert_{eid} with {len(expert_samples)} reference samples "
                        f"(query_device={retrievers[eid].resolved_query_device})"
                    )
            else:
                # Build from per-expert tagged samples
                print(f"Retrieval cache not found, building per-expert in-memory retrievers.")
                for eid in (0, 1):
                    retrievers[eid] = KNNRetriever(
                        tagged[eid],
                        top_k=2,
                        query_device=args.retrieval_device,
                        query_batch_size=args.retrieval_query_batch_size,
                        corpus_chunk_size=args.retrieval_corpus_chunk_size,
                    )
                    print(
                        f"Built per-expert retriever for expert_{eid} with {len(tagged[eid])} reference samples "
                        f"(query_device={retrievers[eid].resolved_query_device})"
                    )
        except Exception as e:
            print(f"Warning: Failed to build per-expert retrievers, falling back to shared retriever: {e}")
            shared_retriever = KNNRetriever(
                raw_samples,
                top_k=2,
                query_device=args.retrieval_device,
                query_batch_size=args.retrieval_query_batch_size,
                corpus_chunk_size=args.retrieval_corpus_chunk_size,
            )
            retrievers = {0: shared_retriever, 1: shared_retriever}

    # 5. Train each expert (with per-expert retriever isolation)
    trained = {}
    for eid in (0, 1):
        trained[eid] = train_one_expert(
            eid,
            tagged[eid] if need_tagged_samples else None,
            out_dir,
            args,
            retriever=retrievers[eid],  # Pass per-expert retriever
            weather=splits.get("weather"),
            price=splits.get("price"),
            node_meta=splits.get("node_meta"),
            node_ids=splits.get("node_ids"),
            poi=splits.get("poi"),
            tokenized_cache_dir=tokenized_cache_dir,
            retrieval_cache_dir=retrieval_cache_dir,
            variant_name=variant_name,
        )

    # 6. Evaluate both experts on test split
    results = {}
    if args.eval_max_samples > 0 and args.eval_nodes_per_expert > 0:
        test_samples = sample_cache.get("test_samples")
        if test_samples is None:
            print("Building test samples …")
            test_map = build_samples(
                splits["test"], splits["timestamps_test"],
                adj=splits.get("adj"),
                horizons=[args.horizon],
                history_len=args.history_len,
                context_history_len=args.context_history_len,
                neighbor_k=args.neighbor_k,
                window_stride=args.window_stride,
            )
            test_samples = test_map[args.horizon]
        for eid in (0, 1):
            m = evaluate_one_expert(
                eid,
                trained[eid],
                tokenizer,
                test_samples,
                router,
                splits,
                args,
                retriever=retrievers[eid],
            )
            if m is not None:
                results[f"expert_{eid}"] = m
                print(f"Expert {eid}: {m['overall']}")
    else:
        print("Skipping post-training quick evaluation (eval_max_samples<=0 or eval_nodes_per_expert<=0).")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "run_id": out_dir.name,
            "dataset": args.dataset,
            "horizon": args.horizon,
            "results": results,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_length": args.max_length,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "history_len": args.history_len,
                "context_history_len": args.context_history_len,
                "neighbor_k": args.neighbor_k,
                "window_stride": args.window_stride,
                "use_dora": args.use_dora,
                "use_diff_dora": args.use_diff_dora,
                "diff_dora_impl": "prompt_only" if args.use_diff_dora else None,
                "use_rag": args.use_rag,
                "prompt_style": args.prompt_style,
                "max_samples_per_expert": args.max_samples_per_expert,
                "retrieval_bank_max_samples_per_expert": args.retrieval_bank_max_samples_per_expert,
                "eval_steps": args.eval_steps,
                "sample_cache": str(resolved_sample_cache_path),
                "tokenized_cache_dir": str(tokenized_cache_dir) if tokenized_cache_dir is not None else None,
                "retrieval_results_cache_dir": str(retrieval_cache_dir),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"Saved to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
