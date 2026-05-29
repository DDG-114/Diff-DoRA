"""
Train a single shared adapter on all selected nodes of a dataset.

This is useful for source-domain pretraining before cross-dataset transfer,
especially when the source dataset has only a few semantically similar nodes
(for example `renewable_solar`).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.loaders import DATASET_CHOICES, load_dataset
from src.data.windowing import default_retrieval_cache_path, resolve_window_stride
from src.eval.shot_utils import (
    build_tagged_node_samples,
    build_tagged_node_samples_shuffled_pairs,
    evaluate_shared_adapter,
    make_retriever,
    train_shared_adapter,
)
from src.utils.price_candidate import attach_candidate_curve, attach_candidate_refine_mask, load_daylevel_prediction_map
from src.utils.active_selection import select_active_source_items, select_price_dynamic_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="renewable_solar", choices=DATASET_CHOICES)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--output_dir", default="outputs/shared_adapter")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2560)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument(
        "--context_history_len",
        type=int,
        default=0,
        help="Optional long-range history length. `0` disables extra context and keeps classic single-window samples.",
    )
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--window_stride", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--use_diff_dora", action="store_true")
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--prompt_style", choices=["auto", "cot", "direct_physical", "vanilla"], default="direct_physical")
    parser.add_argument(
        "--target_style",
        choices=["auto", "numeric_only", "candidate_residual", "candidate_selective_residual", "candidate_chunk_offset"],
        default="auto",
        help="Training target format. numeric_only keeps the prompt style but supervises only the final numeric list.",
    )
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--max_eval", type=int, default=0)
    parser.add_argument("--max_nodes", type=int, default=0)
    parser.add_argument("--node_sampling", choices=["head", "random"], default="head")
    parser.add_argument("--sampling", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--item_sampling", choices=["legacy_order", "shuffled_pairs"], default="shuffled_pairs")
    parser.add_argument(
        "--active_selection",
        choices=["none", "source", "price_dynamic"],
        default="none",
        help="Enable heuristic active sample selection before training.",
    )
    parser.add_argument(
        "--active_budget_ratio",
        type=float,
        default=0.5,
        help="Fraction of candidate source items to keep when --active_selection source is enabled.",
    )
    parser.add_argument("--retrieval_cache", default="")
    parser.add_argument(
        "--retrieval_device",
        default="auto",
        help="Retrieval query backend: cpu, auto, or a CUDA device like cuda:0.",
    )
    parser.add_argument("--retrieval_query_batch_size", type=int, default=128)
    parser.add_argument("--retrieval_corpus_chunk_size", type=int, default=65536)
    parser.add_argument(
        "--candidate_prediction_csv",
        default="",
        help="Optional day-level candidate prediction CSV with columns day,slot,prediction; used to inject skeleton forecasts into prompts.",
    )
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

    raw = load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    candidate_prediction_map = None
    if args.candidate_prediction_csv:
        candidate_prediction_map = load_daylevel_prediction_map(args.candidate_prediction_csv)
    train_raw = build_samples(
        splits["train"],
        splits["timestamps_train"],
        adj=splits.get("adj"),
        aux_features=splits.get("weather"),
        horizons=[args.horizon],
        history_len=args.history_len,
        context_history_len=args.context_history_len,
        neighbor_k=args.neighbor_k,
        window_stride=args.window_stride,
    )[args.horizon]
    test_raw = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        aux_features=splits.get("weather"),
        horizons=[args.horizon],
        history_len=args.history_len,
        context_history_len=args.context_history_len,
        neighbor_k=args.neighbor_k,
        window_stride=args.window_stride,
    )[args.horizon]
    if candidate_prediction_map is not None:
        train_raw = [
            attach_candidate_refine_mask(
                attach_candidate_curve(sample, horizon=args.horizon, prediction_map=candidate_prediction_map)
            )
            for sample in train_raw
        ]
        test_raw = [
            attach_candidate_refine_mask(
                attach_candidate_curve(sample, horizon=args.horizon, prediction_map=candidate_prediction_map)
            )
            for sample in test_raw
        ]

    all_nodes = list(range(splits["train"].shape[1]))
    if args.max_nodes > 0:
        all_nodes = all_nodes[: args.max_nodes]

    if args.active_selection == "source":
        tagged_train = select_active_source_items(
            train_raw,
            all_nodes,
            weather=splits.get("weather"),
            budget_ratio=args.active_budget_ratio,
            max_items=args.max_train_items,
        )
    elif args.active_selection == "price_dynamic":
        tagged_train = select_price_dynamic_items(
            train_raw,
            all_nodes,
            budget_ratio=args.active_budget_ratio,
            max_items=args.max_train_items,
        )
    else:
        if args.item_sampling == "shuffled_pairs":
            tagged_train = build_tagged_node_samples_shuffled_pairs(
                train_raw,
                all_nodes,
                seed=args.seed,
                max_items=args.max_train_items,
            )
        else:
            tagged_train = build_tagged_node_samples(
                train_raw,
                all_nodes,
                max_items=args.max_train_items,
            )

    retriever = None
    if args.use_rag:
        retriever = make_retriever(
            train_raw,
            use_rag=True,
            query_device=args.retrieval_device,
            query_batch_size=args.retrieval_query_batch_size,
            corpus_chunk_size=args.retrieval_corpus_chunk_size,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, adapter_dir, train_metrics = train_shared_adapter(
        train_items=tagged_train,
        splits=splits,
        horizon=args.horizon,
        out_dir=out_dir,
        args=args,
        retriever=retriever,
    )

    eval_result = None
    if args.max_eval != 0:
        eval_result = evaluate_shared_adapter(
            model=model,
            tokenizer=tokenizer,
            test_samples=test_raw,
            node_indices=all_nodes,
            splits=splits,
            horizon=args.horizon,
            args=args,
            retriever=retriever,
        )

    payload = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "adapter_dir": str(adapter_dir),
        "train_items": len(tagged_train),
        "source_nodes": all_nodes,
        "config": vars(args),
        "train_metrics": train_metrics,
        "eval_result": eval_result,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))

    del model
    del tokenizer
    if retriever is not None:
        del retriever
    gc.collect()


if __name__ == "__main__":
    main()
