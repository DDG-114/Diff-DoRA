"""
src/eval/eval_fewshot.py
-------------------------
Few-shot forecasting on the same node set with temporally scarce training data.

Definition used here:
- test split is unchanged
- training uses only the first k% of train windows (time-prefix sampling)
- model still trains/evaluates on the same station set
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.load_st_evcdp import (
    FULL_DATA_NORMALIZATION,
    TRAIN_ONLY_NORMALIZATION,
    load_st_evcdp,
)
from src.data.load_urbanev import load_urbanev
from src.data.shot_manifests import (
    DEFAULT_FEWSHOT_RATIOS,
    DEFAULT_WINDOW_STRIDE,
    ensure_strict_fewshot_manifest,
    ratio_key,
    strict_processed_path,
)
from src.eval.shot_utils import (
    build_tagged_node_samples,
    build_tagged_node_samples_shuffled_pairs,
    cleanup_model,
    evaluate_shared_adapter,
    load_retriever,
    make_retriever,
    parse_ratio_csv,
    select_node_subset,
    train_shared_adapter,
)

DEFAULT_RATIOS = "0.05,0.10,0.20,0.40,1.00"


def _load_dataset(dataset: str, *, normalization_source: str, horizon: int) -> dict:
    if dataset == "st_evcdp":
        processed_path = None
        if normalization_source == TRAIN_ONLY_NORMALIZATION:
            processed_path = strict_processed_path(dataset, horizon)
        return load_st_evcdp(
            processed_path=processed_path,
            normalization_source=normalization_source,
        )
    if normalization_source != FULL_DATA_NORMALIZATION:
        raise ValueError("train_only normalization is currently supported only for st_evcdp.")
    return load_urbanev()


def _effective_eval_args(args, *, strict_protocol: bool):
    if not strict_protocol:
        return args
    data = vars(args).copy()
    data["max_eval"] = 0
    data["sampling"] = "head"
    return argparse.Namespace(**data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--output_dir", default="outputs/fewshot")
    parser.add_argument(
        "--fewshot_ratios",
        default=DEFAULT_RATIOS,
        help="Comma-separated train-window prefix ratios, e.g. 0.05,0.1,0.2",
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2560)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")

    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--use_diff_dora", action="store_true")
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument(
        "--prompt_style",
        choices=sorted({"auto", "cot", "direct_physical", "vanilla"}),
        default="auto",
    )

    parser.add_argument(
        "--max_train_items",
        type=int,
        default=4000,
        help="Training item cap after node expansion; <=0 uses all items.",
    )
    parser.add_argument(
        "--max_eval",
        type=int,
        default=0,
        help="Evaluation sample count; <=0 uses the full test split.",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=0,
        help="Evaluation/training node count; <=0 uses all nodes.",
    )
    parser.add_argument("--node_sampling", choices=["head", "random"], default="head")
    parser.add_argument("--sampling", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument(
        "--normalization_source",
        choices=[FULL_DATA_NORMALIZATION, TRAIN_ONLY_NORMALIZATION],
        default=FULL_DATA_NORMALIZATION,
    )
    parser.add_argument(
        "--item_sampling",
        choices=["legacy_order", "shuffled_pairs"],
        default="legacy_order",
    )
    parser.add_argument("--strict_protocol", action="store_true")
    args = parser.parse_args()

    if args.use_diff_dora and not args.use_dora:
        raise ValueError("--use_diff_dora requires --use_dora")
    if args.use_diff_dora and not args.use_rag:
        raise ValueError("--use_diff_dora requires --use_rag")
    if args.use_diff_dora and args.prompt_style == "vanilla":
        raise ValueError("--use_diff_dora is invalid with --prompt_style vanilla")
    if args.prompt_style in {"cot", "direct_physical"} and not args.use_rag:
        raise ValueError("--prompt_style cot/direct_physical requires --use_rag")

    if args.strict_protocol:
        if args.window_stride == 1:
            args.window_stride = DEFAULT_WINDOW_STRIDE
        if args.normalization_source == FULL_DATA_NORMALIZATION:
            args.normalization_source = TRAIN_ONLY_NORMALIZATION
        if args.item_sampling == "legacy_order":
            args.item_sampling = "shuffled_pairs"
        if args.dataset != "st_evcdp":
            raise ValueError("--strict_protocol currently supports only st_evcdp.")
        if args.horizon != 6:
            raise ValueError("--strict_protocol currently targets horizon=6.")
        if args.window_stride != DEFAULT_WINDOW_STRIDE:
            raise ValueError(
                f"--strict_protocol requires --window_stride {DEFAULT_WINDOW_STRIDE} "
                f"(got {args.window_stride})."
            )
        if args.normalization_source != TRAIN_ONLY_NORMALIZATION:
            raise ValueError("--strict_protocol requires --normalization_source train_only.")
        if args.item_sampling != "shuffled_pairs":
            raise ValueError("--strict_protocol requires --item_sampling shuffled_pairs.")
        if args.max_eval > 0 or args.max_nodes > 0:
            raise ValueError("--strict_protocol requires full evaluation; leave --max_eval/--max_nodes at 0.")

    raw = _load_dataset(
        args.dataset,
        normalization_source=args.normalization_source,
        horizon=args.horizon,
    )
    splits = build_splits(raw, args.dataset)
    train_raw = build_samples(
        splits["train"],
        splits["timestamps_train"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
        window_stride=args.window_stride,
    )[args.horizon]
    test_raw = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
        window_stride=args.window_stride,
    )[args.horizon]

    manifest = None
    manifest_path = None
    ratios = parse_ratio_csv(args.fewshot_ratios)
    if args.strict_protocol:
        manifest, manifest_path = ensure_strict_fewshot_manifest(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.window_stride,
            seed=args.seed,
            normalization_source=args.normalization_source,
            max_train_items=args.max_train_items,
            ratios=ratios or [float(v) for v in DEFAULT_FEWSHOT_RATIOS],
            manifest_path=args.manifest_path or None,
            force_rebuild=False,
        )

    all_nodes = list(range(splits["train"].shape[1]))
    node_indices = (
        list(all_nodes)
        if args.strict_protocol
        else select_node_subset(all_nodes, args.max_nodes, args.node_sampling, args.seed)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    eval_args = _effective_eval_args(args, strict_protocol=args.strict_protocol)

    for ratio in ratios:
        ratio_dir = out_dir / f"ratio_{ratio:.2f}"
        ratio_manifest = None
        if manifest is not None:
            ratio_manifest = manifest["ratios"][ratio_key(ratio)]
            selected_train_raw = [train_raw[idx] for idx in ratio_manifest["window_indices"]]
            retriever = load_retriever(
                ratio_manifest["retrieval_cache_path"],
                use_rag=args.use_rag,
            )
            tagged_train = build_tagged_node_samples_shuffled_pairs(
                selected_train_raw,
                node_indices,
                seed=args.seed,
                max_items=args.max_train_items,
            )
            train_window_count = ratio_manifest["train_window_count"]
        else:
            n_windows = max(1, int(len(train_raw) * ratio))
            selected_train_raw = train_raw[:n_windows]
            train_window_count = n_windows
            retriever = make_retriever(selected_train_raw, use_rag=args.use_rag)
            tagged_train = build_tagged_node_samples(
                selected_train_raw,
                node_indices,
                max_items=args.max_train_items,
            )

        print(
            f"\n=== Few-shot ratio={ratio:.2f} | train_windows={train_window_count} "
            f"| train_items={len(tagged_train)} ==="
        )
        model, tokenizer, adapter_dir, train_metrics = train_shared_adapter(
            train_items=tagged_train,
            splits=splits,
            horizon=args.horizon,
            out_dir=ratio_dir,
            args=args,
            retriever=retriever,
        )
        eval_result = evaluate_shared_adapter(
            model=model,
            tokenizer=tokenizer,
            test_samples=test_raw,
            node_indices=node_indices,
            splits=splits,
            horizon=args.horizon,
            args=eval_args,
            retriever=retriever,
        )
        cleanup_model(model, tokenizer, retriever)
        gc.collect()

        result_payload = {
            "train_window_count": int(train_window_count),
            "train_item_count": int(len(tagged_train)),
            "node_count": int(len(node_indices)),
            "adapter_dir": str(adapter_dir),
            "train_metrics": train_metrics,
            "window_stride": int(args.window_stride),
            "normalization_source": args.normalization_source,
            "item_sampling": args.item_sampling,
            **eval_result,
        }
        if ratio_manifest is not None:
            result_payload["retrieval_cache_path"] = ratio_manifest["retrieval_cache_path"]
            result_payload["manifest_hash"] = manifest["manifest_hash"]
            result_payload["manifest_path"] = str(manifest_path)
        results[str(ratio)] = result_payload

        metrics = eval_result.get("metrics")
        if metrics:
            print(
                f"ratio={ratio:.2f} "
                f"RMSE={metrics['overall']['rmse']:.4f} "
                f"MAE={metrics['overall']['mae']:.4f} "
                f"parse={eval_result['parse_success_rate']:.3f}"
            )

    payload = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "fewshot_ratios": ratios,
        "use_dora": args.use_dora,
        "use_diff_dora": args.use_diff_dora,
        "use_rag": args.use_rag,
        "prompt_style": args.prompt_style,
        "node_count": len(node_indices),
        "node_indices": node_indices,
        "max_train_items": args.max_train_items,
        "max_eval": eval_args.max_eval,
        "sampling": eval_args.sampling,
        "seed": args.seed,
        "window_stride": args.window_stride,
        "normalization_source": args.normalization_source,
        "item_sampling": args.item_sampling,
        "strict_protocol": args.strict_protocol,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    if manifest is not None:
        payload["manifest_hash"] = manifest["manifest_hash"]
        payload["manifest_path"] = str(manifest_path)
    with open(out_dir / "fewshot_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out_dir}/fewshot_results.json")


if __name__ == "__main__":
    main()
