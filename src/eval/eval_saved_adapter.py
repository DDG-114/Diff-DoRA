"""
src/eval/eval_saved_adapter.py
-----------------------------
Evaluate a saved PEFT adapter directly on a chosen dataset/horizon.

Usage:
  python -m src.eval.eval_saved_adapter \
      --dataset st_evcdp \
      --horizon 6 \
      --adapter_dir outputs/single_dora_rag_h6/adapter \
      --use_rag \
      --output outputs/single_dora_rag_h6/eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src.data.loaders import DATASET_CHOICES, load_dataset
from src.data.build_splits import build_splits
from src.data.build_samples import build_samples
from src.data.windowing import default_retrieval_cache_path, resolve_window_stride
from src.utils.history_window import price_at_history_end, weather_at_history_end
from src.utils.node_context import extract_node_static_context, normalise_domain_label
from src.eval.metrics import per_horizon_metrics
from src.models.qwen_peft import load_model_and_tokenizer, load_peft_model, generate
from src.prompts.prompt_vanilla import build_direct_physical_prompt, build_vanilla_prompt
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.parser import parse_output
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.knn_retriever import KNNRetriever
from src.utils.price_candidate import attach_candidate_curve, attach_candidate_refine_mask, load_daylevel_prediction_map
from src.utils.price_candidate import combine_candidate_prediction
def _compute_retrieval_diff(sample: dict, retrieved: list[dict], splits: dict, node_idx: int) -> dict:
    weather_current = weather_at_history_end(splits.get("weather"), sample)
    weather_retrieved = [weather_at_history_end(splits.get("weather"), rs) for rs in retrieved]
    price_current = price_at_history_end(
        splits.get("price"),
        sample,
        node_idx,
        node_ids=splits.get("node_ids"),
        node_meta=splits.get("node_meta"),
    )
    price_retrieved = [
        price_at_history_end(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="st_evcdp", choices=DATASET_CHOICES)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--adapter_dir", required=True,
                        help="Path to saved adapter directory")
    parser.add_argument("--node_idx", type=int, default=0)
    parser.add_argument("--node_id", default="")
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--context_history_len", type=int, default=0,
                        help="Optional long-context history window for dual-history samples")
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--window_stride", type=int, default=0)
    parser.add_argument("--prompt_style", choices=["cot", "direct_physical", "vanilla"], default="direct_physical")
    parser.add_argument("--use_rag", action="store_true",
                        help="Enable retrieval-augmented evaluation")
    parser.add_argument("--use_diff_dora", action="store_true",
                        help="Inject paper-style environmental differentials into retrieval prompts")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/{dataset}_h{horizon}.pkl")
    parser.add_argument("--max_eval", type=int, default=200,
                        help="Maximum number of test samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=160,
                        help="Generation max_new_tokens during evaluation (CoT + output)")
    parser.add_argument("--sampling", choices=["head", "random"], default="head",
                        help="Evaluation sample selection: head=first N, random=random N")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation sampling when sampling=random")
    parser.add_argument("--save_generations", action="store_true", default=True,
                        help="Save raw model generations and parse status into output JSON")
    parser.add_argument("--candidate_prediction_csv", default="",
                        help="Optional candidate day-level prediction CSV to inject as a skeleton forecast into prompts.")
    parser.add_argument("--candidate_mode", choices=["absolute", "residual", "selective_residual"], default="absolute",
                        help="Whether the adapter outputs absolute prices or residuals relative to the candidate skeleton.")
    parser.add_argument("--candidate_residual_clip", type=float, default=0.0,
                        help="If >0 and candidate_mode=residual, clip the residual correction magnitude to this value.")
    parser.add_argument("--candidate_value_min", type=float, default=0.0)
    parser.add_argument("--candidate_value_max", type=float, default=1000.0)
    parser.add_argument("--output", default="outputs/eval_saved_adapter_metrics.json")
    args = parser.parse_args()

    if args.use_diff_dora and not args.use_rag:
        raise ValueError("--use_diff_dora requires --use_rag")
    if args.prompt_style in {"cot", "direct_physical"} and not args.use_rag:
        raise ValueError("--prompt_style cot/direct_physical requires --use_rag")
    if args.use_diff_dora and args.prompt_style == "vanilla":
        raise ValueError("--use_diff_dora is invalid with --prompt_style vanilla")

    args.window_stride = resolve_window_stride(args.window_stride, horizon=args.horizon)

    raw = load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    node_ids = [str(node_id) for node_id in splits.get("node_ids") or raw.get("node_ids") or []]
    if args.node_id:
        if args.node_id not in node_ids:
            raise ValueError(f"--node_id {args.node_id!r} not found in dataset node ids: {node_ids}")
        args.node_idx = node_ids.index(args.node_id)
    else:
        args.node_id = str(resolve_node_id(args.node_idx, node_ids=node_ids, node_meta=splits.get("node_meta")))

    test_map = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        aux_features=splits.get("weather"),
        horizons=[args.horizon],
        history_len=args.history_len,
        context_history_len=args.context_history_len,
        neighbor_k=args.neighbor_k,
        window_stride=args.window_stride,
    )
    test_samples = test_map[args.horizon]
    candidate_prediction_map = None
    if args.candidate_prediction_csv:
        candidate_prediction_map = load_daylevel_prediction_map(args.candidate_prediction_csv)
        test_samples = [
            attach_candidate_refine_mask(
                attach_candidate_curve(sample, horizon=args.horizon, prediction_map=candidate_prediction_map)
            )
            for sample in test_samples
        ]

    retriever = None
    if args.use_rag:
        if args.retrieval_cache:
            cache_path = Path(args.retrieval_cache)
        else:
            cache_path = default_retrieval_cache_path(args.dataset, args.horizon, args.window_stride)
        if cache_path.exists():
            retriever = KNNRetriever.load(cache_path)
            print(f"Loaded retrieval cache: {cache_path}")
        else:
            train_map = build_samples(
                splits["train"],
                splits["timestamps_train"],
                adj=splits.get("adj"),
                aux_features=splits.get("weather"),
                horizons=[args.horizon],
                history_len=args.history_len,
                context_history_len=args.context_history_len,
                neighbor_k=args.neighbor_k,
                window_stride=args.window_stride,
            )
            train_pool = train_map[args.horizon]
            print(
                f"Retrieval cache not found, building in-memory retriever from train pool: {len(train_pool)}"
            )
            retriever = KNNRetriever(train_pool, top_k=2)

    print("Loading base model …")
    base_model, tokenizer = load_model_and_tokenizer()
    print(f"Loading adapter … {args.adapter_dir}")
    model = load_peft_model(base_model, args.adapter_dir)
    ctrl_path = Path(args.adapter_dir).parent / "diff_controller.pt"
    if ctrl_path.exists():
        print(
            f"[WARN] Ignoring legacy Diff-DoRA controller checkpoint: {ctrl_path} "
            "(paper-style Diff-DoRA is prompt-only now)."
        )
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    print("Evaluating saved adapter …")
    preds, trues = [], []
    generation_records = []
    eval_limit = len(test_samples) if args.max_eval <= 0 else min(args.max_eval, len(test_samples))
    if args.sampling == "random":
        rng = random.Random(args.seed)
        subset = rng.sample(test_samples, eval_limit)
    else:
        subset = test_samples[:eval_limit]
    for idx, sample in enumerate(subset):
        sample_node_idx = int(sample.get("node_idx", args.node_idx))
        static_context = extract_node_static_context(
            sample_node_idx,
            node_ids=splits.get("node_ids"),
            node_meta=splits.get("node_meta"),
        )
        domain_label = normalise_domain_label(static_context.get("zone_type"))
        retrieved = []
        diff = None
        if args.use_rag and retriever is not None:
            retrieved = retriever.query(sample, exclude_t_start=None)
            diff = _compute_retrieval_diff(sample, retrieved, splits, sample_node_idx)

        if args.prompt_style == "cot" and retrieved:
            sys_msg, usr_msg = build_cot_prompt(
                sample,
                retrieved,
                diff,
                sample_node_idx,
                args.horizon,
                domain_label=domain_label,
                static_context=static_context,
                include_env_diff=args.use_diff_dora,
            )
        elif args.prompt_style == "direct_physical" and retrieved:
            sys_msg, usr_msg = build_direct_physical_prompt(
                sample,
                retrieved,
                diff,
                node_idx=sample_node_idx,
                horizon=args.horizon,
                domain_label=domain_label,
                static_context=static_context,
                include_env_diff=args.use_diff_dora,
                target_mode=args.candidate_mode,
            )
        else:
            sys_msg, usr_msg = build_vanilla_prompt(
                sample,
                sample_node_idx,
                args.horizon,
                domain_label=domain_label,
                static_context=static_context,
                target_mode=args.candidate_mode,
            )

        raw_output = generate(
            model,
            tokenizer,
            sys_msg,
            usr_msg,
            max_new_tokens=args.max_new_tokens,
        )
        parsed = parse_output(raw_output, expected_len=args.horizon)
        parse_ok = parsed is not None and len(parsed) == args.horizon
        if parse_ok:
            parsed = combine_candidate_prediction(
                sample,
                parsed,
                mode=args.candidate_mode,
                residual_clip=(args.candidate_residual_clip if args.candidate_residual_clip > 0 else None),
                value_clip=(args.candidate_value_min, args.candidate_value_max),
            )
            preds.append(parsed)
            trues.append(sample["y"][:args.horizon, sample_node_idx])

        if args.save_generations:
            generation_records.append({
                "sample_index": idx,
                "t_start": int(sample.get("t_start", -1)),
                "node_idx": sample_node_idx,
                "system_prompt": sys_msg,
                "user_prompt": usr_msg,
                "parse_ok": parse_ok,
                "raw_generation": raw_output,
                "parsed_prediction": parsed.tolist() if parse_ok else None,
                "target": sample["y"][:args.horizon, sample_node_idx].tolist(),
            })
        if (idx + 1) % 5 == 0 or (idx + 1) == len(subset):
            print(
                f"[eval_saved_adapter] {idx + 1}/{len(subset)} samples, "
                f"parsed={len(preds)}, failures={(idx + 1) - len(preds)}"
            )

    result = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "node_idx": args.node_idx,
        "node_id": args.node_id,
        "adapter_dir": args.adapter_dir,
        "use_rag": args.use_rag,
        "use_diff_dora": args.use_diff_dora,
        "diff_dora_impl": "prompt_only" if args.use_diff_dora else None,
        "history_len": args.history_len,
        "context_history_len": args.context_history_len,
        "neighbor_k": args.neighbor_k,
        "window_stride": args.window_stride,
        "prompt_style": args.prompt_style,
        "max_new_tokens": args.max_new_tokens,
        "evaluated_samples": len(preds),
        "requested_samples": eval_limit,
        "sampling": args.sampling,
        "seed": args.seed,
        "parse_failures": eval_limit - len(preds),
        "parse_success_rate": len(preds) / max(eval_limit, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": None,
        "generations": generation_records if args.save_generations else None,
    }

    if preds:
        metrics = per_horizon_metrics(
            preds,
            trues,
            args.horizon,
            splits["norm_min"],
            splits["norm_max"],
        )
        result["metrics"] = metrics
        print(json.dumps(metrics, indent=2))
    else:
        print("No parseable predictions.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
