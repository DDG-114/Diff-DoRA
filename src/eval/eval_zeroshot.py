"""
src/eval/eval_zeroshot.py
--------------------------
Zero-shot generalisation on unseen target nodes with hard-routing MoE training.

Definition used here:
- split nodes into source and target subsets
- train two routed experts only on source nodes
- evaluate only on target nodes, routed by the same hard router
- optionally mask target-node history during retrieval querying to avoid leakage
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import random
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
    DEFAULT_WINDOW_STRIDE,
    DEFAULT_ZEROSHOT_HALF_TRAIN_RATIOS,
    DEFAULT_ZEROSHOT_TEST_WINDOW_DIVISOR,
    ensure_strict_zeroshot_manifest,
    ratio_key,
    strict_processed_path,
)
from src.eval.eval_paper_ablation import _build_prompt
from src.eval.metrics import per_horizon_metrics
from src.eval.shot_utils import (
    build_tagged_node_samples,
    cleanup_model,
    make_retriever,
    mask_occ_and_adj,
    mask_samples_for_nodes,
    parse_ratio_csv,
    select_node_subset,
    select_sample_subset_with_indices,
)
from src.models.qwen_peft import generate_batch, load_model_and_tokenizer, load_peft_model
from src.prompts.parser import parse_output
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router import HardRouter
from src.train.train_experts import train_one_expert
from src.utils.node_context import extract_node_static_context, normalise_domain_label

DEFAULT_SOURCE_RATIOS = "0.20,0.40,0.60,0.80"


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


def _split_source_target_nodes(total_nodes: int, ratio: float, seed: int) -> tuple[list[int], list[int]]:
    all_nodes = list(range(total_nodes))
    n_source = max(1, int(total_nodes * ratio))
    if n_source >= total_nodes:
        n_source = total_nodes - 1
    rng = random.Random(seed + int(ratio * 1000))
    source_nodes = sorted(rng.sample(all_nodes, n_source))
    target_nodes = sorted([idx for idx in all_nodes if idx not in set(source_nodes)])
    return source_nodes, target_nodes


def _split_source_target_nodes_stratified(
    router: HardRouter,
    total_nodes: int,
    ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    by_expert = {
        0: router.nodes_for_expert(0),
        1: router.nodes_for_expert(1),
    }
    target_total = max(1, int(total_nodes * ratio))
    if target_total >= total_nodes:
        target_total = total_nodes - 1

    quotas = {}
    remainders = []
    assigned = 0
    for expert_id, nodes in by_expert.items():
        exact = len(nodes) * float(ratio)
        base = min(len(nodes), int(exact))
        quotas[expert_id] = base
        assigned += base
        remainders.append((exact - base, expert_id))

    remaining = max(0, target_total - assigned)
    for _, expert_id in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if quotas[expert_id] < len(by_expert[expert_id]):
            quotas[expert_id] += 1
            remaining -= 1

    source_nodes = []
    for expert_id, nodes in by_expert.items():
        rng = random.Random(seed + int(ratio * 1000) + expert_id)
        source_nodes.extend(sorted(rng.sample(nodes, quotas[expert_id])))
    source_nodes = sorted(source_nodes)
    target_nodes = sorted(idx for idx in range(total_nodes) if idx not in set(source_nodes))
    return source_nodes, target_nodes


def _effective_eval_args(args, *, strict_protocol: bool):
    if not strict_protocol:
        return args
    data = vars(args).copy()
    data["max_eval"] = 0
    data["sampling"] = "head"
    return argparse.Namespace(**data)


def _expert_node_subsets(router: HardRouter, source_nodes: list[int]) -> dict[int, list[int]]:
    by_expert = {0: [], 1: []}
    for node_idx in source_nodes:
        by_expert[router.route(node_idx)].append(int(node_idx))
    return by_expert


def _distribute_total_cap(full_counts: dict[int, int], total_cap: int) -> dict[int, int]:
    if total_cap <= 0:
        return {eid: int(v) for eid, v in full_counts.items()}

    total_full = sum(int(v) for v in full_counts.values())
    if total_full <= total_cap:
        return {eid: int(v) for eid, v in full_counts.items()}

    caps = {}
    remainders = []
    assigned = 0
    for eid, count in full_counts.items():
        exact = float(total_cap) * float(count) / float(total_full) if total_full > 0 else 0.0
        base = min(int(count), int(exact))
        caps[eid] = base
        assigned += base
        remainders.append((exact - base, eid))

    remaining = max(0, int(total_cap) - assigned)
    for _, eid in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if caps[eid] < int(full_counts[eid]):
            caps[eid] += 1
            remaining -= 1
    return caps


def _build_tagged_samples_with_cap(
    train_raw: list[dict],
    node_subset: list[int],
    cap: int,
) -> list[dict]:
    if cap < 0:
        raise ValueError(f"cap must be non-negative, got {cap}")
    if cap == 0:
        return []
    return build_tagged_node_samples(train_raw, node_subset, max_items=cap)


def _make_expert_train_args(base_args, *, expert_train_cap: int, expert_retrieval_cap: int):
    data = vars(base_args).copy()
    data["max_samples_per_expert"] = int(expert_train_cap)
    data["retrieval_bank_max_samples_per_expert"] = int(expert_retrieval_cap)
    data["eval_steps"] = int(base_args.eval_steps)
    data["rebuild_tokenized_cache"] = False
    return argparse.Namespace(**data)


def _load_expert_models(expert_dirs: dict[int, str]):
    base_model, tokenizer = load_model_and_tokenizer()
    expert_0 = load_peft_model(base_model, expert_dirs[0])
    expert_0.eval()
    if hasattr(expert_0, "gradient_checkpointing_disable"):
        expert_0.gradient_checkpointing_disable()
    if hasattr(expert_0, "config"):
        expert_0.config.use_cache = True

    del base_model
    cleanup_model()

    base_model, _ = load_model_and_tokenizer()
    expert_1 = load_peft_model(base_model, expert_dirs[1])
    expert_1.eval()
    if hasattr(expert_1, "gradient_checkpointing_disable"):
        expert_1.gradient_checkpointing_disable()
    if hasattr(expert_1, "config"):
        expert_1.config.use_cache = True
    return {0: expert_0, 1: expert_1}, tokenizer


def _evaluate_zero_shot_moe(
    *,
    expert_dirs: dict[int, str],
    retrievers: dict[int, object],
    router: HardRouter,
    test_samples: list[dict],
    retrieval_query_samples: list[dict] | None,
    eval_target_nodes: list[int],
    splits: dict,
    args,
) -> dict:
    experts, tokenizer = _load_expert_models(expert_dirs)
    subset = select_sample_subset_with_indices(test_samples, args.max_eval, args.sampling, args.seed)

    preds, trues = [], []
    routing_stats = {0: 0, 1: 0}
    domain_preds = {"CBD": [], "Residential": []}
    domain_trues = {"CBD": [], "Residential": []}
    domain_parse_stats = {
        "CBD": {"requested": 0, "parsed": 0},
        "Residential": {"requested": 0, "parsed": 0},
    }
    requested = len(subset) * len(eval_target_nodes)

    for sample_index, sample in subset:
        retrieval_query_sample = None if retrieval_query_samples is None else retrieval_query_samples[sample_index]
        jobs_by_expert = {0: [], 1: []}

        for node_idx in eval_target_nodes:
            expert_id = router.route(node_idx)
            expert_domain = "CBD" if expert_id == 0 else "Residential"
            domain_parse_stats[expert_domain]["requested"] += 1
            static_context = extract_node_static_context(
                node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )
            domain_label = normalise_domain_label(static_context.get("zone_type")) or expert_domain
            sys_msg, usr_msg = _build_prompt(
                sample=sample,
                node_idx=node_idx,
                horizon=args.horizon,
                prompt_style=args.prompt_style,
                use_rag=args.use_rag,
                use_diff_dora=args.use_diff_dora,
                retriever=retrievers.get(expert_id),
                retrieval_query_sample=retrieval_query_sample,
                static_context=static_context,
                domain_label=domain_label,
                splits=splits,
            )
            jobs_by_expert[expert_id].append(
                {
                    "node_idx": node_idx,
                    "domain": expert_domain,
                    "prompt": (sys_msg, usr_msg),
                    "target": sample["y"][:args.horizon, node_idx],
                }
            )

        for expert_id, jobs in jobs_by_expert.items():
            if not jobs:
                continue
            routing_stats[expert_id] += len(jobs)
            for start in range(0, len(jobs), max(1, args.infer_batch_size)):
                chunk = jobs[start:start + max(1, args.infer_batch_size)]
                prompts = [job["prompt"] for job in chunk]
                raw_outputs = generate_batch(
                    experts[expert_id],
                    tokenizer,
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                )
                for job, raw_output in zip(chunk, raw_outputs):
                    arr = parse_output(raw_output, expected_len=args.horizon)
                    if arr is not None and len(arr) == args.horizon:
                        preds.append(arr)
                        trues.append(job["target"])
                        domain_preds[job["domain"]].append(arr)
                        domain_trues[job["domain"]].append(job["target"])
                        domain_parse_stats[job["domain"]]["parsed"] += 1

    result = {
        "requested_predictions": requested,
        "parsed_predictions": len(preds),
        "parse_success_rate": len(preds) / max(requested, 1),
        "routing_stats": routing_stats,
        "metrics": None,
        "domain_metrics": {"CBD": None, "Residential": None},
        "domain_parse_stats": domain_parse_stats,
    }
    if preds:
        result["metrics"] = per_horizon_metrics(
            preds,
            trues,
            args.horizon,
            splits["norm_min"],
            splits["norm_max"],
        )
    for domain, stats in domain_parse_stats.items():
        stats["parse_success_rate"] = stats["parsed"] / max(stats["requested"], 1)
        if domain_preds[domain]:
            result["domain_metrics"][domain] = per_horizon_metrics(
                domain_preds[domain],
                domain_trues[domain],
                args.horizon,
                splits["norm_min"],
                splits["norm_max"],
            )

    cleanup_model(experts[0], experts[1], tokenizer)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--output_dir", default="outputs/zeroshot")
    parser.add_argument(
        "--source_ratios",
        default=DEFAULT_SOURCE_RATIOS,
        help="Comma-separated source-node ratios, e.g. 0.2,0.4,0.6,0.8",
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
    parser.add_argument("--eval_steps", type=int, default=0)
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
        help="Total training item cap across both experts; <=0 uses all source items.",
    )
    parser.add_argument(
        "--max_eval",
        type=int,
        default=0,
        help="Evaluation sample count; <=0 uses the full test split.",
    )
    parser.add_argument(
        "--max_target_nodes",
        type=int,
        default=0,
        help="Optional target-node cap; <=0 uses all target nodes.",
    )
    parser.add_argument("--target_node_sampling", choices=["head", "random"], default="head")
    parser.add_argument("--sampling", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip expert training and evaluate using existing expert adapters.")
    parser.add_argument("--expert_0_dir", default="",
                        help="Optional explicit expert_0 adapter dir for eval-only mode.")
    parser.add_argument("--expert_1_dir", default="",
                        help="Optional explicit expert_1 adapter dir for eval-only mode.")
    parser.add_argument("--retrieval_device", default="cpu",
                        help="Retrieval query backend: cpu, auto, or a CUDA device like cuda:0.")
    parser.add_argument("--retrieval_query_batch_size", type=int, default=128,
                        help="Number of query samples to score per retrieval batch when GPU retrieval is enabled.")
    parser.add_argument("--retrieval_corpus_chunk_size", type=int, default=65536,
                        help="Number of retrieval-bank vectors to compare per GPU retrieval chunk.")
    parser.add_argument("--manifest_path", default="")
    parser.add_argument(
        "--normalization_source",
        choices=[FULL_DATA_NORMALIZATION, TRAIN_ONLY_NORMALIZATION],
        default=FULL_DATA_NORMALIZATION,
    )
    parser.add_argument("--zeroshot_half_train_ratios", default="0.60,0.80")
    parser.add_argument("--zeroshot_test_window_divisor", type=int, default=DEFAULT_ZEROSHOT_TEST_WINDOW_DIVISOR)
    parser.add_argument("--node_split", choices=["random", "stratified_zone"], default="random")
    parser.add_argument("--retrieval_query_view", choices=["full_graph", "source_masked"], default="full_graph")
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
        if args.node_split == "random":
            args.node_split = "stratified_zone"
        if args.retrieval_query_view == "full_graph":
            args.retrieval_query_view = "source_masked"
        if args.zeroshot_half_train_ratios == "0.60,0.80":
            args.zeroshot_half_train_ratios = ",".join(f"{v:.2f}" for v in DEFAULT_ZEROSHOT_HALF_TRAIN_RATIOS)
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
        if args.node_split != "stratified_zone":
            raise ValueError("--strict_protocol requires --node_split stratified_zone.")
        if args.retrieval_query_view != "source_masked":
            raise ValueError("--strict_protocol requires --retrieval_query_view source_masked.")
        if args.max_eval > 0 or args.max_target_nodes > 0:
            raise ValueError("--strict_protocol requires full target-node evaluation.")

    raw = _load_dataset(args.dataset, normalization_source=args.normalization_source, horizon=args.horizon)
    splits = build_splits(raw, args.dataset)
    labels = build_routing_labels(splits["train"], splits.get("node_meta"))
    router = HardRouter(labels)
    total_nodes = splits["train"].shape[1]
    ratios = parse_ratio_csv(args.source_ratios)
    half_train_ratios = parse_ratio_csv(args.zeroshot_half_train_ratios)

    test_raw_full = build_samples(
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
    if args.strict_protocol:
        manifest, manifest_path = ensure_strict_zeroshot_manifest(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.window_stride,
            seed=args.seed,
            normalization_source=args.normalization_source,
            max_train_items=args.max_train_items,
            source_ratios=ratios,
            half_train_ratios=half_train_ratios,
            test_window_divisor=args.zeroshot_test_window_divisor,
            manifest_path=args.manifest_path or None,
            force_rebuild=False,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    eval_args = _effective_eval_args(args, strict_protocol=args.strict_protocol)

    for ratio in ratios:
        ratio_dir = out_dir / f"source_ratio_{ratio:.2f}"
        ratio_manifest = None
        if manifest is not None:
            ratio_manifest = manifest["source_ratios"][ratio_key(ratio)]
            source_nodes = ratio_manifest["source_nodes"]
            target_nodes = ratio_manifest["target_nodes"]
            eval_target_nodes = ratio_manifest["eval_target_nodes"]
            test_window_indices = ratio_manifest.get("test_window_indices", manifest.get("test_window_indices", []))
            if test_window_indices:
                test_raw = [test_raw_full[idx] for idx in test_window_indices]
            else:
                test_raw = list(test_raw_full)
            with open(ratio_manifest["sample_cache_path"], "rb") as f:
                cached = pickle.load(f)
            train_raw = cached["train_samples"]
        else:
            if args.node_split == "stratified_zone":
                source_nodes, target_nodes = _split_source_target_nodes_stratified(
                    router,
                    total_nodes,
                    ratio,
                    args.seed,
                )
            else:
                source_nodes, target_nodes = _split_source_target_nodes(total_nodes, ratio, args.seed)
            eval_target_nodes = select_node_subset(
                target_nodes,
                args.max_target_nodes,
                args.target_node_sampling,
                args.seed,
            )
            source_occ_train, source_adj_train = mask_occ_and_adj(
                splits["train"],
                splits.get("adj"),
                source_nodes,
            )
            train_raw = build_samples(
                source_occ_train,
                splits["timestamps_train"],
                adj=source_adj_train,
                horizons=[args.horizon],
                history_len=args.history_len,
                neighbor_k=args.neighbor_k,
                window_stride=args.window_stride,
            )[args.horizon]
            test_raw = list(test_raw_full)

        source_nodes_by_expert = _expert_node_subsets(router, source_nodes)
        if not source_nodes_by_expert[0] or not source_nodes_by_expert[1]:
            raise ValueError(
                f"Zero-shot ratio={ratio:.2f} leaves an empty expert source pool: "
                f"expert_0={len(source_nodes_by_expert[0])}, expert_1={len(source_nodes_by_expert[1])}"
            )

        full_item_counts = {
            eid: len(train_raw) * len(source_nodes_by_expert[eid])
            for eid in (0, 1)
        }
        expert_train_caps = _distribute_total_cap(full_item_counts, args.max_train_items)
        tagged = {
            eid: _build_tagged_samples_with_cap(
                train_raw,
                source_nodes_by_expert[eid],
                expert_train_caps[eid],
            )
            for eid in (0, 1)
        }
        if not tagged[0] or not tagged[1]:
            raise ValueError(
                f"Zero-shot ratio={ratio:.2f} yields empty expert training items after capping: "
                f"expert_0_cap={expert_train_caps[0]}, expert_1_cap={expert_train_caps[1]}"
            )
        retrievers = {
            eid: make_retriever(
                tagged[eid],
                use_rag=args.use_rag,
                query_device=args.retrieval_device,
                query_batch_size=args.retrieval_query_batch_size,
                corpus_chunk_size=args.retrieval_corpus_chunk_size,
            )
            for eid in (0, 1)
        }
        retrieval_query_samples = (
            mask_samples_for_nodes(test_raw, source_nodes)
            if args.retrieval_query_view == "source_masked"
            else None
        )

        print(
            f"\n=== Zero-shot MoE source_ratio={ratio:.2f} "
            f"| source_nodes={len(source_nodes)} target_nodes={len(target_nodes)} "
            f"| expert_0_items={len(tagged[0])} expert_1_items={len(tagged[1])} ==="
        )

        trained = {}
        if args.eval_only:
            default_expert_root = ratio_dir
            trained[0] = args.expert_0_dir or str(default_expert_root / "expert_0" / "adapter")
            trained[1] = args.expert_1_dir or str(default_expert_root / "expert_1" / "adapter")
            for expert_id in (0, 1):
                if not Path(trained[expert_id]).exists():
                    raise FileNotFoundError(
                        f"Eval-only adapter for expert_{expert_id} not found: {trained[expert_id]}"
                    )
            print(
                f"Using existing expert adapters for eval-only mode: "
                f"expert_0={trained[0]} expert_1={trained[1]}"
            )
        else:
            for expert_id in (0, 1):
                expert_args = _make_expert_train_args(
                    args,
                    expert_train_cap=len(tagged[expert_id]),
                    expert_retrieval_cap=len(tagged[expert_id]),
                )
                trained[expert_id] = train_one_expert(
                    expert_id,
                    tagged[expert_id],
                    ratio_dir,
                    expert_args,
                    retriever=retrievers[expert_id],
                    weather=splits.get("weather"),
                    price=splits.get("price"),
                    node_meta=splits.get("node_meta"),
                    node_ids=splits.get("node_ids"),
                    poi=splits.get("poi"),
                    tokenized_cache_dir=None,
                    retrieval_cache_dir=None,
                    variant_name=f"zeroshot_src{ratio_key(ratio)}",
                )

        eval_result = _evaluate_zero_shot_moe(
            expert_dirs=trained,
            retrievers=retrievers,
            router=router,
            test_samples=test_raw,
            retrieval_query_samples=retrieval_query_samples,
            eval_target_nodes=eval_target_nodes,
            splits=splits,
            args=eval_args,
        )
        cleanup_model(retrievers[0], retrievers[1])
        gc.collect()

        result_payload = {
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "eval_target_nodes": eval_target_nodes,
            "train_window_count": len(train_raw),
            "test_window_count": len(test_raw),
            "train_item_count": len(tagged[0]) + len(tagged[1]),
            "expert_train_item_counts": {str(eid): len(tagged[eid]) for eid in (0, 1)},
            "expert_source_node_counts": {str(eid): len(source_nodes_by_expert[eid]) for eid in (0, 1)},
            "expert_adapter_dirs": {str(eid): trained[eid] for eid in (0, 1)},
            "window_stride": int(args.window_stride),
            "normalization_source": args.normalization_source,
            "node_split": args.node_split,
            "retrieval_query_view": args.retrieval_query_view,
            "retrieval_device": args.retrieval_device,
            "retrieval_query_batch_size": args.retrieval_query_batch_size,
            "retrieval_corpus_chunk_size": args.retrieval_corpus_chunk_size,
            "training_architecture": "hard_routing_moe",
            **eval_result,
        }
        if ratio_manifest is not None:
            result_payload["domain_counts"] = ratio_manifest["domain_counts"]
            result_payload["sample_cache_path"] = ratio_manifest["sample_cache_path"]
            result_payload["manifest_hash"] = manifest["manifest_hash"]
            result_payload["manifest_path"] = str(manifest_path)
        results[str(ratio)] = result_payload

        metrics = eval_result.get("metrics")
        if metrics:
            print(
                f"source_ratio={ratio:.2f} "
                f"RMSE={metrics['overall']['rmse']:.4f} "
                f"MAE={metrics['overall']['mae']:.4f} "
                f"parse={eval_result['parse_success_rate']:.3f}"
            )

    payload = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "source_ratios": ratios,
        "use_dora": args.use_dora,
        "use_diff_dora": args.use_diff_dora,
        "use_rag": args.use_rag,
        "prompt_style": args.prompt_style,
        "max_train_items": args.max_train_items,
        "max_eval": eval_args.max_eval,
        "sampling": eval_args.sampling,
        "seed": args.seed,
        "window_stride": args.window_stride,
        "normalization_source": args.normalization_source,
        "node_split": args.node_split,
        "retrieval_query_view": args.retrieval_query_view,
        "retrieval_device": args.retrieval_device,
        "retrieval_query_batch_size": args.retrieval_query_batch_size,
        "retrieval_corpus_chunk_size": args.retrieval_corpus_chunk_size,
        "strict_protocol": args.strict_protocol,
        "zeroshot_half_train_ratios": half_train_ratios,
        "zeroshot_test_window_divisor": args.zeroshot_test_window_divisor,
        "training_architecture": "hard_routing_moe",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    if manifest is not None:
        payload["manifest_hash"] = manifest["manifest_hash"]
        payload["manifest_path"] = str(manifest_path)
    with open(out_dir / "zeroshot_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out_dir}/zeroshot_results.json")


if __name__ == "__main__":
    main()
