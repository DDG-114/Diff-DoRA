"""
src/eval/eval_paper_ablation.py
--------------------------------
Evaluate the paper-defined ablation variants.

This entrypoint now supports expert-only ablation runs with fixed sample/node
subsets so routed variants can be compared fairly under a limited runtime
budget.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.utils.node_context import extract_node_static_context, resolve_node_id
from src.eval.diagnostics import compute_sequence_diagnostics
from src.eval.metrics import per_horizon_metrics
from src.models.diff_dora import DiffDoRAModel, set_diff_context
from src.models.qwen_peft import load_model_and_tokenizer, load_peft_model, generate_batch
from src.prompts.parser import parse_output
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.prompt_vanilla import build_direct_physical_prompt, build_vanilla_prompt
from src.retrieval.diff_features import compute_diff_features, format_diff_block
from src.retrieval.knn_retriever import KNNRetriever
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router import HardRouter

PARSE_RERUN_THRESHOLD = 0.95
PARSE_RERUN_MAX_NEW_TOKENS = 320


def _load_dataset(dataset: str) -> dict:
    return load_st_evcdp() if dataset == "st_evcdp" else load_urbanev()


def _load_model(
    adapter_dir: str | None,
    *,
    use_diff_dora: bool,
    diff_hidden_dim: int,
    diff_scale: float,
):
    base_model, tokenizer = load_model_and_tokenizer()
    if adapter_dir:
        model = load_peft_model(base_model, adapter_dir)
    else:
        model = base_model

    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    if not use_diff_dora:
        return model, tokenizer, base_model

    wrapped = DiffDoRAModel(
        model,
        diff_input_dim=3,
        hidden_dim=diff_hidden_dim,
        scale=diff_scale,
    )
    wrapped.to(next(model.parameters()).device)
    if adapter_dir:
        ctrl_path = Path(adapter_dir).parent / "diff_controller.pt"
        if ctrl_path.exists():
            wrapped.controller.load_state_dict(torch.load(ctrl_path, map_location="cpu"))
            wrapped.controller.to(next(model.parameters()).device)
        else:
            print(f"[WARN] Diff-DoRA controller not found for {adapter_dir}; using default wrapper weights.")
    wrapped.eval()
    return wrapped, tokenizer, base_model


def _cleanup_models(*objs):
    for obj in objs:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _resolve_prompt_style(prompt_style: str, has_retrieval: bool) -> str:
    if prompt_style == "auto":
        return "cot" if has_retrieval else "vanilla"
    if prompt_style in {"cot", "direct_physical"} and not has_retrieval:
        raise ValueError(
            f"prompt_style={prompt_style!r} requires retrieval context, but no retriever was available."
        )
    return prompt_style


def _weather_at(weather, t_start: int) -> dict | None:
    if weather is None or getattr(weather, "empty", True):
        return None
    idx = min(max(int(t_start) + 11, 0), len(weather) - 1)
    row = weather.iloc[idx]
    if hasattr(row, "to_dict"):
        d = row.to_dict()
        for k in list(d.keys()):
            lk = str(k).lower()
            if "temp" in lk and "temperature" not in d:
                d["temperature"] = d[k]
            if "humid" in lk and "humidity" not in d:
                d["humidity"] = d[k]
        return d
    return None


def _price_at(price, t_start: int, node_idx: int, *, node_ids=None, node_meta=None) -> float | None:
    if price is None or getattr(price, "empty", True):
        return None
    idx = min(max(int(t_start) + 11, 0), len(price) - 1)
    row = price.iloc[idx]
    node_id = resolve_node_id(node_idx, node_ids=node_ids, node_meta=node_meta)
    if hasattr(row, "item") and not hasattr(row, "to_dict"):
        return float(row.item())
    if hasattr(row, "to_dict"):
        d = row.to_dict()
        if node_id in d:
            return float(d[node_id])
        if str(node_id) in d:
            return float(d[str(node_id)])
        if node_idx in d:
            return float(d[node_idx])
        if str(node_idx) in d:
            return float(d[str(node_idx)])
        vals = [float(v) for v in d.values() if v is not None]
        if vals:
            return float(sum(vals) / len(vals))
    return None


def _compute_retrieval_diff(sample: dict, retrieved: list[dict], splits: dict, node_idx: int) -> dict:
    weather_current = _weather_at(splits.get("weather"), sample.get("t_start", 0))
    weather_retrieved = [_weather_at(splits.get("weather"), rs.get("t_start", 0)) for rs in retrieved]
    price_current = _price_at(
        splits.get("price"),
        sample.get("t_start", 0),
        node_idx,
        node_ids=splits.get("node_ids"),
        node_meta=splits.get("node_meta"),
    )
    price_retrieved = [
        _price_at(
            splits.get("price"),
            rs.get("t_start", 0),
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


def _diff_tensor(diff: dict | None) -> torch.Tensor:
    if diff is None:
        return torch.zeros(3, dtype=torch.float32)
    return torch.tensor([
        float(diff.get("diff_occ", 0.0) or 0.0),
        float(diff.get("diff_temp", 0.0) or 0.0),
        float(diff.get("diff_price", 0.0) or 0.0),
    ], dtype=torch.float32)


def _build_prompt(
    *,
    sample: dict,
    node_idx: int,
    horizon: int,
    prompt_style: str,
    use_rag: bool,
    retriever: KNNRetriever | None,
    static_context: dict,
    domain_label: str,
    splits: dict,
):
    retrieved = []
    diff = None
    if use_rag and retriever is not None:
        retrieved = retriever.query(sample, exclude_t_start=None)
        diff = _compute_retrieval_diff(sample, retrieved, splits, node_idx)

    resolved_style = _resolve_prompt_style(prompt_style, bool(retrieved) and diff is not None)
    if resolved_style == "cot":
        prompt = build_cot_prompt(
            sample,
            retrieved,
            diff,
            node_idx=node_idx,
            horizon=horizon,
            domain_label=domain_label,
            static_context=static_context,
        )
    elif resolved_style == "direct_physical":
        prompt = build_direct_physical_prompt(
            sample,
            retrieved,
            format_diff_block(diff),
            node_idx=node_idx,
            horizon=horizon,
            domain_label=domain_label,
            static_context=static_context,
        )
    else:
        prompt = build_vanilla_prompt(
            sample,
            node_idx=node_idx,
            horizon=horizon,
            domain_label=domain_label,
            static_context=static_context,
        )
    return prompt, _diff_tensor(diff)


def _select_sample_subset(all_samples: list[dict], max_eval: int, sampling: str, seed: int):
    indexed = list(enumerate(all_samples))
    count = min(max_eval, len(indexed))
    if sampling == "random":
        rng = random.Random(seed)
        return rng.sample(indexed, count)
    return indexed[:count]


def _select_node_indices(
    *,
    router: HardRouter,
    total_nodes: int,
    max_nodes: int,
    node_sampling: str,
    max_nodes_per_domain: int,
    seed: int,
):
    cbd_nodes = router.nodes_for_expert(0)
    res_nodes = router.nodes_for_expert(1)
    if node_sampling == "balanced_random":
        rng = random.Random(seed)
        if max_nodes_per_domain > 0:
            per_domain = min(max_nodes_per_domain, len(cbd_nodes), len(res_nodes))
        else:
            per_domain = min(len(cbd_nodes), len(res_nodes))
        selected_cbd = sorted(rng.sample(cbd_nodes, per_domain))
        selected_res = sorted(rng.sample(res_nodes, per_domain))
        return selected_cbd + selected_res, {
            "CBD": selected_cbd,
            "Residential": selected_res,
        }

    node_indices = list(range(total_nodes))
    if max_nodes > 0:
        node_indices = node_indices[:max_nodes]
    by_domain = {
        "CBD": [idx for idx in node_indices if router.route(idx) == 0],
        "Residential": [idx for idx in node_indices if router.route(idx) == 1],
    }
    return node_indices, by_domain


def _batched(items: list[dict], batch_size: int):
    for start_idx in range(0, len(items), max(1, batch_size)):
        yield items[start_idx:start_idx + max(1, batch_size)]


def evaluate_variant(
    *,
    variant_name: str,
    variant_cfg: dict,
    splits: dict,
    subset: list[tuple[int, dict]],
    node_indices: list[int],
    router: HardRouter,
    retriever: KNNRetriever | None,
    horizon: int,
    max_new_tokens: int,
    infer_batch_size: int,
) -> dict:
    preds, trues = [], []
    records = []
    routing_stats = {0: 0, 1: 0}
    domain_preds = {"CBD": [], "Residential": []}
    domain_trues = {"CBD": [], "Residential": []}
    domain_parse_stats = {
        "CBD": {"requested": 0, "parsed": 0},
        "Residential": {"requested": 0, "parsed": 0},
    }

    if variant_cfg["mode"] == "routed":
        model_0, tokenizer, base_0 = _load_model(
            variant_cfg["expert_0_dir"],
            use_diff_dora=variant_cfg.get("use_diff_dora", False),
            diff_hidden_dim=variant_cfg["diff_hidden_dim"],
            diff_scale=variant_cfg["diff_scale"],
        )
        model_1, _, base_1 = _load_model(
            variant_cfg["expert_1_dir"],
            use_diff_dora=variant_cfg.get("use_diff_dora", False),
            diff_hidden_dim=variant_cfg["diff_hidden_dim"],
            diff_scale=variant_cfg["diff_scale"],
        )
        models = {0: model_0, 1: model_1}
    else:
        shared_model, tokenizer, shared_base = _load_model(
            variant_cfg.get("adapter_dir"),
            use_diff_dora=variant_cfg.get("use_diff_dora", False),
            diff_hidden_dim=variant_cfg["diff_hidden_dim"],
            diff_scale=variant_cfg["diff_scale"],
        )
        models = {"shared": shared_model}

    total_requested = len(subset) * len(node_indices)
    total_done = 0
    for sample_offset, (sample_dataset_index, sample) in enumerate(subset):
        jobs_by_model: dict[object, list[dict]] = {key: [] for key in models.keys()}
        for node_idx in node_indices:
            expert_id = router.route(node_idx)
            domain_label = "CBD" if expert_id == 0 else "Residential"
            routing_stats[expert_id] += 1
            domain_parse_stats[domain_label]["requested"] += 1
            static_context = extract_node_static_context(
                node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )
            (sys_msg, usr_msg), diff_vec = _build_prompt(
                sample=sample,
                node_idx=node_idx,
                horizon=horizon,
                prompt_style=variant_cfg["prompt_style"],
                use_rag=variant_cfg["use_rag"],
                retriever=retriever,
                static_context=static_context,
                domain_label=domain_label,
                splits=splits,
            )
            model_key = expert_id if variant_cfg["mode"] == "routed" else "shared"
            jobs_by_model.setdefault(model_key, []).append({
                "node_idx": node_idx,
                "expert_id": expert_id if variant_cfg["mode"] == "routed" else None,
                "domain": domain_label,
                "target": sample["y"][:horizon, node_idx],
                "prompt": (sys_msg, usr_msg),
                "diff_vec": diff_vec,
                "sample_index": sample_offset,
                "sample_dataset_index": sample_dataset_index,
                "t_start": int(sample.get("t_start", -1)),
            })

        for model_key, jobs in jobs_by_model.items():
            if not jobs:
                continue
            model = models[model_key]
            for chunk in _batched(jobs, infer_batch_size):
                prompts = [job["prompt"] for job in chunk]
                if variant_cfg.get("use_diff_dora", False):
                    batch_diff = torch.stack([job["diff_vec"] for job in chunk]).float()
                    set_diff_context(batch_diff)
                raw_outputs = generate_batch(
                    model,
                    tokenizer,
                    prompts,
                    max_new_tokens=max_new_tokens,
                )
                if variant_cfg.get("use_diff_dora", False):
                    set_diff_context(None)

                for job, raw_output in zip(chunk, raw_outputs):
                    total_done += 1
                    parsed = parse_output(raw_output, expected_len=horizon)
                    parse_ok = parsed is not None and len(parsed) == horizon
                    target = job["target"]
                    if parse_ok:
                        preds.append(parsed)
                        trues.append(target)
                        domain_preds[job["domain"]].append(parsed)
                        domain_trues[job["domain"]].append(target)
                        domain_parse_stats[job["domain"]]["parsed"] += 1

                    records.append({
                        "sample_index": job["sample_index"],
                        "sample_dataset_index": job["sample_dataset_index"],
                        "t_start": job["t_start"],
                        "node_idx": job["node_idx"],
                        "expert_id": job["expert_id"],
                        "domain": job["domain"],
                        "parse_ok": parse_ok,
                        "parsed_prediction": parsed.tolist() if parse_ok else None,
                        "target": target.tolist(),
                        "raw_generation": raw_output,
                    })

                if total_done % 100 == 0 or total_done == total_requested:
                    print(f"[{variant_name}] {total_done}/{total_requested} calls, parsed={len(preds)}")

    for stats in domain_parse_stats.values():
        stats["parse_success_rate"] = stats["parsed"] / max(stats["requested"], 1)

    result = {
        "variant": variant_name,
        "mode": variant_cfg["mode"],
        "prompt_style": variant_cfg["prompt_style"],
        "max_new_tokens": max_new_tokens,
        "infer_batch_size": infer_batch_size,
        "requested_predictions": total_requested,
        "parsed_predictions": len(preds),
        "parse_success_rate": len(preds) / max(total_requested, 1),
        "routing_stats": routing_stats if variant_cfg["mode"] == "routed" else None,
        "domain_parse_stats": domain_parse_stats,
        "metrics": None,
        "domain_metrics": {},
        "diagnostics": compute_sequence_diagnostics(
            preds,
            trues,
            splits["norm_min"],
            splits["norm_max"],
        ),
        "domain_diagnostics": {},
        "records": records,
    }
    if preds:
        result["metrics"] = per_horizon_metrics(
            preds,
            trues,
            horizon,
            splits["norm_min"],
            splits["norm_max"],
        )
    for domain in ("CBD", "Residential"):
        if domain_preds[domain]:
            result["domain_metrics"][domain] = per_horizon_metrics(
                domain_preds[domain],
                domain_trues[domain],
                horizon,
                splits["norm_min"],
                splits["norm_max"],
            )
        else:
            result["domain_metrics"][domain] = None
        result["domain_diagnostics"][domain] = compute_sequence_diagnostics(
            domain_preds[domain],
            domain_trues[domain],
            splits["norm_min"],
            splits["norm_max"],
        )

    if variant_cfg["mode"] == "routed":
        _cleanup_models(model_0, model_1, base_0, base_1)
    else:
        _cleanup_models(shared_model, shared_base)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--retrieval_cache", default="")
    parser.add_argument("--max_eval", type=int, default=100)
    parser.add_argument("--max_nodes", type=int, default=0,
                        help="If >0, only evaluate the first N nodes when node_sampling=all.")
    parser.add_argument("--node_sampling", choices=["all", "balanced_random"], default="all")
    parser.add_argument("--max_nodes_per_domain", type=int, default=0,
                        help="Used when node_sampling=balanced_random; if 0, use the largest balanced subset.")
    parser.add_argument("--sampling", choices=["head", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--diff_hidden_dim", type=int, default=32)
    parser.add_argument("--diff_scale", type=float, default=0.5)

    parser.add_argument("--full_expert_0_dir", default=None)
    parser.add_argument("--full_expert_1_dir", default=None)
    parser.add_argument("--wo_moe_dir", default=None)
    parser.add_argument("--wo_cot_expert_0_dir", default=None)
    parser.add_argument("--wo_cot_expert_1_dir", default=None)
    parser.add_argument("--wo_dora_expert_0_dir", default=None)
    parser.add_argument("--wo_dora_expert_1_dir", default=None)
    parser.add_argument("--wo_diffdora_expert_0_dir", default=None)
    parser.add_argument("--wo_diffdora_expert_1_dir", default=None)
    parser.add_argument("--wo_rag_expert_0_dir", default=None)
    parser.add_argument("--wo_rag_expert_1_dir", default=None)
    parser.add_argument("--skip_base_model", action="store_true")
    parser.add_argument("--output", default="outputs/paper_ablation_results.json")
    args = parser.parse_args()

    raw = _load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    router = HardRouter(build_routing_labels(splits["train"], raw.get("node_meta")))

    split_key = args.split
    sample_map = build_samples(
        splits[split_key],
        splits[f"timestamps_{split_key}"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
    )
    all_samples = sample_map[args.horizon]
    subset = _select_sample_subset(all_samples, args.max_eval, args.sampling, args.seed)

    node_indices, nodes_by_domain = _select_node_indices(
        router=router,
        total_nodes=splits[split_key].shape[1],
        max_nodes=args.max_nodes,
        node_sampling=args.node_sampling,
        max_nodes_per_domain=args.max_nodes_per_domain,
        seed=args.seed,
    )

    retriever = None
    if args.use_rag:
        if args.retrieval_cache:
            cache_path = Path(args.retrieval_cache)
        else:
            cache_path = Path(f"data/retrieval_cache/{args.dataset}_h{args.horizon}.pkl")
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Retrieval cache not found: {cache_path}. Build it first or pass --retrieval_cache."
            )
        retriever = KNNRetriever.load(cache_path)

    variants: list[tuple[str, dict]] = []
    if args.full_expert_0_dir and args.full_expert_1_dir:
        variants.append(("full", {
            "mode": "routed",
            "expert_0_dir": args.full_expert_0_dir,
            "expert_1_dir": args.full_expert_1_dir,
            "prompt_style": "cot",
            "use_rag": args.use_rag,
            "use_diff_dora": True,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))
    if args.wo_moe_dir:
        variants.append(("wo_moe", {
            "mode": "shared",
            "adapter_dir": args.wo_moe_dir,
            "prompt_style": "cot",
            "use_rag": args.use_rag,
            "use_diff_dora": False,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))
    if args.wo_cot_expert_0_dir and args.wo_cot_expert_1_dir:
        variants.append(("wo_cot", {
            "mode": "routed",
            "expert_0_dir": args.wo_cot_expert_0_dir,
            "expert_1_dir": args.wo_cot_expert_1_dir,
            "prompt_style": "direct_physical",
            "use_rag": args.use_rag,
            "use_diff_dora": True,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))
    if args.wo_dora_expert_0_dir and args.wo_dora_expert_1_dir:
        variants.append(("wo_dora", {
            "mode": "routed",
            "expert_0_dir": args.wo_dora_expert_0_dir,
            "expert_1_dir": args.wo_dora_expert_1_dir,
            "prompt_style": "cot",
            "use_rag": args.use_rag,
            "use_diff_dora": False,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))
    if args.wo_diffdora_expert_0_dir and args.wo_diffdora_expert_1_dir:
        variants.append(("wo_diffdora", {
            "mode": "routed",
            "expert_0_dir": args.wo_diffdora_expert_0_dir,
            "expert_1_dir": args.wo_diffdora_expert_1_dir,
            "prompt_style": "cot",
            "use_rag": args.use_rag,
            "use_diff_dora": False,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))
    if args.wo_rag_expert_0_dir and args.wo_rag_expert_1_dir:
        variants.append(("wo_rag", {
            "mode": "routed",
            "expert_0_dir": args.wo_rag_expert_0_dir,
            "expert_1_dir": args.wo_rag_expert_1_dir,
            "prompt_style": "vanilla",
            "use_rag": False,
            "use_diff_dora": True,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))
    if not args.skip_base_model:
        variants.append(("base_model", {
            "mode": "shared",
            "adapter_dir": None,
            "prompt_style": "cot",
            "use_rag": args.use_rag,
            "use_diff_dora": False,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))

    if not variants:
        raise ValueError("No ablation variants were provided.")

    results = {}
    for name, cfg in variants:
        print(f"\n=== Evaluating variant: {name} ===")
        first_pass = evaluate_variant(
            variant_name=name,
            variant_cfg=cfg,
            splits=splits,
            subset=subset,
            node_indices=node_indices,
            router=router,
            retriever=retriever,
            horizon=args.horizon,
            max_new_tokens=args.max_new_tokens,
            infer_batch_size=args.infer_batch_size,
        )
        final_result = first_pass
        if first_pass["parse_success_rate"] < PARSE_RERUN_THRESHOLD and args.max_new_tokens < PARSE_RERUN_MAX_NEW_TOKENS:
            print(
                f"[{name}] parse_success_rate={first_pass['parse_success_rate']:.3f} < {PARSE_RERUN_THRESHOLD:.2f}; "
                f"rerunning with max_new_tokens={PARSE_RERUN_MAX_NEW_TOKENS}."
            )
            rerun_result = evaluate_variant(
                variant_name=name,
                variant_cfg=cfg,
                splits=splits,
                subset=subset,
                node_indices=node_indices,
                router=router,
                retriever=retriever,
                horizon=args.horizon,
                max_new_tokens=PARSE_RERUN_MAX_NEW_TOKENS,
                infer_batch_size=args.infer_batch_size,
            )
            rerun_result["rerun_from_max_new_tokens"] = args.max_new_tokens
            rerun_result["initial_parse_success_rate"] = first_pass["parse_success_rate"]
            rerun_result["initial_parsed_predictions"] = first_pass["parsed_predictions"]
            final_result = rerun_result
        results[name] = final_result
        metrics = final_result.get("metrics")
        if metrics:
            print(json.dumps(metrics["overall"], indent=2))
        else:
            print("No parseable predictions.")

    output = {
        "dataset": args.dataset,
        "split": args.split,
        "horizon": args.horizon,
        "norm_min": float(splits["norm_min"]),
        "norm_max": float(splits["norm_max"]),
        "history_len": args.history_len,
        "neighbor_k": args.neighbor_k,
        "max_eval": args.max_eval,
        "max_nodes": args.max_nodes,
        "node_sampling": args.node_sampling,
        "max_nodes_per_domain": args.max_nodes_per_domain,
        "selected_sample_indices": [idx for idx, _ in subset],
        "selected_sample_t_starts": [int(sample.get("t_start", -1)) for _, sample in subset],
        "selected_nodes": node_indices,
        "selected_nodes_by_domain": nodes_by_domain,
        "sampling": args.sampling,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "infer_batch_size": args.infer_batch_size,
        "parse_rerun_threshold": PARSE_RERUN_THRESHOLD,
        "parse_rerun_max_new_tokens": PARSE_RERUN_MAX_NEW_TOKENS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
