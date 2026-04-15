"""
src/eval/eval_paper_ablation.py
-------------------------------
Evaluate the paper-defined ablation variants:

- full LR-MoE
- w/o MoE
- w/o CoT
- w/o DoRA
- base model

This script is paper-oriented and intentionally separate from the lighter
engineering ablation runner in `eval_ablation.py`.
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
from src.utils.node_context import extract_node_static_context
from src.eval.metrics import per_horizon_metrics
from src.models.diff_dora import DiffDoRAModel, set_diff_context
from src.models.qwen_peft import load_model_and_tokenizer, load_peft_model, generate
from src.prompts.parser import parse_output
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.prompt_vanilla import build_direct_physical_prompt, build_vanilla_prompt
from src.retrieval.diff_features import compute_diff_features, format_diff_block
from src.retrieval.knn_retriever import KNNRetriever
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router import HardRouter


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
            print(f"[WARN] Diff-DoRA controller not found for {adapter_dir}; using the wrapper with default weights.")
    wrapped.eval()
    return wrapped, tokenizer, base_model


def _cleanup_models(*objs):
    for obj in objs:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _build_prompt(
    *,
    sample: dict,
    node_idx: int,
    horizon: int,
    use_cot: bool,
    use_rag: bool,
    retriever: KNNRetriever | None,
    static_context: dict,
    domain_label: str,
):
    if use_rag and retriever is not None:
        retrieved = retriever.query(sample, exclude_t_start=None)
        diff = compute_diff_features(query_sample=sample, retrieved_samples=retrieved)
    else:
        retrieved = []
        diff = None

    if use_cot and diff is not None:
        prompt = build_cot_prompt(
            sample,
            retrieved,
            diff,
            node_idx=node_idx,
            horizon=horizon,
            domain_label=domain_label,
            static_context=static_context,
        )
        diff_vec = torch.tensor([
            float(diff.get("diff_occ", 0.0) or 0.0),
            float(diff.get("diff_temp", 0.0) or 0.0),
            float(diff.get("diff_price", 0.0) or 0.0),
        ], dtype=torch.float32)
        return prompt, diff_vec

    if use_rag and diff is not None:
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
    return prompt, torch.zeros(3, dtype=torch.float32)


def evaluate_variant(
    *,
    variant_name: str,
    variant_cfg: dict,
    splits: dict,
    subset: list[dict],
    node_indices: list[int],
    router: HardRouter,
    retriever: KNNRetriever | None,
    horizon: int,
    max_new_tokens: int,
) -> dict:
    preds, trues = [], []
    records = []
    routing_stats = {0: 0, 1: 0}

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
    for sample_idx, sample in enumerate(subset):
        for node_idx in node_indices:
            total_done += 1
            expert_id = router.route(node_idx)
            domain_label = "CBD" if expert_id == 0 else "Residential"
            routing_stats[expert_id] += 1
            static_context = extract_node_static_context(
                node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )

            (sys_msg, usr_msg), diff_vec = _build_prompt(
                sample=sample,
                node_idx=node_idx,
                horizon=horizon,
                use_cot=variant_cfg["use_cot"],
                use_rag=variant_cfg["use_rag"],
                retriever=retriever,
                static_context=static_context,
                domain_label=domain_label,
            )

            model_key = expert_id if variant_cfg["mode"] == "routed" else "shared"
            model = models[model_key]
            if variant_cfg.get("use_diff_dora", False):
                set_diff_context(diff_vec)
            raw_output = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=max_new_tokens)
            if variant_cfg.get("use_diff_dora", False):
                set_diff_context(None)

            parsed = parse_output(raw_output, expected_len=horizon)
            parse_ok = parsed is not None and len(parsed) == horizon
            target = sample["y"][:horizon, node_idx]
            if parse_ok:
                preds.append(parsed)
                trues.append(target)

            records.append({
                "sample_index": sample_idx,
                "t_start": int(sample.get("t_start", -1)),
                "node_idx": node_idx,
                "expert_id": expert_id if variant_cfg["mode"] == "routed" else None,
                "domain": domain_label,
                "parse_ok": parse_ok,
                "parsed_prediction": parsed.tolist() if parse_ok else None,
                "target": target.tolist(),
                "raw_generation": raw_output,
            })

            if total_done % 100 == 0 or total_done == total_requested:
                print(f"[{variant_name}] {total_done}/{total_requested} calls, parsed={len(preds)}")

    result = {
        "variant": variant_name,
        "requested_predictions": total_requested,
        "parsed_predictions": len(preds),
        "parse_success_rate": len(preds) / max(total_requested, 1),
        "routing_stats": routing_stats if variant_cfg["mode"] == "routed" else None,
        "metrics": None,
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
                        help="If >0, only evaluate the first N nodes; default evaluates all nodes.")
    parser.add_argument("--sampling", choices=["head", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--diff_hidden_dim", type=int, default=32)
    parser.add_argument("--diff_scale", type=float, default=0.5)

    parser.add_argument("--full_expert_0_dir", default=None)
    parser.add_argument("--full_expert_1_dir", default=None)
    parser.add_argument("--wo_moe_dir", default=None)
    parser.add_argument("--wo_cot_expert_0_dir", default=None)
    parser.add_argument("--wo_cot_expert_1_dir", default=None)
    parser.add_argument("--wo_dora_expert_0_dir", default=None)
    parser.add_argument("--wo_dora_expert_1_dir", default=None)
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
    if args.sampling == "random":
        rng = random.Random(args.seed)
        subset = rng.sample(all_samples, min(args.max_eval, len(all_samples)))
    else:
        subset = all_samples[:args.max_eval]

    node_indices = list(range(splits[split_key].shape[1]))
    if args.max_nodes > 0:
        node_indices = node_indices[:args.max_nodes]

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
            "use_cot": True,
            "use_rag": args.use_rag,
            "use_diff_dora": True,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))
    if args.wo_moe_dir:
        variants.append(("wo_moe", {
            "mode": "shared",
            "adapter_dir": args.wo_moe_dir,
            "use_cot": True,
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
            "use_cot": False,
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
            "use_cot": True,
            "use_rag": args.use_rag,
            "use_diff_dora": False,
            "diff_hidden_dim": args.diff_hidden_dim,
            "diff_scale": args.diff_scale,
        }))
    if not args.skip_base_model:
        variants.append(("base_model", {
            "mode": "shared",
            "adapter_dir": None,
            "use_cot": True,
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
        results[name] = evaluate_variant(
            variant_name=name,
            variant_cfg=cfg,
            splits=splits,
            subset=subset,
            node_indices=node_indices,
            router=router,
            retriever=retriever,
            horizon=args.horizon,
            max_new_tokens=args.max_new_tokens,
        )
        metrics = results[name].get("metrics")
        if metrics:
            print(json.dumps(metrics["overall"], indent=2))
        else:
            print("No parseable predictions.")

    output = {
        "dataset": args.dataset,
        "split": args.split,
        "horizon": args.horizon,
        "history_len": args.history_len,
        "neighbor_k": args.neighbor_k,
        "max_eval": args.max_eval,
        "max_nodes": args.max_nodes,
        "sampling": args.sampling,
        "seed": args.seed,
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
