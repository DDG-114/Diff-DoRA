"""
src/eval/eval_moe_routed.py
---------------------------
Evaluate MoE experts with hard routing based on node physical attributes (CBD/Residential).

In the inference phase, the model activates the corresponding expert based on the
metadata label in the input prompt, exactly as described in the LR-MoE paper.

Usage:
  /home/kaga/diffdora/.venv/bin/python -m src.eval.eval_moe_routed \
      --dataset urbanev \
      --horizon 6 \
      --split test \
      --expert_0_dir outputs/urbanev_moe_experts_cot_rag_h6/expert_0/adapter \
      --expert_1_dir outputs/urbanev_moe_experts_cot_rag_h6/expert_1/adapter \
      --use_rag \
      --max_eval 50 \
      --max_new_tokens 160 \
      --output outputs/urbanev_moe_experts_cot_rag_h6/moe_routed_eval.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.data.build_splits import build_splits
from src.data.build_samples import build_samples
from src.utils.node_context import extract_node_static_context
from src.eval.metrics import per_horizon_metrics
from src.models.qwen_peft import load_model_and_tokenizer, load_peft_model, generate
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.parser import parse_output
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.knn_retriever import KNNRetriever
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router import HardRouter


def _load_dataset(dataset: str) -> dict:
    return load_st_evcdp() if dataset == "st_evcdp" else load_urbanev()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="urbanev", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--expert_0_dir", required=True, help="Path to expert_0 adapter (CBD)")
    parser.add_argument("--expert_1_dir", required=True, help="Path to expert_1 adapter (Residential)")
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/{dataset}_h{horizon}.pkl")
    parser.add_argument("--max_eval", type=int, default=100)
    parser.add_argument("--max_nodes", type=int, default=0,
                        help="If >0, only evaluate the first N nodes; default evaluates all nodes.")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--sampling", choices=["head", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_prompts", action="store_true", default=True,
                        help="Save system and user prompts in output JSON (default: True)")
    parser.add_argument("--output", default="outputs/eval_moe_routed.json")
    args = parser.parse_args()

    raw = _load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)

    # Build router based on training data physical attributes
    labels = build_routing_labels(splits["train"], raw.get("node_meta"))
    router = HardRouter(labels)

    # Load test samples
    split_key = args.split
    ts_key = f"timestamps_{split_key}"
    sample_map = build_samples(
        splits[split_key],
        splits[ts_key],
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

    # Load retriever
    retriever = None
    if args.use_rag:
        if args.retrieval_cache:
            cache_path = Path(args.retrieval_cache)
        else:
            cache_path = Path(f"data/retrieval_cache/{args.dataset}_h{args.horizon}.pkl")
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Retrieval cache not found: {cache_path}. Build cache first or pass --retrieval_cache."
            )
        retriever = KNNRetriever.load(cache_path)

    # Load experts
    print("Loading base model and experts …")
    base_model, tokenizer = load_model_and_tokenizer()
    expert_0 = load_peft_model(base_model, args.expert_0_dir)
    expert_0.eval()
    if hasattr(expert_0, "gradient_checkpointing_disable"):
        expert_0.gradient_checkpointing_disable()
    if hasattr(expert_0, "config"):
        expert_0.config.use_cache = True

    # Re-load base model for expert_1 to avoid CUDA conflicts
    del base_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    base_model, _ = load_model_and_tokenizer()
    expert_1 = load_peft_model(base_model, args.expert_1_dir)
    expert_1.eval()
    if hasattr(expert_1, "gradient_checkpointing_disable"):
        expert_1.gradient_checkpointing_disable()
    if hasattr(expert_1, "config"):
        expert_1.config.use_cache = True

    experts = {0: expert_0, 1: expert_1}

    preds, trues = [], []
    records = []
    routing_stats = {0: 0, 1: 0}
    node_indices = list(range(splits[split_key].shape[1]))
    if args.max_nodes > 0:
        node_indices = node_indices[:args.max_nodes]

    print(f"Evaluating MoE on {args.split} split with hard routing …")
    print(f"Total inferences: {len(subset)} samples x {len(node_indices)} nodes = {len(subset) * len(node_indices)}\n")
    total_done = 0
    total_requested = len(subset) * len(node_indices)
    for i, sample in enumerate(subset):
        if args.use_rag and retriever is not None:
            retrieved = retriever.query(sample, exclude_t_start=None)
            diff = compute_diff_features(query_sample=sample, retrieved_samples=retrieved)
            retrieved_t_starts = [int(s.get("t_start", -1)) for s in retrieved]
        else:
            retrieved = []
            diff = None
            retrieved_t_starts = []

        for sample_node_idx in node_indices:
            total_done += 1

            # HARD ROUTING: route based on node's physical attribute (CBD or Residential)
            expert_id = router.route(sample_node_idx)
            routing_stats[expert_id] += 1

            expert = experts[expert_id]
            expert_domain = "CBD" if expert_id == 0 else "Residential"
            static_context = extract_node_static_context(
                sample_node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )

            # Build prompt with physical attribute metadata
            if args.use_rag and retriever is not None and diff is not None:
                sys_msg, usr_msg = build_cot_prompt(
                    sample,
                    retrieved,
                    diff,
                    sample_node_idx,
                    args.horizon,
                    domain_label=expert_domain,
                    static_context=static_context,
                )
            else:
                sys_msg, usr_msg = build_vanilla_prompt(
                    sample,
                    sample_node_idx,
                    args.horizon,
                    domain_label=expert_domain,
                    static_context=static_context,
                )

            raw_output = generate(
                expert,
                tokenizer,
                sys_msg,
                usr_msg,
                max_new_tokens=args.max_new_tokens,
            )

            parsed = parse_output(raw_output, expected_len=args.horizon)
            parse_ok = parsed is not None and len(parsed) == args.horizon

            target = sample["y"][:args.horizon, sample_node_idx]
            if parse_ok:
                preds.append(parsed)
                trues.append(target)

            row = {
                "sample_index": i,
                "t_start": int(sample.get("t_start", -1)),
                "node_idx": sample_node_idx,
                "expert_id": expert_id,
                "expert_domain": expert_domain,
                "parse_ok": parse_ok,
                "retrieved_t_starts": retrieved_t_starts,
                "raw_generation": raw_output,
                "parsed_prediction": parsed.tolist() if parse_ok else None,
                "target": target.tolist(),
            }
            if args.save_prompts:
                row["system_prompt"] = sys_msg
                row["user_prompt"] = usr_msg
            records.append(row)

        # Progress display every 5 samples
        if (i + 1) % 5 == 0 or i == len(subset) - 1:
            parsed_so_far = len(preds)
            print(f"[{total_done:4d}/{total_requested}] Parsed: {parsed_so_far:4d} | "
                  f"Expert_0(CBD): {routing_stats[0]:4d} | Expert_1(Res): {routing_stats[1]:4d}")

    requested = total_requested
    parsed_n = len(preds)

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "horizon": args.horizon,
        "expert_0_dir": args.expert_0_dir,
        "expert_1_dir": args.expert_1_dir,
        "use_rag": args.use_rag,
        "history_len": args.history_len,
        "neighbor_k": args.neighbor_k,
        "max_eval": args.max_eval,
        "max_nodes": args.max_nodes,
        "sampling": args.sampling,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "requested_samples": requested,
        "evaluated_samples": parsed_n,
        "parse_failures": requested - parsed_n,
        "parse_success_rate": parsed_n / max(requested, 1),
        "routing_stats": routing_stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": None,
        "records": records,
    }

    if preds:
        metrics = per_horizon_metrics(preds, trues, args.horizon, splits["norm_min"], splits["norm_max"])
        result["metrics"] = metrics
        print(json.dumps(metrics, indent=2))
        print(f"\nRouting stats:")
        print(f"  Expert 0 (CBD): {routing_stats[0]} samples")
        print(f"  Expert 1 (Residential): {routing_stats[1]} samples")
    else:
        print("No parseable predictions.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
