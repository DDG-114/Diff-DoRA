from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.eval.eval_paper_ablation import _build_prompt
from src.models.qwen_peft import generate_batch, load_model_and_tokenizer, load_peft_model
from src.prompts.parser import parse_output
from src.retrieval.knn_retriever import KNNRetriever
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router import HardRouter
from src.utils.node_context import extract_node_static_context


def _load_dataset(dataset: str) -> dict:
    return load_st_evcdp() if dataset == "st_evcdp" else load_urbanev()


def _load_expert(adapter_dir: str):
    base_model, tokenizer = load_model_and_tokenizer()
    model = load_peft_model(base_model, adapter_dir)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer, base_model


def _parse_node_csv(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected at least one node index.")
    return values


def _batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start:start + max(1, batch_size)]


def main():
    parser = argparse.ArgumentParser(description="Evaluate one variant on non-overlapping horizon-sized windows.")
    parser.add_argument("--dataset", default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--neighbor-k", type=int, default=7)
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--use_diff_dora", action="store_true")
    parser.add_argument("--prompt_style", default="cot", choices=["auto", "cot", "direct_physical", "vanilla"])
    parser.add_argument("--retrieval_cache", required=True)
    parser.add_argument("--expert_0_dir", required=True)
    parser.add_argument("--expert_1_dir", required=True)
    parser.add_argument("--cbd_nodes", required=True)
    parser.add_argument("--res_nodes", required=True)
    parser.add_argument("--window_count", type=int, default=48,
                        help="Number of non-overlapping windows; 48 x 6 steps = 24 hours at 5-minute resolution.")
    parser.add_argument("--infer_batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = _load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    router = HardRouter(build_routing_labels(splits["train"], raw.get("node_meta")))
    retriever = KNNRetriever.load(args.retrieval_cache) if args.use_rag else None

    test_samples = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
    )[args.horizon]
    test_ts = splits["timestamps_test"]
    train_len = splits["train"].shape[0]
    val_len = splits["val"].shape[0]
    test_occ_raw = raw["occupancy_raw"][train_len + val_len :]

    sample_indices = list(range(0, min(len(test_samples), args.window_count * args.horizon), args.horizon))
    sample_indices = sample_indices[: args.window_count]

    model_0, tokenizer, base_0 = _load_expert(args.expert_0_dir)
    model_1, _, base_1 = _load_expert(args.expert_1_dir)
    experts = {0: model_0, 1: model_1}
    selected_nodes = {
        "CBD": _parse_node_csv(args.cbd_nodes),
        "Residential": _parse_node_csv(args.res_nodes),
    }

    payload = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "history_len": args.history_len,
        "time_granularity_minutes": 5,
        "sample_indices": sample_indices,
        "domains": {"CBD": {}, "Residential": {}},
    }

    for domain, node_indices in selected_nodes.items():
        for node_idx in node_indices:
            expert_id = router.route(node_idx)
            model = experts[expert_id]
            static_context = extract_node_static_context(
                node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )

            jobs = []
            truth = []
            concat_timestamps = []
            for sample_idx in sample_indices:
                sample = test_samples[sample_idx]
                sys_msg, usr_msg = _build_prompt(
                    sample=sample,
                    node_idx=node_idx,
                    horizon=args.horizon,
                    prompt_style=args.prompt_style,
                    use_rag=args.use_rag,
                    use_diff_dora=args.use_diff_dora,
                    retriever=retriever,
                    static_context=static_context,
                    domain_label=domain,
                    splits=splits,
                )
                jobs.append((sys_msg, usr_msg))
                target = sample["y"][:args.horizon, node_idx] * (raw["norm_max"] - raw["norm_min"]) + raw["norm_min"]
                truth.extend(target.tolist())
                start = int(sample["t_start"])
                concat_timestamps.extend([str(ts) for ts in test_ts[start:start + args.horizon]])

            preds = []
            for chunk in _batched(jobs, args.infer_batch_size):
                outputs = generate_batch(
                    model,
                    tokenizer,
                    chunk,
                    max_new_tokens=args.max_new_tokens,
                )
                for text in outputs:
                    arr = parse_output(text, expected_len=args.horizon)
                    if arr is None or len(arr) < args.horizon:
                        preds.extend([None] * args.horizon)
                    else:
                        pred = arr[:args.horizon] * (raw["norm_max"] - raw["norm_min"]) + raw["norm_min"]
                        preds.extend(pred.tolist())

            valid_pairs = [(p, t) for p, t in zip(preds, truth) if p is not None]
            if valid_pairs:
                pred_arr = np.array([p for p, _ in valid_pairs], dtype=np.float32)
                true_arr = np.array([t for _, t in valid_pairs], dtype=np.float32)
                mae = float(np.mean(np.abs(pred_arr - true_arr)))
                rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
            else:
                mae = math.nan
                rmse = math.nan

            payload["domains"][domain][str(node_idx)] = {
                "node_idx": node_idx,
                "expert_id": expert_id,
                "timestamps": concat_timestamps,
                "truth": truth,
                "prediction": preds,
                "parse_success_rate": len(valid_pairs) / max(len(truth), 1),
                "metrics": {"mae": mae, "rmse": rmse},
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    del model_0, model_1, base_0, base_1, tokenizer, retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
