from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.data.build_samples import _neighbour_features, _time_features
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


def _build_inference_windows(
    occ: np.ndarray,
    timestamps,
    adj: np.ndarray | None,
    *,
    history_len: int,
    neighbor_k: int,
) -> list[dict]:
    time_feats = _time_features(timestamps)
    nbr_feats = _neighbour_features(occ, adj, neighbor_k=neighbor_k)

    windows = []
    t_end = occ.shape[0]
    for t in range(history_len, t_end):
        windows.append({
            "x_hist": occ[t - history_len:t],
            "time_feat": time_feats[t - history_len:t],
            "nbr_feat": nbr_feats[t - history_len:t],
            "t_start": t,
        })
    return windows


def _batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start:start + max(1, batch_size)]


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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one weekly-curve variant on a single visible GPU."
    )
    parser.add_argument("--dataset", default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--use_diff_dora", action="store_true")
    parser.add_argument("--prompt_style", default="cot", choices=["auto", "cot", "direct_physical", "vanilla"])
    parser.add_argument("--retrieval_cache", required=True)
    parser.add_argument("--expert_0_dir", required=True)
    parser.add_argument("--expert_1_dir", required=True)
    parser.add_argument("--cbd_nodes", required=True,
                        help="Comma-separated CBD node indices.")
    parser.add_argument("--res_nodes", required=True,
                        help="Comma-separated Residential node indices.")
    parser.add_argument("--max_points", type=int, default=288,
                        help="Number of rolling one-step predictions to keep; 288 = 24 hours at 5-minute resolution.")
    parser.add_argument("--infer_batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = _load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    router = HardRouter(build_routing_labels(splits["train"], raw.get("node_meta")))
    retriever = KNNRetriever.load(args.retrieval_cache) if args.use_rag else None

    test_occ = splits["test"]
    train_len = splits["train"].shape[0]
    val_len = splits["val"].shape[0]
    test_occ_raw = raw["occupancy_raw"][train_len + val_len :]
    test_ts = splits["timestamps_test"]
    adj = splits.get("adj")
    windows = _build_inference_windows(
        test_occ,
        test_ts,
        adj,
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
    )
    if args.max_points > 0:
        windows = windows[:args.max_points]
        eval_timestamps = [str(ts) for ts in test_ts[args.history_len:args.history_len + args.max_points]]
    else:
        eval_timestamps = [str(ts) for ts in test_ts[args.history_len:]]

    model_0, tokenizer, base_0 = _load_expert(args.expert_0_dir)
    model_1, _, base_1 = _load_expert(args.expert_1_dir)
    experts = {0: model_0, 1: model_1}
    cbd_nodes = _parse_node_csv(args.cbd_nodes)
    res_nodes = _parse_node_csv(args.res_nodes)

    outputs = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "history_len": args.history_len,
        "time_granularity_minutes": 5,
        "window_count": len(windows),
        "timestamps": eval_timestamps,
        "domains": {
            "CBD": {},
            "Residential": {},
        },
    }

    for domain, node_group in (("CBD", cbd_nodes), ("Residential", res_nodes)):
        for node_idx in node_group:
            expert_id = router.route(node_idx)
            model = experts[expert_id]
            static_context = extract_node_static_context(
                node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )

            jobs = []
            truth = []
            for window in windows:
                sys_msg, usr_msg = _build_prompt(
                    sample=window,
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
                jobs.append({"prompt": (sys_msg, usr_msg), "t_start": int(window["t_start"])})
                truth.append(float(test_occ_raw[int(window["t_start"]), node_idx]))

            preds = []
            parse_fail_indices = []
            for chunk in _batched(jobs, args.infer_batch_size):
                prompts = [job["prompt"] for job in chunk]
                outputs_text = generate_batch(
                    model,
                    tokenizer,
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                )
                for job, text in zip(chunk, outputs_text):
                    arr = parse_output(text, expected_len=args.horizon)
                    if arr is None or len(arr) == 0:
                        preds.append(None)
                        parse_fail_indices.append(job["t_start"])
                    else:
                        pred_value = float(arr[0]) * (raw["norm_max"] - raw["norm_min"]) + raw["norm_min"]
                        preds.append(pred_value)

            valid_pairs = [(p, t) for p, t in zip(preds, truth) if p is not None]
            if valid_pairs:
                pred_arr = np.array([p for p, _ in valid_pairs], dtype=np.float32)
                true_arr = np.array([t for _, t in valid_pairs], dtype=np.float32)
                mae = float(np.mean(np.abs(pred_arr - true_arr)))
                rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
            else:
                mae = math.nan
                rmse = math.nan

            outputs["domains"][domain][str(node_idx)] = {
                "node_idx": node_idx,
                "expert_id": expert_id,
                "truth": truth,
                "prediction": preds,
                "parse_fail_t_starts": parse_fail_indices,
                "parse_success_rate": (len(valid_pairs) / max(len(truth), 1)),
                "metrics": {"mae": mae, "rmse": rmse},
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    del model_0, model_1, base_0, base_1, tokenizer, retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
