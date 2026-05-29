from __future__ import annotations

import argparse
import gc
import json
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


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate selected sliding-window cases for one variant.")
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
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--infer_batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = _load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    router = HardRouter(build_routing_labels(splits["train"], raw.get("node_meta")))
    retriever = KNNRetriever.load(args.retrieval_cache) if args.use_rag else None
    selection = _load_json(Path(args.selection_json))
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

    model_0, tokenizer, base_0 = _load_expert(args.expert_0_dir)
    model_1, _, base_1 = _load_expert(args.expert_1_dir)
    experts = {0: model_0, 1: model_1}

    requested = []
    for domain, rows in selection["single_windows"].items():
        for row in rows:
            requested.append((domain, int(row["node_idx"]), int(row["sample_dataset_index"]), "single"))
    for domain, cfg in selection["collage_windows"].items():
        for node_idx in cfg["node_indices"]:
            for sample_idx in cfg["window_indices"]:
                requested.append((domain, int(node_idx), int(sample_idx), "collage"))

    unique = {}
    for domain, node_idx, sample_idx, _ in requested:
        unique[(domain, node_idx, sample_idx)] = True

    jobs = []
    meta = []
    for domain, node_idx, sample_idx in unique.keys():
        sample = test_samples[sample_idx]
        static_context = extract_node_static_context(
            node_idx,
            node_ids=splits.get("node_ids"),
            node_meta=splits.get("node_meta"),
        )
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
        expert_id = router.route(node_idx)
        history = (sample["x_hist"][:, node_idx] * (raw["norm_max"] - raw["norm_min"]) + raw["norm_min"]).tolist()
        target = (sample["y"][:args.horizon, node_idx] * (raw["norm_max"] - raw["norm_min"]) + raw["norm_min"]).tolist()
        history_times = [str(ts) for ts in test_ts[sample["t_start"] - args.history_len: sample["t_start"]]]
        future_times = [str(ts) for ts in test_ts[sample["t_start"]: sample["t_start"] + args.horizon]]
        jobs.append((expert_id, (sys_msg, usr_msg)))
        meta.append({
            "domain": domain,
            "node_idx": node_idx,
            "sample_dataset_index": sample_idx,
            "t_start": int(sample["t_start"]),
            "history": history,
            "target": target,
            "history_timestamps": history_times,
            "future_timestamps": future_times,
        })

    preds = [None] * len(jobs)
    for expert_id in (0, 1):
        indices = [i for i, (eid, _) in enumerate(jobs) if eid == expert_id]
        if not indices:
            continue
        prompts = [jobs[i][1] for i in indices]
        for start in range(0, len(prompts), max(1, args.infer_batch_size)):
            batch_idx = indices[start:start + max(1, args.infer_batch_size)]
            batch_prompts = [jobs[i][1] for i in batch_idx]
            outputs = generate_batch(
                experts[expert_id],
                tokenizer,
                batch_prompts,
                max_new_tokens=args.max_new_tokens,
            )
            for row_idx, text in zip(batch_idx, outputs):
                arr = parse_output(text, expected_len=args.horizon)
                if arr is None or len(arr) == 0:
                    preds[row_idx] = None
                else:
                    pred = arr[:args.horizon] * (raw["norm_max"] - raw["norm_min"]) + raw["norm_min"]
                    preds[row_idx] = pred.tolist()

    payload = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "history_len": args.history_len,
        "use_diff_dora": args.use_diff_dora,
        "cases": [],
    }
    for info, pred in zip(meta, preds):
        payload["cases"].append({**info, "prediction": pred})

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
