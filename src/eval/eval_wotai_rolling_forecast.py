"""
Rolling-window forecast evaluation for Wotai.

The script evaluates a continuous time segment instead of isolated random
windows. It selects two CBD and two Residential nodes by default, runs routed
expert inference in batches, and writes metrics plus per-window records.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.loaders import load_dataset
from src.data.windowing import default_retrieval_cache_path, resolve_window_stride
from src.eval.metrics import denormalize
from src.models.qwen_peft import generate_batch, load_model_and_tokenizer, load_peft_model
from src.prompts.parser import parse_output
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.prompt_vanilla import build_direct_physical_prompt, build_vanilla_prompt
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.knn_retriever import KNNRetriever
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router import HardRouter
from src.utils.history_window import weather_at_history_end
from src.utils.node_context import extract_node_static_context, resolve_node_id

DEFAULT_EXPERT_0 = "outputs/full_repro_st_evcdp_h6_bs48/full/expert_0/adapter"
DEFAULT_EXPERT_1 = "outputs/full_repro_st_evcdp_h6_bs48/full/expert_1/adapter"


def _weather_at(weather, sample: dict) -> dict | None:
    return weather_at_history_end(weather, sample)


def _compute_diff(sample: dict, retrieved: list[dict], splits: dict, node_idx: int) -> dict:
    weather_current = _weather_at(splits.get("weather"), sample)
    weather_retrieved = [_weather_at(splits.get("weather"), rs) for rs in retrieved]
    return compute_diff_features(
        query_sample=sample,
        retrieved_samples=retrieved,
        weather_current=weather_current,
        weather_retrieved=weather_retrieved,
        price_current=None,
        price_retrieved=None,
        node_idx=node_idx,
    )


def _round_list(values, ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in np.asarray(values).reshape(-1)]


def _select_nodes(router: HardRouter, n_per_domain: int) -> tuple[list[int], dict[str, list[int]]]:
    cbd = router.nodes_for_expert(0)[:n_per_domain]
    residential = router.nodes_for_expert(1)[:n_per_domain]
    nodes = cbd + residential
    if len(cbd) < n_per_domain or len(residential) < n_per_domain:
        print(
            f"[WARN] Requested {n_per_domain} nodes per domain, got "
            f"CBD={len(cbd)} Residential={len(residential)}."
        )
    return nodes, {"CBD": cbd, "Residential": residential}


def _filter_by_time(samples: list[dict], timestamps, args) -> list[tuple[int, dict]]:
    indexed = []
    for idx, sample in enumerate(samples):
        forecast_start_idx = int(sample.get("t_start", 0))
        if forecast_start_idx < 0 or forecast_start_idx >= len(timestamps):
            continue
        ts = timestamps[forecast_start_idx]
        if args.start_time and str(ts) < args.start_time:
            continue
        if args.end_time and str(ts) > args.end_time:
            continue
        indexed.append((idx, sample))
    if args.max_windows > 0:
        indexed = indexed[: args.max_windows]
    return indexed


def _build_prompt(
    *,
    sample: dict,
    node_idx: int,
    horizon: int,
    expert_domain: str,
    splits: dict,
    retriever: KNNRetriever | None,
    prompt_style: str,
    use_diff_dora: bool,
) -> tuple[str, str, dict | None, list[int]]:
    retrieved = retriever.query(sample, exclude_t_start=None) if retriever is not None else []
    diff = _compute_diff(sample, retrieved, splits, node_idx) if retrieved else None
    static_context = extract_node_static_context(
        node_idx,
        node_ids=splits.get("node_ids"),
        node_meta=splits.get("node_meta"),
    )
    include_env_diff = bool(use_diff_dora and retrieved)
    if prompt_style == "cot" and retrieved:
        sys_msg, usr_msg = build_cot_prompt(
            sample,
            retrieved,
            diff,
            node_idx=node_idx,
            horizon=horizon,
            domain_label=expert_domain,
            static_context=static_context,
            include_env_diff=include_env_diff,
        )
    elif prompt_style == "direct_physical" and retrieved:
        sys_msg, usr_msg = build_direct_physical_prompt(
            sample,
            retrieved,
            diff,
            node_idx=node_idx,
            horizon=horizon,
            domain_label=expert_domain,
            static_context=static_context,
            include_env_diff=include_env_diff,
        )
    else:
        sys_msg, usr_msg = build_vanilla_prompt(
            sample,
            node_idx=node_idx,
            horizon=horizon,
            domain_label=expert_domain,
            static_context=static_context,
        )
    return sys_msg, usr_msg, diff, [int(rs.get("t_start", -1)) for rs in retrieved]


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(math.sqrt(np.mean((pred - true) ** 2)))


def _direction_accuracy(pred: np.ndarray, true: np.ndarray, history_last: float) -> float:
    pred_path = np.concatenate([[history_last], pred])
    true_path = np.concatenate([[history_last], true])
    pred_dir = np.sign(np.diff(pred_path))
    true_dir = np.sign(np.diff(true_path))
    return float(np.mean(pred_dir == true_dir))


def _peak_time_error(pred: np.ndarray, true: np.ndarray) -> int:
    return int(np.argmax(pred) - np.argmax(true))


def _aggregate_metrics(records: list[dict]) -> dict:
    parsed = [row for row in records if row["parse_ok"]]
    out = {
        "records": len(records),
        "parsed": len(parsed),
        "parse_success_rate": len(parsed) / max(len(records), 1),
    }
    if not parsed:
        return out
    out.update(
        {
            "mae": float(np.mean([row["mae"] for row in parsed])),
            "rmse": float(np.mean([row["rmse"] for row in parsed])),
            "direction_accuracy": float(np.mean([row["direction_accuracy"] for row in parsed])),
            "peak_abs_error": float(np.mean([row["peak_abs_error"] for row in parsed])),
            "peak_time_abs_error_steps": float(np.mean([abs(row["peak_time_error_steps"]) for row in parsed])),
        }
    )
    by_domain = {}
    for domain in ("CBD", "Residential"):
        rows = [row for row in parsed if row["expert_domain"] == domain]
        by_domain[domain] = {
            "records": len([row for row in records if row["expert_domain"] == domain]),
            "parsed": len(rows),
            "mae": None if not rows else float(np.mean([row["mae"] for row in rows])),
            "rmse": None if not rows else float(np.mean([row["rmse"] for row in rows])),
            "direction_accuracy": None if not rows else float(np.mean([row["direction_accuracy"] for row in rows])),
        }
    by_node = {}
    for node_id in sorted({row["node_id"] for row in records}):
        rows = [row for row in parsed if row["node_id"] == node_id]
        by_node[node_id] = {
            "records": len([row for row in records if row["node_id"] == node_id]),
            "parsed": len(rows),
            "mae": None if not rows else float(np.mean([row["mae"] for row in rows])),
            "rmse": None if not rows else float(np.mean([row["rmse"] for row in rows])),
            "direction_accuracy": None if not rows else float(np.mean([row["direction_accuracy"] for row in rows])),
        }
    out["by_domain"] = by_domain
    out["by_node"] = by_node
    return out


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wotai_evcdp")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--window_stride", type=int, default=6)
    parser.add_argument("--start_time", default="", help="Forecast-start lower bound, e.g. 2025-06-26 00:00:00")
    parser.add_argument("--end_time", default="", help="Forecast-start upper bound, e.g. 2025-06-26 23:45:00")
    parser.add_argument("--max_windows", type=int, default=96, help="Cap rolling windows after time filtering.")
    parser.add_argument("--nodes_per_domain", type=int, default=2)
    parser.add_argument("--expert_0_dir", default=DEFAULT_EXPERT_0)
    parser.add_argument("--expert_1_dir", default=DEFAULT_EXPERT_1)
    parser.add_argument("--retrieval_cache", default="")
    parser.add_argument("--prompt_style", choices=["cot", "direct_physical", "vanilla"], default="direct_physical")
    parser.add_argument("--use_diff_dora", action="store_true", default=True)
    parser.add_argument("--no_diff_dora", dest="use_diff_dora", action="store_false")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--output", default="outputs/wotai_rolling_forecast/rolling_forecast.json")
    args = parser.parse_args()

    args.window_stride = resolve_window_stride(args.window_stride, horizon=args.horizon)
    raw = load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    labels = build_routing_labels(splits["train"], raw.get("node_meta"))
    router = HardRouter(labels)
    selected_nodes, nodes_by_domain = _select_nodes(router, args.nodes_per_domain)
    node_ids = [str(node_id) for node_id in splits.get("node_ids") or raw.get("node_ids") or []]

    sample_map = build_samples(
        splits[args.split],
        splits[f"timestamps_{args.split}"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
        window_stride=args.window_stride,
    )
    samples = sample_map[args.horizon]
    indexed_samples = _filter_by_time(samples, splits[f"timestamps_{args.split}"], args)
    if not indexed_samples:
        raise ValueError("No rolling samples selected. Check --start_time/--end_time/--max_windows.")

    retrieval_cache = args.retrieval_cache or str(default_retrieval_cache_path(args.dataset, args.horizon, args.window_stride))
    retriever = None if args.prompt_style == "vanilla" else KNNRetriever.load(retrieval_cache)
    if retriever is not None:
        print(f"Loaded retrieval cache: {retrieval_cache}")

    print(f"Selected nodes by domain: {nodes_by_domain}")
    print(f"Rolling windows: {len(indexed_samples)}; total inferences={len(indexed_samples) * len(selected_nodes)}")

    base_model_0, tokenizer = load_model_and_tokenizer()
    experts = {0: load_peft_model(base_model_0, args.expert_0_dir)}
    base_model_1, _ = load_model_and_tokenizer()
    experts[1] = load_peft_model(base_model_1, args.expert_1_dir)
    for model in experts.values():
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True

    jobs_by_expert: dict[int, list[dict]] = defaultdict(list)
    for sample_idx, sample in indexed_samples:
        forecast_start_idx = int(sample.get("t_start", 0))
        forecast_start = str(splits[f"timestamps_{args.split}"][forecast_start_idx])
        for node_idx in selected_nodes:
            expert_id = router.route(node_idx)
            domain = "CBD" if expert_id == 0 else "Residential"
            sys_msg, usr_msg, diff, retrieved_t_starts = _build_prompt(
                sample=sample,
                node_idx=node_idx,
                horizon=args.horizon,
                expert_domain=domain,
                splits=splits,
                retriever=retriever,
                prompt_style=args.prompt_style,
                use_diff_dora=args.use_diff_dora,
            )
            jobs_by_expert[expert_id].append(
                {
                    "sample_idx": sample_idx,
                    "t_start": forecast_start_idx,
                    "forecast_start": forecast_start,
                    "node_idx": node_idx,
                    "node_id": str(resolve_node_id(node_idx, node_ids=node_ids, node_meta=splits.get("node_meta"))),
                    "expert_id": expert_id,
                    "expert_domain": domain,
                    "sample": sample,
                    "system_prompt": sys_msg,
                    "user_prompt": usr_msg,
                    "diff": diff,
                    "retrieved_t_starts": retrieved_t_starts,
                }
            )

    records: list[dict] = []
    norm_min = float(splits["norm_min"])
    norm_max = float(splits["norm_max"])
    output_path = Path(args.output)
    completed = 0
    requested = sum(len(v) for v in jobs_by_expert.values())

    for expert_id in (0, 1):
        model = experts[expert_id]
        jobs = jobs_by_expert.get(expert_id, [])
        for start in range(0, len(jobs), max(1, args.batch_size)):
            chunk = jobs[start : start + max(1, args.batch_size)]
            prompts = [(job["system_prompt"], job["user_prompt"]) for job in chunk]
            outputs = generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            for job, raw_output in zip(chunk, outputs):
                sample = job["sample"]
                node_idx = int(job["node_idx"])
                target_norm = sample["y"][: args.horizon, node_idx]
                history_norm = sample["x_hist"][:, node_idx]
                target_raw = denormalize(target_norm, norm_min, norm_max)
                history_raw = denormalize(history_norm, norm_min, norm_max)
                parsed = parse_output(raw_output, expected_len=args.horizon)
                parse_ok = parsed is not None and len(parsed) == args.horizon
                row = {
                    "sample_idx": int(job["sample_idx"]),
                    "t_start": int(job["t_start"]),
                    "forecast_start": job["forecast_start"],
                    "node_idx": node_idx,
                    "node_id": job["node_id"],
                    "expert_id": int(job["expert_id"]),
                    "expert_domain": job["expert_domain"],
                    "parse_ok": bool(parse_ok),
                    "history_raw": _round_list(history_raw, 2),
                    "target_raw": _round_list(target_raw, 2),
                    "history_norm": _round_list(history_norm),
                    "target_norm": _round_list(target_norm),
                    "diff": job["diff"],
                    "retrieved_t_starts": job["retrieved_t_starts"],
                    "raw_output": raw_output,
                }
                if args.save_prompts:
                    row["system_prompt"] = job["system_prompt"]
                    row["user_prompt"] = job["user_prompt"]
                if parse_ok:
                    pred_norm = np.asarray(parsed, dtype=np.float32)
                    pred_raw = denormalize(pred_norm, norm_min, norm_max)
                    row.update(
                        {
                            "prediction_norm": _round_list(pred_norm),
                            "prediction_raw": _round_list(pred_raw, 2),
                            "mae": _mae(pred_raw, target_raw),
                            "rmse": _rmse(pred_raw, target_raw),
                            "direction_accuracy": _direction_accuracy(pred_raw, target_raw, float(history_raw[-1])),
                            "peak_abs_error": float(abs(np.max(pred_raw) - np.max(target_raw))),
                            "peak_time_error_steps": _peak_time_error(pred_raw, target_raw),
                        }
                    )
                else:
                    row.update(
                        {
                            "prediction_norm": None,
                            "prediction_raw": None,
                            "mae": None,
                            "rmse": None,
                            "direction_accuracy": None,
                            "peak_abs_error": None,
                            "peak_time_error_steps": None,
                        }
                    )
                records.append(row)
                completed += 1

            payload = {
                "summary": {
                    "dataset": args.dataset,
                    "split": args.split,
                    "horizon": args.horizon,
                    "history_len": args.history_len,
                    "window_stride": args.window_stride,
                    "start_time": args.start_time,
                    "end_time": args.end_time,
                    "selected_nodes": selected_nodes,
                    "nodes_by_domain": nodes_by_domain,
                    "requested": requested,
                    "completed": completed,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": _aggregate_metrics(records),
                    "config": vars(args),
                },
                "records": records,
            }
            _write_output(output_path, payload)
            print(f"[{completed:04d}/{requested:04d}] expert_{expert_id} batch_done")

    payload = {
        "summary": {
            "dataset": args.dataset,
            "split": args.split,
            "horizon": args.horizon,
            "history_len": args.history_len,
            "window_stride": args.window_stride,
            "start_time": args.start_time,
            "end_time": args.end_time,
            "selected_nodes": selected_nodes,
            "nodes_by_domain": nodes_by_domain,
            "requested": requested,
            "completed": completed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": _aggregate_metrics(records),
            "config": vars(args),
        },
        "records": records,
    }
    _write_output(output_path, payload)
    print(f"Saved -> {output_path}")
    print(json.dumps(payload["summary"]["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
