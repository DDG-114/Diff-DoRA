"""
Scenario-based zero-shot forecasting for the Wotai dataset.

This script is intended for qualitative demonstrations of LLM forecasting
behaviour. It selects semantically meaningful actual-load windows from Wotai
data, prompts an existing expert adapter with RAG/Diff-DoRA context, and writes
one JSONL record per prediction so partial overnight runs remain useful.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.loaders import load_dataset
from src.data.windowing import resolve_window_stride
from src.eval.metrics import denormalize
from src.models.qwen_peft import load_model_and_tokenizer, load_peft_model, generate
from src.prompts.parser import parse_output
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.knn_retriever import KNNRetriever
from src.utils.history_window import price_at_history_end, weather_at_history_end
from src.utils.node_context import extract_node_static_context, resolve_node_id

DEFAULT_EXPERT_0 = "outputs/full_repro_st_evcdp_h6_bs48/full/expert_0/adapter"
DEFAULT_EXPERT_1 = "outputs/full_repro_st_evcdp_h6_bs48/full/expert_1/adapter"
SCENARIOS = (
    "low_valley",
    "rapid_rise",
    "peak_plateau",
    "rapid_drop",
    "high_volatility",
    "weather_shift",
)


def _weather_at(weather, sample: dict) -> dict | None:
    return weather_at_history_end(weather, sample)


def _price_at(price, sample: dict, node_idx: int, *, node_ids=None, node_meta=None) -> float | None:
    return price_at_history_end(price, sample, node_idx, node_ids=node_ids, node_meta=node_meta)


def _compute_diff(sample: dict, retrieved: list[dict], splits: dict, node_idx: int) -> dict:
    weather_current = _weather_at(splits.get("weather"), sample)
    weather_retrieved = [_weather_at(splits.get("weather"), rs) for rs in retrieved]
    price_current = _price_at(
        splits.get("price"),
        sample,
        node_idx,
        node_ids=splits.get("node_ids"),
        node_meta=splits.get("node_meta"),
    )
    price_retrieved = [
        _price_at(
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


def _score_sample(sample: dict, node_idx: int, weather) -> dict[str, float]:
    hist = sample["x_hist"][:, node_idx].astype(float)
    fut = sample["y"][:, node_idx].astype(float)
    combined = np.concatenate([hist, fut])
    short_slope = float(hist[-1] - hist[-4])
    future_slope = float(fut[-1] - fut[0])
    t_start = int(sample.get("t_start", 0))
    temp_delta = 0.0
    if weather is not None and not getattr(weather, "empty", True) and "temperature" in weather.columns:
        start_idx = min(max(t_start, 0), len(weather) - 1)
        end_idx = min(max(t_start + len(fut) - 1, 0), len(weather) - 1)
        temp_delta = float(weather["temperature"].iloc[end_idx] - weather["temperature"].iloc[start_idx])
    return {
        "hist_mean": float(hist.mean()),
        "hist_last": float(hist[-1]),
        "future_mean": float(fut.mean()),
        "short_slope": short_slope,
        "future_slope": future_slope,
        "volatility": float(combined.std()),
        "peak_level": float(combined.max()),
        "valley_level": float(combined.min()),
        "weather_shift": abs(temp_delta),
    }


def _rank(values: list[tuple[int, dict]], key: str, *, reverse: bool = True) -> list[int]:
    return [
        idx
        for idx, _ in sorted(
            values,
            key=lambda item: (item[1][key], -item[0] if reverse else item[0]),
            reverse=reverse,
        )
    ]


def select_scenarios(samples: list[dict], node_idx: int, weather, per_scenario: int) -> list[tuple[str, int]]:
    scored = [(idx, _score_sample(sample, node_idx, weather)) for idx, sample in enumerate(samples)]
    ranked = {
        "low_valley": _rank(scored, "hist_mean", reverse=False),
        "rapid_rise": _rank(scored, "future_slope", reverse=True),
        "peak_plateau": _rank(scored, "peak_level", reverse=True),
        "rapid_drop": _rank(scored, "future_slope", reverse=False),
        "high_volatility": _rank(scored, "volatility", reverse=True),
        "weather_shift": _rank(scored, "weather_shift", reverse=True),
    }
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for scenario in SCENARIOS:
        count = 0
        for idx in ranked[scenario]:
            if idx in used:
                continue
            selected.append((scenario, idx))
            used.add(idx)
            count += 1
            if count >= per_scenario:
                break
    return selected


def _round_list(arr, ndigits: int = 3) -> list[float]:
    return [round(float(v), ndigits) for v in np.asarray(arr).reshape(-1)]


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(math.sqrt(np.mean((pred - true) ** 2)))


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_summary(path: Path, records: list[dict], args) -> None:
    parsed = [record for record in records if record["parse_ok"]]
    summary = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "node_id": args.node_id,
        "node_idx": args.node_idx,
        "records": len(records),
        "parsed": len(parsed),
        "parse_success_rate": len(parsed) / max(len(records), 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "by_scenario": {},
    }
    if parsed:
        summary["overall_mae"] = float(np.mean([record["mae"] for record in parsed]))
        summary["overall_rmse"] = float(np.mean([record["rmse"] for record in parsed]))
    for scenario in SCENARIOS:
        rows = [record for record in parsed if record["scenario"] == scenario]
        summary["by_scenario"][scenario] = {
            "records": len([record for record in records if record["scenario"] == scenario]),
            "parsed": len(rows),
            "mae": None if not rows else float(np.mean([record["mae"] for record in rows])),
            "rmse": None if not rows else float(np.mean([record["rmse"] for record in rows])),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wotai_evcdp")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--node_id", default="actual_load")
    parser.add_argument("--node_idx", type=int, default=-1)
    parser.add_argument("--expert_0_dir", default=DEFAULT_EXPERT_0)
    parser.add_argument("--expert_1_dir", default=DEFAULT_EXPERT_1)
    parser.add_argument("--retrieval_cache", default="data/retrieval_cache/wotai_evcdp_h6_step6.pkl")
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--window_stride", type=int, default=6)
    parser.add_argument("--per_scenario", type=int, default=12)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--use_rag", action="store_true", default=True)
    parser.add_argument("--no_rag", dest="use_rag", action="store_false")
    parser.add_argument("--use_diff_dora", action="store_true", default=True)
    parser.add_argument("--no_diff_dora", dest="use_diff_dora", action="store_false")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_json", default="outputs/wotai_scenario_forecast/summary.json")
    parser.add_argument("--output_jsonl", default="outputs/wotai_scenario_forecast/predictions.jsonl")
    args = parser.parse_args()

    args.window_stride = resolve_window_stride(args.window_stride, horizon=args.horizon)
    raw = load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    node_ids = [str(node_id) for node_id in splits.get("node_ids") or raw.get("node_ids") or []]
    if args.node_idx >= 0:
        node_idx = int(args.node_idx)
    else:
        node_idx = node_ids.index(args.node_id) if args.node_id in node_ids else 0
    args.node_idx = node_idx
    args.node_id = str(resolve_node_id(node_idx, node_ids=node_ids, node_meta=splits.get("node_meta")))

    split_key = args.split
    sample_map = build_samples(
        splits[split_key],
        splits[f"timestamps_{split_key}"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
        window_stride=args.window_stride,
    )
    samples = sample_map[args.horizon]
    selected = select_scenarios(samples, node_idx, splits.get("weather"), args.per_scenario)
    if args.max_cases > 0:
        selected = selected[: args.max_cases]

    retriever = None
    if args.use_rag:
        retriever = KNNRetriever.load(args.retrieval_cache)
        print(f"Loaded retrieval cache: {args.retrieval_cache}")

    adapter_path = args.expert_0_dir
    domain_label = "CBD"
    print(f"Loading expert adapter for node {args.node_id}: {adapter_path}")
    base_model, tokenizer = load_model_and_tokenizer()
    model = load_peft_model(base_model, adapter_path)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    jsonl_path = Path(args.output_jsonl)
    json_path = Path(args.output_json)
    if jsonl_path.exists():
        jsonl_path.unlink()

    records = []
    norm_min = float(splits["norm_min"])
    norm_max = float(splits["norm_max"])
    print(f"Selected {len(selected)} scenario cases.")
    for case_id, (scenario, sample_idx) in enumerate(selected, start=1):
        sample = samples[sample_idx]
        retrieved = retriever.query(sample, exclude_t_start=None) if retriever is not None else []
        diff = _compute_diff(sample, retrieved, splits, node_idx) if retrieved else None
        static_context = extract_node_static_context(
            node_idx,
            node_ids=splits.get("node_ids"),
            node_meta=splits.get("node_meta"),
        )
        if args.use_rag and retrieved:
            sys_msg, usr_msg = build_cot_prompt(
                sample,
                retrieved,
                diff,
                node_idx=node_idx,
                horizon=args.horizon,
                domain_label=domain_label,
                static_context=static_context,
                include_env_diff=args.use_diff_dora,
            )
        else:
            sys_msg, usr_msg = build_vanilla_prompt(
                sample,
                node_idx=node_idx,
                horizon=args.horizon,
                domain_label=domain_label,
                static_context=static_context,
            )

        raw_output = generate(
            model,
            tokenizer,
            sys_msg,
            usr_msg,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        parsed = parse_output(raw_output, expected_len=args.horizon)
        parse_ok = parsed is not None and len(parsed) == args.horizon

        hist_norm = sample["x_hist"][:, node_idx]
        target_norm = sample["y"][: args.horizon, node_idx]
        hist_raw = denormalize(hist_norm, norm_min, norm_max)
        target_raw = denormalize(target_norm, norm_min, norm_max)
        record = {
            "case_id": case_id,
            "scenario": scenario,
            "sample_idx": int(sample_idx),
            "t_start": int(sample.get("t_start", -1)),
            "node_idx": node_idx,
            "node_id": args.node_id,
            "parse_ok": bool(parse_ok),
            "diff": diff,
            "history_norm": _round_list(hist_norm),
            "target_norm": _round_list(target_norm),
            "history_raw": _round_list(hist_raw, 2),
            "target_raw": _round_list(target_raw, 2),
            "retrieved_t_starts": [int(rs.get("t_start", -1)) for rs in retrieved],
            "system_prompt": sys_msg,
            "user_prompt": usr_msg,
            "raw_output": raw_output,
        }
        if parse_ok:
            pred_norm = np.asarray(parsed, dtype=np.float32)
            pred_raw = denormalize(pred_norm, norm_min, norm_max)
            record.update(
                {
                    "prediction_norm": _round_list(pred_norm),
                    "prediction_raw": _round_list(pred_raw, 2),
                    "mae": _mae(pred_raw, target_raw),
                    "rmse": _rmse(pred_raw, target_raw),
                }
            )
        else:
            record.update({"prediction_norm": None, "prediction_raw": None, "mae": None, "rmse": None})

        records.append(record)
        _append_jsonl(jsonl_path, record)
        _write_summary(json_path, records, args)
        status = "ok" if parse_ok else "parse_fail"
        metric = "" if not parse_ok else f" mae={record['mae']:.2f} rmse={record['rmse']:.2f}"
        print(f"[{case_id:03d}/{len(selected):03d}] {scenario:15s} {status}{metric}")

    _write_summary(json_path, records, args)
    print(f"Saved JSONL: {jsonl_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
