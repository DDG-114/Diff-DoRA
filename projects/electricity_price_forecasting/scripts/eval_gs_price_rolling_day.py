#!/usr/bin/env python3
"""Autoregressive rolling-day evaluation for GS price adapters.

The adapter predicts a short horizon, e.g. 16 points = 4 hours at 15-minute
resolution. This script rolls that model forward 6 chunks to evaluate a full
96-point day without feeding future true prices back into the history window.
Known auxiliary variables remain available for each future chunk.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").exists() and (parent / "data").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_samples import _neighbour_features, _time_features
from src.data.build_splits import build_splits
from src.data.loaders import load_dataset
from src.eval.metrics import denormalize
from src.models.qwen_peft import generate, load_model_and_tokenizer, load_peft_model
from src.prompts.parser import parse_output
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.prompt_vanilla import build_direct_physical_prompt, build_vanilla_prompt
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.knn_retriever import KNNRetriever
from src.utils.history_window import price_at_history_end, weather_at_history_end
from src.utils.node_context import extract_node_static_context, normalise_domain_label
from src.utils.price_candidate import (
    attach_candidate_curve,
    attach_candidate_refine_mask,
    combine_candidate_prediction,
    load_daylevel_prediction_map,
)


def _build_train_pool(
    *,
    splits: dict,
    horizon: int,
    history_len: int,
    context_history_len: int,
    neighbor_k: int,
    window_stride: int,
) -> list[dict]:
    from src.data.build_samples import build_samples

    return build_samples(
        splits["train"],
        splits["timestamps_train"],
        adj=splits.get("adj"),
        aux_features=splits.get("weather"),
        horizons=[horizon],
        history_len=history_len,
        context_history_len=context_history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
    )[horizon]


def _sample_from_arrays(
    *,
    occ_roll: np.ndarray,
    timestamps,
    aux_values: np.ndarray | None,
    aux_columns: list[str] | None,
    adj: np.ndarray | None,
    t_start: int,
    horizon: int,
    history_len: int,
    context_history_len: int,
    neighbor_k: int,
) -> dict:
    time_feats = _time_features(timestamps)
    nbr_feats = _neighbour_features(occ_roll, adj, neighbor_k=neighbor_k)
    sample = {
        "x_hist": occ_roll[t_start - history_len : t_start],
        "time_feat": time_feats[t_start - history_len : t_start],
        "nbr_feat": nbr_feats[t_start - history_len : t_start],
        "x_context": occ_roll[t_start - context_history_len : t_start],
        "time_feat_context": time_feats[t_start - context_history_len : t_start],
        "nbr_feat_context": nbr_feats[t_start - context_history_len : t_start],
        "y": occ_roll[t_start : t_start + horizon],
        "t_start": int(t_start),
        "history_len": int(history_len),
        "context_history_len": int(context_history_len),
        "history_end_idx": int(t_start - 1),
        "history_end_timestamp": timestamps[t_start - 1],
        "target_start_timestamp": timestamps[t_start],
    }
    if aux_values is not None:
        sample["aux_hist"] = aux_values[t_start - history_len : t_start]
        sample["aux_context"] = aux_values[t_start - context_history_len : t_start]
        sample["aux_future"] = aux_values[t_start : t_start + horizon]
        sample["aux_columns"] = aux_columns
    return sample


def _compute_diff(sample: dict, retrieved: list[dict], splits: dict, node_idx: int) -> dict:
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


def _build_prompt(
    *,
    sample: dict,
    horizon: int,
    prompt_style: str,
    use_rag: bool,
    use_diff_dora: bool,
    retriever: KNNRetriever | None,
    splits: dict,
    node_idx: int,
) -> tuple[str, str, dict | None, list[int]]:
    static_context = extract_node_static_context(
        node_idx,
        node_ids=splits.get("node_ids"),
        node_meta=splits.get("node_meta"),
    )
    domain_label = normalise_domain_label(static_context.get("zone_type"))
    retrieved = retriever.query(sample, exclude_t_start=None) if use_rag and retriever is not None else []
    diff = _compute_diff(sample, retrieved, splits, node_idx) if retrieved else None
    include_env_diff = bool(use_diff_dora and retrieved)

    if prompt_style == "cot" and retrieved:
        sys_msg, usr_msg = build_cot_prompt(
            sample,
            retrieved,
            diff,
            node_idx=node_idx,
            horizon=horizon,
            domain_label=domain_label,
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
            domain_label=domain_label,
            static_context=static_context,
            include_env_diff=include_env_diff,
            target_mode="residual" if "candidate_future" in sample else "absolute",
        )
    else:
        sys_msg, usr_msg = build_vanilla_prompt(
            sample,
            node_idx=node_idx,
            horizon=horizon,
            domain_label=domain_label,
            static_context=static_context,
            target_mode="residual" if "candidate_future" in sample else "absolute",
        )
    return sys_msg, usr_msg, diff, [int(rs.get("t_start", -1)) for rs in retrieved]


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(math.sqrt(np.mean((pred - true) ** 2)))


def _daily_mean_accuracy(pred: np.ndarray, true: np.ndarray, *, floor_value: float = 40.0) -> float:
    pred_mean = float(np.mean(pred))
    true_mean = float(np.mean(true))
    return float(max(0.0, 1.0 - abs(pred_mean - true_mean) / max(abs(true_mean), floor_value)))


def _parse_with_optional_padding(raw_output: str, expected_len: int, *, allow_pad: bool) -> tuple[np.ndarray | None, bool]:
    parsed = parse_output(raw_output, expected_len=expected_len)
    if parsed is not None and len(parsed) == expected_len:
        return np.asarray(parsed, dtype=np.float32), False

    if not allow_pad:
        return None, False

    loose = parse_output(raw_output, expected_len=None)
    if loose is None or len(loose) == 0 or len(loose) > expected_len:
        return None, False
    padded = np.asarray(loose, dtype=np.float32)
    padded = np.pad(padded, (0, expected_len - len(padded)), mode="edge")
    return padded, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gs_price_2025")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--node_id", default="Price")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--day_horizon", type=int, default=96)
    parser.add_argument("--history_len", type=int, default=96)
    parser.add_argument("--context_history_len", type=int, default=1344)
    parser.add_argument("--neighbor_k", type=int, default=0)
    parser.add_argument("--day_stride", type=int, default=96)
    parser.add_argument("--retrieval_stride", type=int, default=16)
    parser.add_argument("--max_days", type=int, default=0)
    parser.add_argument("--prompt_style", choices=["cot", "direct_physical", "vanilla"], default="direct_physical")
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--use_diff_dora", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--allow_short_padding",
        action="store_true",
        help="If a chunk returns fewer than horizon values, repeat its last value to complete the chunk.",
    )
    parser.add_argument(
        "--candidate_prediction_csv",
        default="",
        help="Optional candidate day-level prediction CSV to inject as a baseline curve into each prompt chunk.",
    )
    parser.add_argument(
        "--candidate_mode",
        choices=["absolute", "residual", "selective_residual", "chunk_offset"],
        default="absolute",
        help="Whether parsed model outputs are absolute values or residuals relative to the candidate skeleton.",
    )
    parser.add_argument("--candidate_residual_clip", type=float, default=0.0,
                        help="If >0 and candidate_mode=residual, clip the residual correction magnitude to this value.")
    parser.add_argument("--candidate_value_min", type=float, default=0.0)
    parser.add_argument("--candidate_value_max", type=float, default=1000.0)
    parser.add_argument("--output", default="outputs/gs_price_rolling_h16_day96.json")
    args = parser.parse_args()

    if args.day_horizon % args.horizon != 0:
        raise ValueError("--day_horizon must be divisible by --horizon")
    if args.use_diff_dora and not args.use_rag:
        raise ValueError("--use_diff_dora requires --use_rag")
    if args.prompt_style in {"cot", "direct_physical"} and not args.use_rag:
        raise ValueError("--prompt_style cot/direct_physical requires --use_rag")

    raw = load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    candidate_prediction_map = None
    if args.candidate_prediction_csv:
        candidate_prediction_map = load_daylevel_prediction_map(args.candidate_prediction_csv)
    node_ids = [str(node_id) for node_id in splits.get("node_ids") or raw.get("node_ids") or []]
    if args.node_id not in node_ids:
        raise ValueError(f"--node_id {args.node_id!r} not found in node ids: {node_ids}")
    node_idx = node_ids.index(args.node_id)

    retriever = None
    if args.use_rag:
        train_pool = _build_train_pool(
            splits=splits,
            horizon=args.horizon,
            history_len=args.history_len,
            context_history_len=args.context_history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.retrieval_stride,
        )
        print(f"Building rolling retriever from train pool: {len(train_pool)}")
        retriever = KNNRetriever(train_pool, top_k=2, query_device="auto")

    test_occ = np.asarray(splits["test"], dtype=np.float32)
    timestamps = splits["timestamps_test"]
    weather = splits.get("weather")
    aux_frame = None
    if weather is not None and not getattr(weather, "empty", True):
        aux_frame = weather.reindex(timestamps).interpolate(method="time", limit_direction="both").ffill().bfill()
    aux_values = None if aux_frame is None else np.asarray(aux_frame, dtype=np.float32)
    aux_columns = None if aux_frame is None else [str(col) for col in aux_frame.columns]

    starts = list(range(args.context_history_len, len(test_occ) - args.day_horizon + 1, args.day_stride))
    if args.max_days > 0:
        starts = starts[: args.max_days]
    if not starts:
        raise ValueError("No rolling days selected.")

    print("Loading base model ...")
    base_model, tokenizer = load_model_and_tokenizer()
    print(f"Loading adapter ... {args.adapter_dir}")
    model = load_peft_model(base_model, args.adapter_dir)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    norm_min = float(splits.get("norm_min", 0.0))
    norm_max = float(splits.get("norm_max", 1.0))
    chunks_per_day = args.day_horizon // args.horizon
    records = []

    for day_idx, day_start in enumerate(starts):
        occ_roll = np.array(test_occ, copy=True)
        pred_chunks = []
        true_chunks = []
        chunk_records = []
        day_parse_ok = True
        for chunk_idx in range(chunks_per_day):
            t_start = day_start + chunk_idx * args.horizon
            sample = _sample_from_arrays(
                occ_roll=occ_roll,
                timestamps=timestamps,
                aux_values=aux_values,
                aux_columns=aux_columns,
                adj=splits.get("adj"),
                t_start=t_start,
                horizon=args.horizon,
                history_len=args.history_len,
                context_history_len=args.context_history_len,
                neighbor_k=args.neighbor_k,
            )
            if candidate_prediction_map is not None:
                sample = attach_candidate_refine_mask(
                    attach_candidate_curve(
                        sample,
                        horizon=args.horizon,
                        prediction_map=candidate_prediction_map,
                    )
                )
            sys_msg, usr_msg, diff, retrieved_t_starts = _build_prompt(
                sample=sample,
                horizon=args.horizon,
                prompt_style=args.prompt_style,
                use_rag=args.use_rag,
                use_diff_dora=args.use_diff_dora,
                retriever=retriever,
                splits=splits,
                node_idx=node_idx,
            )
            raw_output = generate(
                model,
                tokenizer,
                sys_msg,
                usr_msg,
                max_new_tokens=args.max_new_tokens,
            )
            parsed, padded_short_output = _parse_with_optional_padding(
                raw_output,
                args.horizon,
                allow_pad=args.allow_short_padding,
            )
            parse_ok = parsed is not None and len(parsed) == args.horizon
            target = test_occ[t_start : t_start + args.horizon, node_idx]
            if parse_ok:
                parsed = combine_candidate_prediction(
                    sample,
                    parsed,
                    mode=args.candidate_mode,
                    residual_clip=(args.candidate_residual_clip if args.candidate_residual_clip > 0 else None),
                    value_clip=(args.candidate_value_min, args.candidate_value_max),
                )
                parsed = np.asarray(parsed, dtype=np.float32)
                occ_roll[t_start : t_start + args.horizon, node_idx] = parsed
                pred_chunks.append(parsed)
                true_chunks.append(target)
            else:
                day_parse_ok = False
            chunk_records.append(
                {
                    "chunk_index": chunk_idx,
                    "t_start": int(t_start),
                    "target_start_timestamp": str(timestamps[t_start]),
                    "parse_ok": bool(parse_ok),
                    "padded_short_output": bool(padded_short_output),
                    "raw_generation": raw_output,
                    "parsed_prediction": None if not parse_ok else parsed.tolist(),
                    "target": target.tolist(),
                    "retrieved_t_starts": retrieved_t_starts,
                    "diff": diff,
                }
            )
            if not parse_ok:
                break

        record = {
            "day_index": day_idx,
            "day_t_start": int(day_start),
            "day_start_timestamp": str(timestamps[day_start]),
            "parse_ok": bool(day_parse_ok),
            "chunks": chunk_records,
            "parsed_prediction": None,
            "target": test_occ[day_start : day_start + args.day_horizon, node_idx].tolist(),
            "mae": None,
            "rmse": None,
        }
        if day_parse_ok and len(pred_chunks) == chunks_per_day:
            pred = np.concatenate(pred_chunks)
            true = np.concatenate(true_chunks)
            pred_raw = denormalize(pred, norm_min, norm_max)
            true_raw = denormalize(true, norm_min, norm_max)
            record["parsed_prediction"] = pred.tolist()
            record["mae"] = _mae(pred_raw, true_raw)
            record["rmse"] = _rmse(pred_raw, true_raw)
            record["prediction_range"] = float(pred_raw.max() - pred_raw.min())
            record["target_range"] = float(true_raw.max() - true_raw.min())
            record["daily_mean_accuracy"] = _daily_mean_accuracy(pred_raw, true_raw)
        records.append(record)
        print(
            f"[rolling_day] {day_idx + 1}/{len(starts)} "
            f"parse_ok={record['parse_ok']} mae={record['mae']}"
        )

    parsed_records = [row for row in records if row["parse_ok"]]
    payload = {
        "dataset": args.dataset,
        "adapter_dir": args.adapter_dir,
        "node_id": args.node_id,
        "horizon": args.horizon,
        "day_horizon": args.day_horizon,
        "history_len": args.history_len,
        "context_history_len": args.context_history_len,
        "day_stride": args.day_stride,
        "requested_days": len(records),
        "parsed_days": len(parsed_records),
        "parse_success_rate": len(parsed_records) / max(len(records), 1),
        "metrics": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }
    if parsed_records:
        all_pred = denormalize(
            np.stack([np.asarray(row["parsed_prediction"], dtype=np.float32) for row in parsed_records]),
            norm_min,
            norm_max,
        )
        all_true = denormalize(
            np.stack([np.asarray(row["target"], dtype=np.float32) for row in parsed_records]),
            norm_min,
            norm_max,
        )
        payload["metrics"] = {
            "overall": {
                "mae": _mae(all_pred, all_true),
                "rmse": _rmse(all_pred, all_true),
            },
            "mean_day_mae": float(np.mean([row["mae"] for row in parsed_records])),
            "mean_day_rmse": float(np.mean([row["rmse"] for row in parsed_records])),
            "mean_prediction_range": float(np.mean([row["prediction_range"] for row in parsed_records])),
            "mean_target_range": float(np.mean([row["target_range"] for row in parsed_records])),
            "mean_daily_mean_accuracy": float(np.mean([row["daily_mean_accuracy"] for row in parsed_records])),
            "median_daily_mean_accuracy": float(np.median([row["daily_mean_accuracy"] for row in parsed_records])),
            "share_days_daily_mean_accuracy_ge_0_8": float(
                (np.asarray([row["daily_mean_accuracy"] for row in parsed_records]) >= 0.8).mean()
            ),
        }
        print(json.dumps(payload["metrics"], indent=2))
    else:
        print("No fully parseable rolling days.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
