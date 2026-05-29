"""
src/prompts/prompt_vanilla.py
------------------------------
Vanilla prompt template for single-expert occupancy forecasting.
No CoT, no RAG.

Usage:
  from src.prompts.prompt_vanilla import build_vanilla_prompt
  prompt = build_vanilla_prompt(sample, horizon=6)
"""
from __future__ import annotations

import numpy as np

TREND_EPS = 0.01


def _task_profile(
    *,
    domain_label: str | None = None,
    static_context: dict | None = None,
) -> dict[str, str]:
    static_context = static_context or {}
    node_id = str(static_context.get("node_id", "")).lower()
    site_type = str(static_context.get("site_type", "")).lower()
    source_column = str(static_context.get("source_column", "")).lower()
    node_type = str(static_context.get("type", "")).lower()
    zone_type = str(static_context.get("zone_type", "")).lower()

    if "price" in node_id or "price" in node_type or "price" in source_column:
        return {
            "system_role": "electricity price forecaster",
            "series_label": "electricity price",
            "series_label_title": "Electricity price",
            "target_noun": "electricity price",
            "unit_hint": "price",
            "domain_label": domain_label or static_context.get("zone_type") or "Market",
        }

    if any(
        token in node_type or token in zone_type or token in source_column
        for token in (
            "market",
            "generation_forecast",
            "load_forecast",
            "storage_dispatch",
            "renewable_forecast",
            "tie_line",
            "interconnection",
        )
    ):
        return {
            "system_role": "electricity system signal forecaster",
            "series_label": "system signal",
            "series_label_title": "System signal",
            "target_noun": "system signal value",
            "unit_hint": "system signal",
            "domain_label": domain_label or static_context.get("zone_type") or "Grid System",
        }

    if "pv" in node_id or site_type == "solar" or "pv" in source_column or "solar" in source_column:
        return {
            "system_role": "PV power generation forecaster",
            "series_label": "power output",
            "series_label_title": "Power output",
            "target_noun": "power output",
            "unit_hint": "power",
            "domain_label": "Solar",
        }
    if "wind" in node_id or site_type == "wind":
        return {
            "system_role": "wind power generation forecaster",
            "series_label": "power output",
            "series_label_title": "Power output",
            "target_noun": "power output",
            "unit_hint": "power",
            "domain_label": "Wind",
        }
    if "load" in node_id or "demand" in source_column or "demand" in node_id:
        return {
            "system_role": "power demand forecaster",
            "series_label": "power demand",
            "series_label_title": "Power demand",
            "target_noun": "power demand",
            "unit_hint": "power",
            "domain_label": domain_label or "Demand",
        }
    if "grid" in node_id or "storage" in node_id:
        return {
            "system_role": "power flow forecaster",
            "series_label": "power flow",
            "series_label_title": "Power flow",
            "target_noun": "power flow",
            "unit_hint": "power",
            "domain_label": domain_label or "Power Flow",
        }
    return {
        "system_role": "EV charging demand forecaster",
        "series_label": "occupancy",
        "series_label_title": "Occupancy",
        "target_noun": "occupancy",
        "unit_hint": "occupancy",
        "domain_label": domain_label or "EV",
    }


def build_system_message(*, static_context: dict | None = None, domain_label: str | None = None, cot: bool = False) -> str:
    profile = _task_profile(domain_label=domain_label, static_context=static_context)
    if cot:
        return (
            f"You are an expert {profile['system_role']}. "
            "You will reason step by step through three stages: "
            "(1) Gap Analysis, (2) Spatial Logic, (3) Numerical Prediction. "
            "Always end with exactly:\n"
            "Numerical Prediction: [v1, v2, ..., vH]\n"
            "where H is the forecast horizon."
        )
    return (
        f"You are an expert {profile['system_role']}. "
        f"You will be given historical {profile['series_label']} data and must predict future values. "
        "Always output ONLY a JSON list of floats, one value per step. "
        "Do not output any explanation."
    )


def _format_series(arr: np.ndarray, precision: int = 3) -> str:
    """Format a 1-D array as a compact list string."""
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"


def _sample_series(arr: np.ndarray, max_points: int = 28) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32).reshape(-1)
    if values.size <= max_points:
        return values
    indices = np.linspace(0, values.size - 1, num=max_points, dtype=int)
    return values[indices]


def format_long_history_block(
    sample: dict,
    *,
    node_idx: int,
    series_label: str,
    max_points: int = 28,
) -> str:
    context = sample.get("x_context")
    if context is None:
        return ""

    long_arr = np.asarray(context[:, node_idx], dtype=np.float32)
    recent_len = int(sample.get("history_len", sample["x_hist"].shape[0]))
    if long_arr.size <= recent_len:
        return ""

    sampled = _sample_series(long_arr, max_points=max_points)
    return (
        f"Long-range historical {series_label} ({long_arr.size} steps, sampled to {sampled.size} anchors): "
        f"{_format_series(sampled)}\n"
        f"Long-range summary: mean={long_arr.mean():.3f}, std={long_arr.std():.3f}, "
        f"min={long_arr.min():.3f}, max={long_arr.max():.3f}"
    )


def format_retrieved_examples(
    retrieved_samples: list[dict],
    *,
    node_idx: int,
    horizon: int,
    include_future: bool = True,
) -> str:
    """Render retrieved examples as history-only or history->future exemplars."""
    lines = []
    for i, rs in enumerate(retrieved_samples):
        rx = rs["x_hist"][:, node_idx]
        lines.append(f"  Ref {i+1} history: {_format_series(rx)}")
        if include_future and "y" in rs:
            ry = rs["y"][:horizon, node_idx]
            lines.append(f"  Ref {i+1} future:  {_format_series(ry)}")
    return "\n".join(lines) if lines else "  (none)"


def _normalise_scalar(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 3)
    return value


def _trend_label(short_slope: float, medium_slope: float, threshold: float = TREND_EPS) -> str:
    if abs(short_slope) >= threshold:
        return "rising" if short_slope > 0 else "falling"
    if abs(medium_slope) >= threshold:
        return "rising" if medium_slope > 0 else "falling"
    return "flat"


def summarize_trend_series(arr: np.ndarray) -> dict:
    """Summarize a node's recent shape using clamped short/medium lookbacks."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("summarize_trend_series expects a non-empty 1-D history array.")

    last_value = float(x[-1])
    short_idx = max(len(x) - 2, 0)
    medium_idx = max(len(x) - 4, 0)
    short_slope = float(x[-1] - x[short_idx])
    medium_slope = float(x[-1] - x[medium_idx])
    return {
        "last_value": last_value,
        "short_slope": short_slope,
        "medium_slope": medium_slope,
        "trend_label": _trend_label(short_slope, medium_slope),
    }


def summarize_trend_context(
    sample: dict,
    retrieved_samples: list[dict] | None = None,
    *,
    node_idx: int = 0,
) -> dict:
    """Build a compact trend summary for the current node and retrieved references."""
    current = summarize_trend_series(sample["x_hist"][:, node_idx])

    retrieved = []
    for rs in retrieved_samples or []:
        retrieved.append(summarize_trend_series(rs["x_hist"][:, node_idx]))

    if retrieved:
        retrieved_last_value = float(np.mean([item["last_value"] for item in retrieved]))
        retrieved_short_slope = float(np.mean([item["short_slope"] for item in retrieved]))
        retrieved_medium_slope = float(np.mean([item["medium_slope"] for item in retrieved]))
        retrieved_trend_label = _trend_label(retrieved_short_slope, retrieved_medium_slope)
    else:
        retrieved_last_value = None
        retrieved_short_slope = None
        retrieved_medium_slope = None
        retrieved_trend_label = None

    return {
        "current_last_value": current["last_value"],
        "current_short_slope": current["short_slope"],
        "current_medium_slope": current["medium_slope"],
        "current_trend_label": current["trend_label"],
        "retrieved_last_value": retrieved_last_value,
        "retrieved_short_slope": retrieved_short_slope,
        "retrieved_medium_slope": retrieved_medium_slope,
        "retrieved_trend_label": retrieved_trend_label,
    }


def format_trend_block(trend_summary: dict, *, include_retrieved: bool) -> str:
    """Render the trend cues that the prompt must expose explicitly."""
    lines = [
        f"Current last value: {trend_summary['current_last_value']:.3f}",
        f"Current short-term slope: {trend_summary['current_short_slope']:+.3f}",
        f"Current medium-term slope: {trend_summary['current_medium_slope']:+.3f}",
        f"Recent trend: {trend_summary['current_trend_label']}",
    ]
    if include_retrieved and trend_summary["retrieved_last_value"] is not None:
        lines.extend([
            f"Retrieved last value: {trend_summary['retrieved_last_value']:.3f}",
            f"Retrieved short-term slope: {trend_summary['retrieved_short_slope']:+.3f}",
            f"Retrieved medium-term slope: {trend_summary['retrieved_medium_slope']:+.3f}",
            f"Retrieved recent trend: {trend_summary['retrieved_trend_label']}",
        ])
    return "\n".join(lines)


def constant_forecast_instruction(*, include_retrieved: bool) -> str:
    """Hard guardrail against collapse to flat predictions."""
    if include_retrieved:
        return (
            "Do not output a constant forecast unless the recent node history and the "
            "retrieved examples are also flat."
        )
    return "Do not output a constant forecast unless the recent node history is flat."


def format_static_context(static_context: dict | None) -> str:
    """Render a compact metadata block only for available fields."""
    if not static_context:
        return ""

    profile = _task_profile(static_context=static_context)
    node_id = str(static_context.get("node_id", "")).lower()
    source_column = str(static_context.get("source_column", "")).lower()
    is_functional_energy_node = any(token in node_id for token in ("pv_", "grid_", "storage_", "load")) or any(
        token in source_column for token in ("pv", "load", "gate", "storage")
    )

    lines = []
    if static_context.get("node_id") is not None:
        lines.append(f"Node id: {_normalise_scalar(static_context['node_id'])}")
    if static_context.get("site_type"):
        lines.append(f"Site type: {static_context['site_type']}")
    elif static_context.get("zone_type") and not is_functional_energy_node:
        lines.append(f"Zone type: {static_context['zone_type']}")
    elif profile.get("domain_label") and is_functional_energy_node:
        lines.append(f"Signal type: {profile['domain_label']}")
    if static_context.get("type"):
        lines.append(f"Node role: {_normalise_scalar(static_context['type'])}")
    if static_context.get("capacity") is not None:
        lines.append(f"Node capacity: {_normalise_scalar(static_context['capacity'])}")
    if static_context.get("area") is not None:
        lines.append(f"Station area: {_normalise_scalar(static_context['area'])}")
    if static_context.get("unit"):
        lines.append(f"Unit: {_normalise_scalar(static_context['unit'])}")
    if static_context.get("source_column"):
        lines.append(f"Source signal: {_normalise_scalar(static_context['source_column'])}")
    if static_context.get("coverage_ratio") is not None:
        lines.append(f"Coverage ratio: {_normalise_scalar(static_context['coverage_ratio'])}")
    if static_context.get("road_length") is not None:
        lines.append(f"Nearby road length: {_normalise_scalar(static_context['road_length'])}")
    if static_context.get("poi_summary"):
        lines.append(f"POI summary: {static_context['poi_summary']}")

    if not lines:
        return ""
    return "Static station context:\n" + "\n".join(lines)


def format_auxiliary_feature_block(sample: dict, max_features: int = 8) -> str:
    """Render compact known covariate summaries for price/system forecasting."""
    aux_hist = sample.get("aux_hist")
    aux_future = sample.get("aux_future")
    columns = sample.get("aux_columns")
    if aux_hist is None and aux_future is None:
        return ""

    if columns is None:
        width = 0
        if aux_future is not None:
            width = np.asarray(aux_future).shape[1]
        elif aux_hist is not None:
            width = np.asarray(aux_hist).shape[1]
        columns = [f"feature_{idx}" for idx in range(width)]

    lines = ["Known auxiliary power-system features:"]
    for idx, name in enumerate(list(columns)[:max_features]):
        parts = [str(name)]
        if aux_hist is not None:
            hist = np.asarray(aux_hist, dtype=np.float32)[:, idx]
            parts.append(f"hist_last={hist[-1]:.3f}")
            parts.append(f"hist_mean={hist.mean():.3f}")
        if aux_future is not None:
            fut = np.asarray(aux_future, dtype=np.float32)[:, idx]
            parts.append(f"future_mean={fut.mean():.3f}")
            parts.append(f"future_min={fut.min():.3f}")
            parts.append(f"future_max={fut.max():.3f}")
            parts.append(f"future_delta={fut[-1] - fut[0]:+.3f}")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def format_candidate_curve_block(sample: dict, *, horizon: int, precision: int = 3) -> str:
    """Render optional candidate / skeleton forecasts for refinement-style prompts."""
    candidate = sample.get("candidate_future")
    if candidate is None:
        return ""

    arr = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return ""
    arr = arr[:horizon]

    parts = [
        f"Candidate forecast ({len(arr)} steps): {_format_series(arr, precision=precision)}",
        f"Candidate summary: mean={arr.mean():.3f}, min={arr.min():.3f}, max={arr.max():.3f}, delta={arr[-1] - arr[0]:+.3f}",
    ]

    lower = sample.get("candidate_lower")
    upper = sample.get("candidate_upper")
    if lower is not None and upper is not None:
        lo = np.asarray(lower, dtype=np.float32).reshape(-1)[: len(arr)]
        hi = np.asarray(upper, dtype=np.float32).reshape(-1)[: len(arr)]
        parts.append(
            f"Candidate interval summary: lower_mean={lo.mean():.3f}, upper_mean={hi.mean():.3f}, "
            f"avg_width={(hi - lo).mean():.3f}"
        )
    refine_mask = sample.get("candidate_refine_mask")
    if refine_mask is not None:
        mask = np.asarray(refine_mask, dtype=np.float32).reshape(-1)[: len(arr)]
        active = [int(idx) for idx, value in enumerate(mask) if value >= 0.5]
        parts.append(f"Candidate refine steps: {active}")
    return "\n".join(parts)


def format_neighbour_context_line(
    sample: dict,
    *,
    node_idx: int,
    series_label: str,
) -> str:
    """Render neighbour context only when a real multi-node signal exists."""
    nbr_feat = sample.get("nbr_feat")
    x_hist = sample.get("x_hist")
    if nbr_feat is None or x_hist is None:
        return ""
    x_arr = np.asarray(x_hist)
    if x_arr.ndim < 2 or x_arr.shape[1] <= 1:
        return ""

    nbr = float(np.asarray(nbr_feat)[-1, node_idx])
    return f"Neighbour {series_label} (last step): {nbr:.3f}\n\n"


def build_vanilla_prompt(
    sample: dict,
    node_idx: int = 0,
    horizon: int = 6,
    domain_label: str | None = None,
    static_context: dict | None = None,
    target_mode: str = "absolute",
) -> tuple[str, str]:
    """
    Build (system_message, user_message) for a single node.

    Parameters
    ----------
    sample    : dict from build_samples – keys: x_hist, time_feat, y
    node_idx  : which node to predict (0-indexed)
    horizon   : forecast horizon

    Returns
    -------
    (system_msg, user_msg)
    """
    x = sample["x_hist"][:, node_idx]
    recent_steps = len(x)
    profile = _task_profile(domain_label=domain_label, static_context=static_context)
    long_history_block = format_long_history_block(
        sample,
        node_idx=node_idx,
        series_label=profile["series_label"],
    )
    long_history_prefix = f"{long_history_block}\n\n" if long_history_block else ""
    trend_summary = summarize_trend_context(sample, node_idx=node_idx)

    meta_parts = []
    if profile["domain_label"]:
        meta_parts.append(f"[Domain: {profile['domain_label']}]")
    static_block = format_static_context(static_context)
    if static_block:
        meta_parts.append(static_block)
    meta_prefix = "\n\n".join(meta_parts)
    if meta_prefix:
        meta_prefix += "\n\n"
    aux_block = format_auxiliary_feature_block(sample)
    aux_prefix = f"{aux_block}\n\n" if aux_block else ""
    candidate_block = format_candidate_curve_block(sample, horizon=horizon)
    candidate_prefix = f"{candidate_block}\n\n" if candidate_block else ""
    if target_mode not in {"absolute", "residual", "selective_residual", "chunk_offset"}:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")
    target_instruction = (
        f"Output a JSON list of {horizon} floats representing predicted {profile['target_noun']}."
        if target_mode == "absolute"
        else (
            "Output a JSON list of "
            f"{horizon} floats that all stay close to a single block-level correction offset. "
            "This offset will be added to the candidate forecast across the whole horizon."
            if target_mode == "chunk_offset"
        else (
            f"Output a JSON list of {horizon} floats representing residual corrections to the candidate forecast. "
            "Each value should be added to the candidate at the same future step."
        ))
    )
    refinement_instruction = (
        "If a candidate forecast is provided, refine it rather than copying it mechanically. "
        "Preserve plausible ramps, floor-price periods, and peak-price intervals when supported by the context."
        if target_mode == "absolute"
        else (
            "If a candidate forecast is provided, do not regenerate the full future price path. "
            "Instead, output only the per-step correction needed to improve the candidate curve. "
            "For steps outside the candidate refine mask, keep the correction near zero."
            if target_mode != "chunk_offset"
            else (
                "If a candidate forecast is provided, do not regenerate the whole block. "
                "Output a nearly constant correction profile so the candidate block is shifted up or down while preserving its shape."
            )
        )
    )

    user_msg = (
        f"{meta_prefix}"
        f"{long_history_prefix}"
        f"Historical {profile['series_label']} (last {recent_steps} steps): {_format_series(x)}\n"
        f"Recent trend summary:\n{format_trend_block(trend_summary, include_retrieved=False)}\n\n"
        f"{aux_prefix}"
        f"{candidate_prefix}"
        f"Forecast horizon: {horizon} steps\n\n"
        f"{constant_forecast_instruction(include_retrieved=False)}\n"
        f"{refinement_instruction}\n"
        f"{target_instruction}\n"
        f"Numerical Prediction:"
    )
    return build_system_message(static_context=static_context, domain_label=profile["domain_label"], cot=False), user_msg


def build_direct_physical_prompt(
    sample: dict,
    retrieved_samples: list[dict],
    diff_features: dict | None,
    *,
    node_idx: int = 0,
    horizon: int = 6,
    domain_label: str | None = None,
    static_context: dict | None = None,
    include_env_diff: bool = False,
    target_mode: str = "absolute",
) -> tuple[str, str]:
    """
    Prompt without explicit CoT steps, but still exposes retrieved and physical context.
    Useful for paper-style "w/o CoT" ablations.
    """
    x = sample["x_hist"][:, node_idx]
    recent_steps = len(x)
    trend_summary = summarize_trend_context(
        sample,
        retrieved_samples,
        node_idx=node_idx,
    )
    profile = _task_profile(domain_label=domain_label, static_context=static_context)
    long_history_block = format_long_history_block(
        sample,
        node_idx=node_idx,
        series_label=profile["series_label"],
    )
    long_history_prefix = f"{long_history_block}\n\n" if long_history_block else ""

    meta_parts = []
    if profile["domain_label"]:
        meta_parts.append(f"[Domain: {profile['domain_label']}]")
    static_block = format_static_context(static_context)
    if static_block:
        meta_parts.append(static_block)
    meta_prefix = "\n\n".join(meta_parts)
    if meta_prefix:
        meta_prefix += "\n\n"
    aux_block = format_auxiliary_feature_block(sample)
    aux_prefix = f"{aux_block}\n\n" if aux_block else ""
    candidate_block = format_candidate_curve_block(sample, horizon=horizon)
    candidate_prefix = f"{candidate_block}\n\n" if candidate_block else ""
    if target_mode not in {"absolute", "residual", "selective_residual", "chunk_offset"}:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")
    target_instruction = (
        "Use the retrieved history-to-future examples together with the physical context above and output the forecast directly. "
        "If a candidate forecast is provided, treat it as a baseline curve to refine instead of a hard target."
        if target_mode == "absolute"
        else (
            "Use the retrieved history-to-future examples together with the physical context above, and if a candidate forecast is provided, "
            "output a nearly constant block-level correction that shifts the candidate curve up or down while preserving its internal shape."
            if target_mode == "chunk_offset"
        else (
            "Use the retrieved history-to-future examples together with the physical context above, but if a candidate forecast is provided, "
            "output only the per-step residual correction needed to improve that candidate. "
            "For steps outside the candidate refine mask, keep the correction near zero."
        ))
    )

    retr_block = format_retrieved_examples(
        retrieved_samples,
        node_idx=node_idx,
        horizon=horizon,
        include_future=True,
    )
    env_diff_block = ""
    if include_env_diff:
        from src.retrieval.diff_features import format_diff_block
        env_diff_block = f"Environmental differentials: {format_diff_block(diff_features)}\n\n"
    neighbour_line = format_neighbour_context_line(
        sample,
        node_idx=node_idx,
        series_label=profile["series_label"],
    )

    user_msg = (
        f"{meta_prefix}"
        f"{long_history_prefix}"
        f"Current historical {profile['series_label']} ({recent_steps} steps): {_format_series(x)}\n\n"
        f"Retrieved similar examples:\n{retr_block}\n\n"
        f"Recent trend summary:\n{format_trend_block(trend_summary, include_retrieved=bool(retrieved_samples))}\n\n"
        f"{env_diff_block}"
        f"{aux_prefix}"
        f"{candidate_prefix}"
        f"{neighbour_line}"
        f"Forecast horizon: {horizon} steps\n\n"
        f"{constant_forecast_instruction(include_retrieved=bool(retrieved_samples))}\n"
        f"{target_instruction}\n"
        "Numerical Prediction:"
    )
    return build_system_message(static_context=static_context, domain_label=profile["domain_label"], cot=False), user_msg


def build_vanilla_prompt_multi(
    sample: dict,
    horizon: int = 6,
) -> tuple[str, str]:
    """
    Build prompt for ALL nodes simultaneously.
    The model outputs a list-of-lists: [ [node0_step0...], [node1_step0...], ... ]
    """
    N = sample["x_hist"].shape[1]
    hist_steps = sample["x_hist"].shape[0]
    lines = []
    for n in range(N):
        x = sample["x_hist"][:, n]
        lines.append(f"  Node {n}: {_format_series(x)}")
    hist_block = "\n".join(lines)

    user_msg = (
        f"Historical occupancy (last {hist_steps} steps) for {N} nodes:\n{hist_block}\n\n"
        f"Forecast horizon: {horizon} steps\n"
        f"Output a JSON array of {N} arrays, each with {horizon} floats.\n"
        f"Numerical Prediction:"
    )
    return build_system_message(cot=False), user_msg
