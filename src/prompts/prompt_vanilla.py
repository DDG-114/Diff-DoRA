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

SYSTEM_MSG = (
    "You are an expert EV charging demand forecaster. "
    "You will be given historical occupancy data and must predict future values. "
    "Always output ONLY a JSON list of floats, one value per step. "
    "Do not output any explanation."
)

TREND_EPS = 0.01


def _format_series(arr: np.ndarray, precision: int = 3) -> str:
    """Format a 1-D array as a compact list string."""
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"


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

    lines = []
    if static_context.get("node_id") is not None:
        lines.append(f"Node id: {_normalise_scalar(static_context['node_id'])}")
    if static_context.get("zone_type"):
        lines.append(f"Zone type: {static_context['zone_type']}")
    if static_context.get("capacity") is not None:
        lines.append(f"Node capacity: {_normalise_scalar(static_context['capacity'])}")
    if static_context.get("area") is not None:
        lines.append(f"Station area: {_normalise_scalar(static_context['area'])}")
    if static_context.get("road_length") is not None:
        lines.append(f"Nearby road length: {_normalise_scalar(static_context['road_length'])}")
    if static_context.get("poi_summary"):
        lines.append(f"POI summary: {static_context['poi_summary']}")

    if not lines:
        return ""
    return "Static station context:\n" + "\n".join(lines)


def build_vanilla_prompt(
    sample: dict,
    node_idx: int = 0,
    horizon: int = 6,
    domain_label: str | None = None,
    static_context: dict | None = None,
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
    x = sample["x_hist"][:, node_idx]   # (12,)
    trend_summary = summarize_trend_context(sample, node_idx=node_idx)

    meta_parts = []
    if domain_label:
        meta_parts.append(f"[Domain: {domain_label}]")
    static_block = format_static_context(static_context)
    if static_block:
        meta_parts.append(static_block)
    meta_prefix = "\n\n".join(meta_parts)
    if meta_prefix:
        meta_prefix += "\n\n"

    user_msg = (
        f"{meta_prefix}"
        f"Historical occupancy (last 12 steps): {_format_series(x)}\n"
        f"Recent trend summary:\n{format_trend_block(trend_summary, include_retrieved=False)}\n\n"
        f"Forecast horizon: {horizon} steps\n\n"
        f"{constant_forecast_instruction(include_retrieved=False)}\n"
        f"Output a JSON list of {horizon} floats representing predicted occupancy.\n"
        f"Numerical Prediction:"
    )
    return SYSTEM_MSG, user_msg


def build_direct_physical_prompt(
    sample: dict,
    retrieved_samples: list[dict],
    diff_block: str,
    *,
    node_idx: int = 0,
    horizon: int = 6,
    domain_label: str | None = None,
    static_context: dict | None = None,
) -> tuple[str, str]:
    """
    Prompt without explicit CoT steps, but still exposes retrieved and physical context.
    Useful for paper-style "w/o CoT" ablations.
    """
    x = sample["x_hist"][:, node_idx]
    nbr = sample["nbr_feat"][-1, node_idx]
    trend_summary = summarize_trend_context(
        sample,
        retrieved_samples,
        node_idx=node_idx,
    )

    meta_parts = []
    if domain_label:
        meta_parts.append(f"[Domain: {domain_label}]")
    static_block = format_static_context(static_context)
    if static_block:
        meta_parts.append(static_block)
    meta_prefix = "\n\n".join(meta_parts)
    if meta_prefix:
        meta_prefix += "\n\n"

    retr_lines = []
    for i, rs in enumerate(retrieved_samples):
        rx = rs["x_hist"][:, node_idx]
        retr_lines.append(f"  Ref {i+1}: {_format_series(rx)}")
    retr_block = "\n".join(retr_lines) if retr_lines else "  (none)"

    user_msg = (
        f"{meta_prefix}"
        f"Current historical occupancy (12 steps): {_format_series(x)}\n\n"
        f"Retrieved similar windows:\n{retr_block}\n\n"
        f"Recent trend summary:\n{format_trend_block(trend_summary, include_retrieved=bool(retrieved_samples))}\n\n"
        f"Environmental differentials: {diff_block}\n\n"
        f"Neighbour occupancy (last step): {nbr:.3f}\n\n"
        f"Forecast horizon: {horizon} steps\n\n"
        f"{constant_forecast_instruction(include_retrieved=bool(retrieved_samples))}\n"
        "Use the physical context above and output the forecast directly.\n"
        "Numerical Prediction:"
    )
    return SYSTEM_MSG, user_msg


def build_vanilla_prompt_multi(
    sample: dict,
    horizon: int = 6,
) -> tuple[str, str]:
    """
    Build prompt for ALL nodes simultaneously.
    The model outputs a list-of-lists: [ [node0_step0...], [node1_step0...], ... ]
    """
    N = sample["x_hist"].shape[1]
    lines = []
    for n in range(N):
        x = sample["x_hist"][:, n]
        lines.append(f"  Node {n}: {_format_series(x)}")
    hist_block = "\n".join(lines)

    user_msg = (
        f"Historical occupancy (last 12 steps) for {N} nodes:\n{hist_block}\n\n"
        f"Forecast horizon: {horizon} steps\n"
        f"Output a JSON array of {N} arrays, each with {horizon} floats.\n"
        f"Numerical Prediction:"
    )
    return SYSTEM_MSG, user_msg
