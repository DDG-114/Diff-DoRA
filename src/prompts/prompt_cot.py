"""
src/prompts/prompt_cot.py
--------------------------
Three-stage Chain-of-Thought prompt template (paper Section 3.x):

  Stage 1 – Gap Analysis     : Compare current occupancy vs. retrieved historical
  Stage 2 – Spatial Logic    : Incorporate neighbour influence
  Stage 3 – Numerical Pred.  : Output the final list

The parser only reads the last "Numerical Prediction: [...]" line.
"""
from __future__ import annotations

import numpy as np

from src.prompts.prompt_vanilla import (
    _format_series,
    _task_profile,
    build_system_message,
    constant_forecast_instruction,
    format_long_history_block,
    format_auxiliary_feature_block,
    format_neighbour_context_line,
    format_retrieved_examples,
    format_static_context,
    format_trend_block,
    summarize_trend_context,
)
from src.retrieval.diff_features import format_diff_block

COT_TARGET_TEMPLATE = """\
## Stage 1 – Gap Analysis
Current mean {series_label}: {curr_occ:.3f}
Retrieved mean {series_label}: {retr_occ:.3f}
Current last value: {curr_last:.3f}
Current short-term slope: {curr_short:+.3f}
Current medium-term slope: {curr_medium:+.3f}
Retrieved last value: {retr_last:.3f}
Retrieved short-term slope: {retr_short:+.3f}
Current recent trend: {curr_trend}
Retrieved recent trend: {retr_trend}
{env_diff_line}\
The gap suggests {gap_desc}.
Trend comparison: {trend_desc}.

## Stage 2 – Spatial Logic
{static_block}Neighbour mean {series_label} (last step): {nbr_occ:.3f}
Spatial influence: {spatial_desc}.

## Stage 3 – Numerical Prediction
Current trend check: {curr_trend}.
Retrieved trend check: {retr_trend}.
Constant-forecast check: {const_desc}
Based on the recent trend, retrieved trend, and physical context:
Numerical Prediction: {pred_list}"""


def build_cot_prompt(
    sample: dict,
    retrieved_samples: list[dict],
    diff_features: dict | None,
    node_idx: int = 0,
    horizon: int = 6,
    domain_label: str | None = None,
    static_context: dict | None = None,
    include_env_diff: bool = False,
) -> tuple[str, str]:
    """
    Build (system_msg, user_msg) including retrieved context, diff features, and optional domain label.
    
    The domain_label (e.g., "CBD" or "Residential") acts as a metadata tag to guide the MoE router,
    as described in the LR-MoE paper: "In the inference phase, the model activates the corresponding
    expert based on the metadata label in the input prompt."
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

    retr_block = format_retrieved_examples(
        retrieved_samples,
        node_idx=node_idx,
        horizon=horizon,
        include_future=True,
    )

    env_diff_block = ""
    if include_env_diff:
        diff_str = format_diff_block(diff_features)
        env_diff_block = f"Environmental differentials: {diff_str}\n\n"
    aux_block = format_auxiliary_feature_block(sample)
    aux_prefix = f"{aux_block}\n\n" if aux_block else ""
    neighbour_line = format_neighbour_context_line(
        sample,
        node_idx=node_idx,
        series_label=profile["series_label"],
    )

    # Add domain label as metadata prefix if provided
    meta_parts = []
    if profile["domain_label"]:
        meta_parts.append(f"[Domain: {profile['domain_label']}]")
    static_block = format_static_context(static_context)
    if static_block:
        meta_parts.append(static_block)
    meta_prefix = "\n\n".join(meta_parts)
    if meta_prefix:
        meta_prefix += "\n\n"

    user_msg = (
        f"{meta_prefix}"
        f"{long_history_prefix}"
        f"Current historical {profile['series_label']} ({recent_steps} steps): {_format_series(x)}\n\n"
        f"Retrieved similar examples:\n{retr_block}\n\n"
        f"Stage 1 trend cues:\n{format_trend_block(trend_summary, include_retrieved=bool(retrieved_samples))}\n\n"
        f"{env_diff_block}"
        f"{aux_prefix}"
        f"{neighbour_line}"
        f"Forecast horizon: {horizon} steps\n\n"
        f"You must output exactly {horizon} numeric values in the final list, one value for each future step. "
        f"Do not output a shortened list or a single representative value.\n\n"
        f"{constant_forecast_instruction(include_retrieved=bool(retrieved_samples))}\n"
        "For electricity price forecasting, preserve the day-ahead profile shape: low-price floor periods, ramps, "
        "peak-price intervals, and later recovery should be predicted step by step rather than collapsed to the "
        "minimum observed price.\n"
        "In Stage 1 compare both level and local trend cues. "
        "Use the retrieved history-to-future examples as references, but adjust them according to the current trend, "
        "known auxiliary features, environmental differentials, and neighbour context rather than copying them mechanically. "
        "In Stage 3 decide whether the forecast should keep changing or remain nearly flat based on those cues.\n"
        "Numerical Prediction:"
    )
    return build_system_message(static_context=static_context, domain_label=profile["domain_label"], cot=True), user_msg


def build_cot_target(
    sample: dict,
    retrieved_samples: list[dict],
    diff_features: dict | None,
    node_idx: int = 0,
    horizon: int = 6,
    static_context: dict | None = None,
    include_env_diff: bool = False,
) -> str:
    """
    Build the supervised target string (for training).
    """
    x       = sample["x_hist"][:, node_idx]
    y       = sample["y"][:horizon, node_idx]
    trend_summary = summarize_trend_context(
        sample,
        retrieved_samples,
        node_idx=node_idx,
    )
    profile = _task_profile(static_context=static_context)
    diff_features = diff_features or {"diff_occ": 0.0, "diff_temp": None, "diff_price": None}

    curr_occ = float(x.mean())
    retr_occ = float(np.mean([rs["x_hist"][:, node_idx].mean() for rs in retrieved_samples])) \
               if retrieved_samples else curr_occ
    diff_occ = diff_features.get("diff_occ", 0.0)
    gap_desc  = "higher demand" if diff_occ > 0.05 else ("lower demand" if diff_occ < -0.05 else "stable demand")
    has_neighbour_context = np.asarray(sample.get("x_hist")).ndim >= 2 and np.asarray(sample.get("x_hist")).shape[1] > 1
    nbr = float(sample["nbr_feat"][-1, node_idx]) if has_neighbour_context else curr_occ
    spatial_desc = "increasing pressure" if has_neighbour_context and nbr > curr_occ + 0.05 else "stable neighbourhood"
    if trend_summary["retrieved_trend_label"] is None:
        retr_trend = trend_summary["current_trend_label"]
        retr_last = trend_summary["current_last_value"]
        retr_short = trend_summary["current_short_slope"]
    else:
        retr_trend = trend_summary["retrieved_trend_label"]
        retr_last = trend_summary["retrieved_last_value"]
        retr_short = trend_summary["retrieved_short_slope"]

    if trend_summary["current_trend_label"] == retr_trend:
        trend_desc = "retrieved references reinforce the current local trend"
    else:
        trend_desc = "retrieved references suggest a different local shape, so the forecast should balance both cues"

    constant_ok = trend_summary["current_trend_label"] == "flat" and retr_trend == "flat"
    const_desc = (
        "a constant forecast is only acceptable because both the node history and retrieved references are flat."
        if constant_ok else
        "a non-constant forecast is required because the recent node history or retrieved references are not flat."
    )

    static_block = format_static_context(static_context)
    if static_block:
        static_block = static_block.replace("Static station context:\n", "")
        static_block = static_block + "\n"
    env_diff_line = ""
    if include_env_diff:
        env_diff_line = f"Environmental differentials: {format_diff_block(diff_features)}\n"

    if horizon > 32:
        y = sample["y"][:horizon, node_idx]
        neighbour_target_line = (
            f"Neighbour mean {profile['series_label']} (last step): {nbr:.3f}.\n"
            if has_neighbour_context else
            "No multi-node neighbour context is available for this target.\n"
        )
        return (
            "## Stage 1 - Gap Analysis\n"
            f"Current recent trend: {trend_summary['current_trend_label']}.\n"
            f"Retrieved recent trend: {retr_trend}.\n"
            f"{env_diff_line}"
            "\n## Stage 2 - Spatial Logic\n"
            f"{neighbour_target_line}"
            "\n## Stage 3 - Numerical Prediction\n"
            f"Numerical Prediction: {_format_series(y)}"
        )

    return COT_TARGET_TEMPLATE.format(
        series_label = profile["series_label"],
        curr_occ    = curr_occ,
        retr_occ    = retr_occ,
        curr_last   = trend_summary["current_last_value"],
        curr_short  = trend_summary["current_short_slope"],
        curr_medium = trend_summary["current_medium_slope"],
        retr_last   = retr_last,
        retr_short  = retr_short,
        curr_trend  = trend_summary["current_trend_label"],
        retr_trend  = retr_trend,
        env_diff_line = env_diff_line,
        gap_desc    = gap_desc,
        trend_desc  = trend_desc,
        static_block= static_block,
        nbr_occ     = nbr,
        spatial_desc= spatial_desc,
        const_desc  = const_desc,
        pred_list   = _format_series(y),
    )
