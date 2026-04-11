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


def _format_series(arr: np.ndarray, precision: int = 3) -> str:
    """Format a 1-D array as a compact list string."""
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"


def build_vanilla_prompt(
    sample: dict,
    node_idx: int = 0,
    horizon: int = 6,
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
    y = sample["y"][:horizon, node_idx] # (horizon,)  – for reference only

    user_msg = (
        f"Historical occupancy (last 12 steps): {_format_series(x)}\n"
        f"Forecast horizon: {horizon} steps\n\n"
        f"Output a JSON list of {horizon} floats representing predicted occupancy.\n"
        f"Numerical Prediction:"
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
