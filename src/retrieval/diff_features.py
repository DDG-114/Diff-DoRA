"""
src/retrieval/diff_features.py
-------------------------------
Compute environmental differential features between a query sample
and its retrieved neighbours.

Diff features written to prompt:
  Diff T  : delta temperature   (current − historical mean)
  Diff P  : delta price         (current − historical mean)
  Diff Occ: delta occupancy     (current mean − historical mean)

Note (paper alignment): the paper's Table 1 shows a typo where
Diff P = +0.0 despite current=1.2 and hist=0.9.
We always compute: diff = current_value − retrieved_mean_value.
"""
from __future__ import annotations

import numpy as np


def _safe_mean(arr, default: float = 0.0) -> float:
    if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
        return default
    return float(np.mean(arr))


def compute_diff_features(
    query_sample: dict,
    retrieved_samples: list[dict],
    weather_current: dict | None = None,
    weather_retrieved: list[dict] | None = None,
    price_current: float | None = None,
    price_retrieved: list[float] | None = None,
    node_idx: int | None = None,
) -> dict:
    """
    Compute differential features between query and retrieved samples.

    Parameters
    ----------
    query_sample      : current sample (from build_samples)
    retrieved_samples : list of retrieved pool samples
    weather_current   : {"temperature": float, "humidity": float, ...} for query
    weather_retrieved : list of the same dicts for retrieved samples
    price_current     : electricity price at query time
    price_retrieved   : list of prices at retrieved times
    node_idx          : if set, compute occupancy diff for this node only;
                        otherwise fallback to the historical graph-wide mean

    Returns
    -------
    {
        "diff_occ":   float,   # occupancy difference (node-specific if node_idx is set)
        "diff_temp":  float | None,
        "diff_price": float | None,
    }
    """
    # Occupancy diff
    if node_idx is not None:
        curr_occ = float(query_sample["x_hist"][:, int(node_idx)].mean())
        hist_occ = _safe_mean([s["x_hist"][:, int(node_idx)].mean() for s in retrieved_samples])
    else:
        curr_occ = float(query_sample["x_hist"].mean())
        hist_occ = _safe_mean([s["x_hist"].mean() for s in retrieved_samples])
    diff_occ = curr_occ - hist_occ

    # Temperature diff
    diff_temp = None
    if weather_current is not None and weather_retrieved:
        t_curr = weather_current.get("temperature")
        t_hist = _safe_mean([w.get("temperature", 0.0) for w in weather_retrieved if w])
        if t_curr is not None:
            diff_temp = float(t_curr) - t_hist

    # Price diff
    diff_price = None
    if price_current is not None and price_retrieved:
        p_hist = _safe_mean([p for p in price_retrieved if p is not None])
        diff_price = float(price_current) - p_hist

    return {
        "diff_occ":   round(diff_occ,   4),
        "diff_temp":  round(diff_temp,  4) if diff_temp  is not None else None,
        "diff_price": round(diff_price, 4) if diff_price is not None else None,
    }


def format_diff_block(diff: dict) -> str:
    """Format diff features as a compact string for injection into prompt."""
    lines = [f"Diff Occ: {diff['diff_occ']:+.3f}"]
    if diff["diff_temp"] is not None:
        lines.append(f"Diff T: {diff['diff_temp']:+.3f}")
    if diff["diff_price"] is not None:
        lines.append(f"Diff P: {diff['diff_price']:+.3f}")
    return " | ".join(lines)
