"""
src/data/build_samples.py
--------------------------
Sliding-window sample builder.

Each sample:
  x_hist  : (12, N)         – normalised historical occupancy
  time_feat: (12, 4)        – [hour_sin, hour_cos, dow_sin, dow_cos]
  nbr_feat : (12, N)        – mean of each node's direct neighbours
  y        : (horizon, N)   – target occupancy

Supports horizons 3, 6, 9, 12.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

HISTORY_LEN = 12
VALID_HORIZONS = (3, 6, 9, 12)


def _time_features(timestamps) -> np.ndarray:
    """Return (T, 4) cyclic time features for a DatetimeIndex."""
    hours = np.array(timestamps.hour, dtype=np.float32)
    dows  = np.array(timestamps.dayofweek, dtype=np.float32)
    feats = np.stack([
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * dows  / 7),
        np.cos(2 * np.pi * dows  / 7),
    ], axis=-1)  # (T, 4)
    return feats


def _neighbour_features(
    occ: np.ndarray,
    adj: np.ndarray | None,
    neighbor_k: int | None = None,
) -> np.ndarray:
    """
    For each node n, average its direct neighbours' occupancy.
    If adj is None or empty, returns zeros.
    """
    if adj is None or adj.size == 0:
        return np.zeros_like(occ)
    # row-normalise adjacency (exclude self-loops)
    A = adj.copy().astype(np.float32)
    np.fill_diagonal(A, 0.0)

    # Optional: keep only top-k neighbours for each node.
    if neighbor_k is not None and neighbor_k > 0 and neighbor_k < A.shape[1]:
        topk_idx = np.argpartition(A, -neighbor_k, axis=1)[:, -neighbor_k:]
        mask = np.zeros_like(A, dtype=bool)
        rows = np.arange(A.shape[0])[:, None]
        mask[rows, topk_idx] = True
        A = np.where(mask, A, 0.0)

    row_sum = A.sum(axis=1, keepdims=True) + 1e-8
    A_norm  = A / row_sum             # (N, N)
    return occ @ A_norm.T             # (T, N)


def build_samples(
    occ: np.ndarray,
    timestamps,
    adj: np.ndarray | None = None,
    horizons: Sequence[int] = VALID_HORIZONS,
    history_len: int = HISTORY_LEN,
    neighbor_k: int | None = None,
) -> dict[int, list[dict]]:
    """
    Build sliding-window samples for each horizon.

    Returns
    -------
    { horizon: [ {"x_hist", "time_feat", "nbr_feat", "y", "t_start"}, ... ] }
    """
    T, N = occ.shape

    # Pre-compute features for entire sequence
    if timestamps is not None and hasattr(timestamps, "hour"):
        time_feats = _time_features(timestamps)   # (T, 4)
    else:
        time_feats = np.zeros((T, 4), dtype=np.float32)

    nbr_feats = _neighbour_features(occ, adj, neighbor_k=neighbor_k)  # (T, N)

    max_horizon = max(horizons)
    samples: dict[int, list] = {h: [] for h in horizons}

    for t in range(history_len, T - max_horizon + 1):
        x_hist   = occ[t - history_len : t]               # (12, N)
        t_feat   = time_feats[t - history_len : t]        # (12, 4)
        n_feat   = nbr_feats[t - history_len : t]         # (12, N)

        for h in horizons:
            if t + h > T:
                continue
            y = occ[t : t + h]                            # (h, N)
            samples[h].append({
                "x_hist":    x_hist,
                "time_feat": t_feat,
                "nbr_feat":  n_feat,
                "y":         y,
                "t_start":   t,
            })

    total = sum(len(v) for v in samples.values())
    print(f"[build_samples] T={T} N={N} | samples per horizon: "
          + ", ".join(f"h={h}:{len(samples[h])}" for h in horizons)
          + f"  total={total}")
    return samples
