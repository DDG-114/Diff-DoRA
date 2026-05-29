"""
src/data/build_samples.py
--------------------------
Window sample builder with configurable stride and optional long-history context.

Each sample always includes a recent history slice:
    x_hist    : (history_len, N)        – normalised recent history
    time_feat : (history_len, 4)        – [hour_sin, hour_cos, dow_sin, dow_cos]
    nbr_feat  : (history_len, N)        – mean of each node's direct neighbours
    y         : (horizon, N)            – target occupancy

When ``context_history_len`` is larger than ``history_len``, the sample also
includes a long-range context slice:
    x_context         : (context_history_len, N)
    time_feat_context : (context_history_len, 4)
    nbr_feat_context  : (context_history_len, N)

Supports horizons 3, 6, 9, 12 by default, while allowing arbitrary positive
horizon values when passed explicitly.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

HISTORY_LEN = 12
VALID_HORIZONS = (3, 6, 9, 12)


def _resolve_context_history_len(history_len: int, context_history_len: int | None) -> int:
    history_len = int(history_len)
    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")
    if context_history_len is None or int(context_history_len) <= 0:
        return history_len

    resolved = int(context_history_len)
    if resolved < history_len:
        raise ValueError(
            f"context_history_len must be >= history_len, got context_history_len={resolved}, "
            f"history_len={history_len}"
        )
    return resolved


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
    aux_features=None,
    horizons: Sequence[int] = VALID_HORIZONS,
    history_len: int = HISTORY_LEN,
    context_history_len: int | None = None,
    neighbor_k: int | None = None,
    window_stride: int = 1,
) -> dict[int, list[dict]]:
    """
    Build window samples for each horizon.

    Returns
    -------
    { horizon: [ {"x_hist", "time_feat", "nbr_feat", "y", "t_start"}, ... ] }
    """
    T, N = occ.shape
    history_len = int(history_len)
    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")
    effective_context_history_len = _resolve_context_history_len(history_len, context_history_len)
    window_stride = int(window_stride)
    if window_stride <= 0:
        raise ValueError(f"window_stride must be positive, got {window_stride}")

    # Pre-compute features for entire sequence
    if timestamps is not None and hasattr(timestamps, "hour"):
        time_feats = _time_features(timestamps)   # (T, 4)
    else:
        time_feats = np.zeros((T, 4), dtype=np.float32)

    aux_values = None
    aux_columns = None
    if aux_features is not None and not getattr(aux_features, "empty", True):
        if hasattr(aux_features, "reindex") and timestamps is not None:
            aux_frame = aux_features.reindex(timestamps).interpolate(method="time", limit_direction="both").ffill().bfill()
        else:
            aux_frame = aux_features
        if hasattr(aux_frame, "columns"):
            aux_columns = [str(col) for col in aux_frame.columns]
        aux_values = np.asarray(aux_frame, dtype=np.float32)
        if aux_values.shape[0] != T:
            raise ValueError(f"aux_features length mismatch: got {aux_values.shape[0]}, expected {T}")

    nbr_feats = _neighbour_features(occ, adj, neighbor_k=neighbor_k)  # (T, N)

    max_horizon = max(horizons)
    samples: dict[int, list] = {h: [] for h in horizons}

    for t in range(effective_context_history_len, T - max_horizon + 1, window_stride):
        x_hist = occ[t - history_len : t]
        t_feat = time_feats[t - history_len : t]
        n_feat = nbr_feats[t - history_len : t]

        x_context = None
        t_context = None
        n_context = None
        if effective_context_history_len > history_len:
            x_context = occ[t - effective_context_history_len : t]
            t_context = time_feats[t - effective_context_history_len : t]
            n_context = nbr_feats[t - effective_context_history_len : t]

        for h in horizons:
            if t + h > T:
                continue
            y = occ[t : t + h]                            # (h, N)
            sample = {
                "x_hist":    x_hist,
                "time_feat": t_feat,
                "nbr_feat":  n_feat,
                "y":         y,
                "t_start":   t,
                "history_len": history_len,
                "context_history_len": effective_context_history_len,
                "history_end_idx": t - 1,
            }
            if timestamps is not None and hasattr(timestamps, "__len__"):
                sample["history_end_timestamp"] = timestamps[t - 1]
                sample["target_start_timestamp"] = timestamps[t]
            if aux_values is not None:
                sample["aux_hist"] = aux_values[t - history_len : t]
                sample["aux_future"] = aux_values[t : t + h]
                sample["aux_columns"] = aux_columns
            if x_context is not None and t_context is not None and n_context is not None:
                sample["x_context"] = x_context
                sample["time_feat_context"] = t_context
                sample["nbr_feat_context"] = n_context
                if aux_values is not None:
                    sample["aux_context"] = aux_values[t - effective_context_history_len : t]
            samples[h].append(sample)

    total = sum(len(v) for v in samples.values())
    print(
        f"[build_samples] T={T} N={N} stride={window_stride} "
        f"recent_history={history_len} context_history={effective_context_history_len} | samples per horizon: "
          + ", ".join(f"h={h}:{len(samples[h])}" for h in horizons)
          + f"  total={total}"
    )
    return samples
