"""
src/eval/metrics.py
--------------------
RMSE and MAE helpers, with optional inverse normalisation.
"""
from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(pred - true)))


def denormalize(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Invert min-max normalisation."""
    return arr * (vmax - vmin) + vmin


def evaluate(
    pred: np.ndarray,
    true: np.ndarray,
    norm_min: float = 0.0,
    norm_max: float = 1.0,
    denorm: bool = True,
) -> dict:
    """
    Compute RMSE and MAE, optionally after denormalisation.

    Parameters
    ----------
    pred, true : (horizon, N) or (B, horizon, N) arrays
    """
    if denorm:
        pred = denormalize(pred, norm_min, norm_max)
        true = denormalize(true, norm_min, norm_max)

    return {"rmse": rmse(pred, true), "mae": mae(pred, true)}


def per_horizon_metrics(
    preds: list[np.ndarray],
    trues: list[np.ndarray],
    horizon: int,
    norm_min: float = 0.0,
    norm_max: float = 1.0,
) -> dict:
    """
    Aggregate per-horizon metrics over a list of samples.

    preds / trues : each element shape (horizon, N)
    """
    p = np.stack(preds)  # (B, horizon, N)
    t = np.stack(trues)
    results = {}
    for h in range(horizon):
        results[h + 1] = evaluate(p[:, h], t[:, h], norm_min, norm_max)
    results["overall"] = evaluate(p, t, norm_min, norm_max)
    return results
