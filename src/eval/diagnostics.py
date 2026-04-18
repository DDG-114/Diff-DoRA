"""
src/eval/diagnostics.py
-----------------------
Diagnostics for horizon-level forecast shape analysis.
"""
from __future__ import annotations

import numpy as np

from src.eval.metrics import denormalize

CONSTANT_TOL = 1e-9


def empty_diagnostics() -> dict:
    return {
        "constant_prediction_count": 0,
        "constant_prediction_rate": 0.0,
        "target_constant_count": 0,
        "target_constant_rate": 0.0,
        "mean_prediction_range": 0.0,
        "mean_target_range": 0.0,
    }


def compute_sequence_diagnostics(
    preds: list[np.ndarray],
    trues: list[np.ndarray],
    norm_min: float,
    norm_max: float,
) -> dict:
    """Compute constant/range diagnostics on parsed, denormalized horizon sequences."""
    if not preds or not trues:
        return empty_diagnostics()

    pred_arr = np.asarray(preds, dtype=np.float32)
    true_arr = np.asarray(trues, dtype=np.float32)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr[None, :]
        true_arr = true_arr[None, :]

    pred_denorm = denormalize(pred_arr, norm_min, norm_max)
    true_denorm = denormalize(true_arr, norm_min, norm_max)

    pred_ranges = np.ptp(pred_denorm, axis=1)
    true_ranges = np.ptp(true_denorm, axis=1)
    constant_pred = pred_ranges <= CONSTANT_TOL
    constant_true = true_ranges <= CONSTANT_TOL

    return {
        "constant_prediction_count": int(constant_pred.sum()),
        "constant_prediction_rate": float(constant_pred.mean()),
        "target_constant_count": int(constant_true.sum()),
        "target_constant_rate": float(constant_true.mean()),
        "mean_prediction_range": float(pred_ranges.mean()),
        "mean_target_range": float(true_ranges.mean()),
    }


def compute_diagnostics_from_records(
    records: list[dict],
    norm_min: float,
    norm_max: float,
) -> tuple[dict, dict]:
    """Backfill diagnostics from ablation records when they were not saved originally."""
    preds = []
    trues = []
    domain_preds = {"CBD": [], "Residential": []}
    domain_trues = {"CBD": [], "Residential": []}

    for record in records:
        if not record.get("parse_ok"):
            continue
        pred = record.get("parsed_prediction")
        true = record.get("target")
        domain = record.get("domain")
        if pred is None or true is None:
            continue

        pred_arr = np.asarray(pred, dtype=np.float32)
        true_arr = np.asarray(true, dtype=np.float32)
        preds.append(pred_arr)
        trues.append(true_arr)
        if domain in domain_preds:
            domain_preds[domain].append(pred_arr)
            domain_trues[domain].append(true_arr)

    overall = compute_sequence_diagnostics(preds, trues, norm_min, norm_max)
    domain_diagnostics = {}
    for domain in ("CBD", "Residential"):
        domain_diagnostics[domain] = compute_sequence_diagnostics(
            domain_preds[domain],
            domain_trues[domain],
            norm_min,
            norm_max,
        )
    return overall, domain_diagnostics
