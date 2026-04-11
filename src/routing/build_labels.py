"""
src/routing/build_labels.py
----------------------------
Construct hard-routing labels for each node (CBD vs. Residential).

Priority:
1. Use zone_type column in node_meta if available.
2. Fall back to percentile-based proxy: nodes with mean occupancy > 50th percentile
   are labelled "cbd" (high-demand), others "res" (residential/low-demand).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_CBD = 0
LABEL_RES = 1


def build_labels_from_meta(node_meta: pd.DataFrame) -> np.ndarray | None:
    """
    Try to extract routing labels from node metadata.
    Returns integer array of shape (N,) or None if column not found.
    """
    if node_meta is None or node_meta.empty:
        return None
    for col in ("zone_type", "type", "area_type", "category"):
        if col in node_meta.columns:
            raw = node_meta[col].astype(str).str.lower()
            labels = np.where(raw.str.contains("cbd|commercial|business"), LABEL_CBD, LABEL_RES)
            return labels.astype(np.int32)
    return None


def build_labels_from_occupancy(occupancy: np.ndarray) -> np.ndarray:
    """
    Proxy label: nodes whose mean occupancy exceeds the median are 'CBD' (0),
    the rest are 'Residential' (1).

    Parameters
    ----------
    occupancy : (T, N) normalised array
    """
    mean_occ = occupancy.mean(axis=0)          # (N,)
    threshold = np.median(mean_occ)
    labels = np.where(mean_occ >= threshold, LABEL_CBD, LABEL_RES)
    return labels.astype(np.int32)


def build_routing_labels(
    occupancy: np.ndarray,
    node_meta: "pd.DataFrame | None" = None,
) -> np.ndarray:
    """
    Main entry-point: returns integer label array (N,).
    0 = CBD / high-demand,  1 = Residential / low-demand.
    """
    labels = None
    if node_meta is not None and not getattr(node_meta, "empty", True):
        labels = build_labels_from_meta(node_meta)

    if labels is None:
        print("[routing] No metadata labels found – using occupancy-based proxy.")
        labels = build_labels_from_occupancy(occupancy)

    n_cbd = (labels == LABEL_CBD).sum()
    n_res = (labels == LABEL_RES).sum()
    print(f"[routing] CBD nodes: {n_cbd}, Residential nodes: {n_res}")
    return labels
