"""
src/data/load_st_evcdp.py
-------------------------
Load and preprocess the ST-EVCDP EV charging occupancy dataset.

Expected raw layout after download:
  data/raw/st_evcdp/
    occupancy.csv       – shape (T, N), float in [0,1]
    nodes.csv           – columns: node_id, zone_type, lat, lon, capacity
    adjacency.csv       – columns: src, dst, weight   (or an .npy matrix)

Usage:
  from src.data.load_st_evcdp import load_st_evcdp
  data = load_st_evcdp()
  # returns dict with keys: occupancy, node_meta, adj
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/st_evcdp")
PROCESSED_PATH = Path("data/processed/st_evcdp.pkl")


def load_occupancy(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return occupancy DataFrame, index=DatetimeIndex, columns=node_id."""
    occ_path = raw_dir / "occupancy.csv"
    if not occ_path.exists():
        raise FileNotFoundError(
            f"Occupancy file not found: {occ_path}\n"
            "Please download ST-EVCDP and place files under data/raw/st_evcdp/."
        )
    df = pd.read_csv(occ_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def _ensure_cached_node_ids(cached: dict, raw_dir: Path) -> dict:
    if cached.get("node_ids") is not None:
        return cached
    try:
        cached["node_ids"] = [str(c) for c in load_occupancy(raw_dir).columns]
    except FileNotFoundError:
        occ = cached.get("occupancy")
        if occ is not None:
            cached["node_ids"] = [str(i) for i in range(occ.shape[1])]
    return cached


def load_node_meta(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return node metadata DataFrame, index=node_id."""
    path = raw_dir / "nodes.csv"
    if not path.exists():
        return pd.DataFrame()  # allow graceful degradation
    df = pd.read_csv(path, index_col=0)
    return df


def load_adjacency(raw_dir: Path = RAW_DIR, n_nodes: int | None = None) -> np.ndarray:
    """Return adjacency matrix as (N, N) ndarray."""
    npy_path = raw_dir / "adjacency.npy"
    csv_path = raw_dir / "adjacency.csv"
    if npy_path.exists():
        return np.load(npy_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if n_nodes is None:
            n_nodes = max(df["src"].max(), df["dst"].max()) + 1
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for _, row in df.iterrows():
            adj[int(row["src"]), int(row["dst"])] = float(row.get("weight", 1.0))
        return adj
    # fallback: identity
    if n_nodes:
        return np.eye(n_nodes, dtype=np.float32)
    return np.empty((0, 0), dtype=np.float32)


def normalize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalise to [0,1], return (normed, min, max)."""
    vmin = float(arr.min())
    vmax = float(arr.max())
    eps = 1e-8
    return (arr - vmin) / (vmax - vmin + eps), vmin, vmax


def load_st_evcdp(
    raw_dir: Path = RAW_DIR,
    processed_path: Path = PROCESSED_PATH,
    force_reprocess: bool = False,
) -> dict:
    """
    Load ST-EVCDP dataset.

    Returns
    -------
    {
        "occupancy": np.ndarray  shape (T, N),
        "timestamps": pd.DatetimeIndex,
        "node_meta":  pd.DataFrame,
        "adj":        np.ndarray shape (N, N),
        "norm_min":   float,
        "norm_max":   float,
    }
    """
    if processed_path.exists() and not force_reprocess:
        with open(processed_path, "rb") as f:
            return _ensure_cached_node_ids(pickle.load(f), raw_dir)

    df = load_occupancy(raw_dir)
    node_meta = load_node_meta(raw_dir)

    T, N = df.shape
    occ_raw = df.values.astype(np.float32)

    # Fill NaN by forward-fill then backward-fill
    occ_df = pd.DataFrame(occ_raw)
    occ_df = occ_df.ffill().bfill()
    occ_raw = occ_df.values.astype(np.float32)

    occ_norm, vmin, vmax = normalize(occ_raw)

    adj = load_adjacency(raw_dir, n_nodes=N)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": df.index,
        "node_ids": [str(c) for c in df.columns],
        "node_meta": node_meta,
        "adj": adj,
        "norm_min": vmin,
        "norm_max": vmax,
    }

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[ST-EVCDP] T={T}, N={N}, NaN%={np.isnan(occ_raw).mean():.4f}")
    return result


if __name__ == "__main__":
    data = load_st_evcdp()
    occ = data["occupancy"]
    print(f"occupancy shape: {occ.shape}, range [{occ.min():.3f}, {occ.max():.3f}]")
    print(f"node_meta columns: {list(data['node_meta'].columns)}")
    print(f"adj shape: {data['adj'].shape}")
