"""
src/data/load_urbanev.py
-------------------------
Load and preprocess the UrbanEV EV charging occupancy dataset.

Expected raw layout after download:
  data/raw/urbanev/
    occupancy.csv     – shape (T, N), float in [0,1]
    weather.csv       – columns: timestamp, temperature, humidity, ...
    price.csv         – columns: timestamp, price
    poi.csv           – columns: node_id, poi_category, poi_count
    adjacency.csv     – columns: src, dst, weight  (or adjacency.npy)

Usage:
  from src.data.load_urbanev import load_urbanev
  data = load_urbanev()
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/urbanev")
PROCESSED_PATH = Path("data/processed/urbanev.pkl")


def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def normalize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    vmin = float(arr.min())
    vmax = float(arr.max())
    eps = 1e-8
    return (arr - vmin) / (vmax - vmin + eps), vmin, vmax


def load_urbanev(
    raw_dir: Path = RAW_DIR,
    processed_path: Path = PROCESSED_PATH,
    force_reprocess: bool = False,
) -> dict:
    """
    Load UrbanEV dataset.

    Returns
    -------
    {
        "occupancy":  np.ndarray (T, N),
        "timestamps": pd.DatetimeIndex,
        "weather":    pd.DataFrame,
        "price":      pd.DataFrame,
        "poi":        pd.DataFrame,
        "adj":        np.ndarray (N, N),
        "norm_min":   float,
        "norm_max":   float,
    }
    """
    if processed_path.exists() and not force_reprocess:
        with open(processed_path, "rb") as f:
            return pickle.load(f)

    occ_path = raw_dir / "occupancy.csv"
    if not occ_path.exists():
        raise FileNotFoundError(
            f"UrbanEV occupancy not found: {occ_path}\n"
            "Download UrbanEV and place files under data/raw/urbanev/."
        )

    occ_df = pd.read_csv(occ_path, index_col=0, parse_dates=True).sort_index()
    T, N = occ_df.shape
    occ_raw = occ_df.values.astype(np.float32)
    occ_raw = pd.DataFrame(occ_raw).ffill().bfill().values.astype(np.float32)
    occ_norm, vmin, vmax = normalize(occ_raw)

    weather = _read_csv_safe(raw_dir / "weather.csv", index_col=0, parse_dates=True)
    price   = _read_csv_safe(raw_dir / "price.csv",   index_col=0, parse_dates=True)
    poi     = _read_csv_safe(raw_dir / "poi.csv",     index_col=0)

    # Adjacency
    npy_path = raw_dir / "adjacency.npy"
    csv_adj  = raw_dir / "adjacency.csv"
    if npy_path.exists():
        adj = np.load(npy_path)
    elif csv_adj.exists():
        df_adj = pd.read_csv(csv_adj)
        adj = np.zeros((N, N), dtype=np.float32)
        for _, row in df_adj.iterrows():
            adj[int(row["src"]), int(row["dst"])] = float(row.get("weight", 1.0))
    else:
        adj = np.eye(N, dtype=np.float32)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": occ_df.index,
        "weather": weather,
        "price": price,
        "poi": poi,
        "adj": adj,
        "norm_min": vmin,
        "norm_max": vmax,
    }

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[UrbanEV] T={T}, N={N}, NaN%={np.isnan(occ_raw).mean():.4f}")
    return result


if __name__ == "__main__":
    data = load_urbanev()
    print(f"occupancy: {data['occupancy'].shape}")
