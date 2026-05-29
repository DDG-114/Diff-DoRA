"""
src/data/build_splits.py
------------------------
Create train / val / test splits for a given dataset dict.

ST-EVCDP: 6:2:2
UrbanEV  : 8:1:1
"""
from __future__ import annotations

import numpy as np


SPLIT_RATIOS = {
    "st_evcdp": (0.6, 0.2, 0.2),
    "urbanev":  (0.8, 0.1, 0.1),
    "wotai_evcdp": (0.6, 0.2, 0.2),
    "renewable_solar": (0.6, 0.2, 0.2),
    "renewable_wind": (0.6, 0.2, 0.2),
    "gs_market": (1.0, 0.0, 0.0),
    "gs_market_2025": (1.0, 0.0, 0.0),
    "gs_price": (1.0, 0.0, 0.0),
    "gs_price_2025": (1.0, 0.0, 0.0),
}


def _slice_optional_frame(frame, indices):
    if frame is None or getattr(frame, "empty", True):
        return frame
    try:
        return frame.iloc[indices]
    except Exception:
        return frame


def build_splits(
    data: dict,
    dataset_name: str = "st_evcdp",
) -> dict:
    """
    Split occupancy (T, N) into train/val/test.

    Parameters
    ----------
    data : output of load_st_evcdp() or load_urbanev()
    dataset_name : one of 'st_evcdp', 'urbanev'

    Returns
    -------
    {
        "train": np.ndarray (T_tr, N),
        "val":   np.ndarray (T_va, N),
        "test":  np.ndarray (T_te, N),
        "timestamps_train": pd.DatetimeIndex,
        "timestamps_val":   pd.DatetimeIndex,
        "timestamps_test":  pd.DatetimeIndex,
    }
    """
    ratios = SPLIT_RATIOS.get(dataset_name, (0.6, 0.2, 0.2))
    occ = data["occupancy"]  # (T, N)
    ts  = data["timestamps"]
    T   = len(occ)

    split_indices = data.get("split_indices")
    if split_indices:
        train_idx = np.asarray(split_indices.get("train", []), dtype=np.int64)
        val_idx = np.asarray(split_indices.get("val", []), dtype=np.int64)
        test_idx = np.asarray(split_indices.get("test", []), dtype=np.int64)
        return {
            "train": occ[train_idx],
            "val": occ[val_idx],
            "test": occ[test_idx],
            "timestamps_train": ts[train_idx],
            "timestamps_val": ts[val_idx],
            "timestamps_test": ts[test_idx],
            "norm_min": data.get("norm_min", 0.0),
            "norm_max": data.get("norm_max", 1.0),
            "node_ids": data.get("node_ids"),
            "node_meta": data.get("node_meta"),
            "adj": data.get("adj"),
            "weather": data.get("weather"),
            "price": data.get("price"),
            "poi": data.get("poi"),
            "missing_mask": data.get("missing_mask"),
        }

    i1 = int(T * ratios[0])
    i2 = int(T * (ratios[0] + ratios[1]))

    return {
        "train": occ[:i1],
        "val":   occ[i1:i2],
        "test":  occ[i2:],
        "timestamps_train": ts[:i1],
        "timestamps_val":   ts[i1:i2],
        "timestamps_test":  ts[i2:],
        "norm_min": data.get("norm_min", 0.0),
        "norm_max": data.get("norm_max", 1.0),
        "node_ids": data.get("node_ids"),
        "node_meta": data.get("node_meta"),
        "adj": data.get("adj"),
        "weather": data.get("weather"),
        "price":   data.get("price"),
        "poi":     data.get("poi"),
    }


def validate_splits(splits: dict) -> None:
    for split in ("train", "val", "test"):
        arr = splits[split]
        nan_pct = np.isnan(arr).mean() * 100
        print(f"  {split}: shape={arr.shape}, NaN={nan_pct:.2f}%")
