from __future__ import annotations

import numpy as np

from src.utils.node_context import resolve_node_id


def _sample_history_len(sample: dict) -> int:
    x_hist = sample.get("x_hist")
    if hasattr(x_hist, "shape") and len(x_hist.shape) >= 1:
        return max(1, int(x_hist.shape[0]))
    return max(1, int(sample.get("history_len", 12)))


def sample_history_end_index(sample: dict, frame) -> int:
    if frame is None or getattr(frame, "empty", True):
        return 0

    ts = sample.get("history_end_timestamp")
    if ts is not None and hasattr(frame, "index"):
        try:
            loc = frame.index.get_loc(ts)
            if isinstance(loc, slice):
                return int(loc.start)
            if isinstance(loc, (np.ndarray, list)):
                arr = np.asarray(loc)
                if arr.dtype == bool:
                    matches = np.flatnonzero(arr)
                    if len(matches):
                        return int(matches[0])
                elif len(arr):
                    return int(arr[0])
            return int(loc)
        except (KeyError, TypeError, ValueError):
            pass

    if "history_end_idx" in sample:
        idx = int(sample["history_end_idx"])
    else:
        idx = int(sample.get("t_start", 0)) + _sample_history_len(sample) - 1
    return min(max(idx, 0), len(frame) - 1)


def weather_at_history_end(weather, sample: dict) -> dict | None:
    if weather is None or getattr(weather, "empty", True):
        return None

    idx = sample_history_end_index(sample, weather)
    row = weather.iloc[idx]
    if hasattr(row, "to_dict"):
        data = row.to_dict()
        for key in list(data.keys()):
            lowered = str(key).lower()
            if "temp" in lowered and "temperature" not in data:
                data["temperature"] = data[key]
            if "humid" in lowered and "humidity" not in data:
                data["humidity"] = data[key]
        return data
    return None


def price_at_history_end(price, sample: dict, node_idx: int, *, node_ids=None, node_meta=None) -> float | None:
    if price is None or getattr(price, "empty", True):
        return None

    idx = sample_history_end_index(sample, price)
    row = price.iloc[idx]
    node_id = resolve_node_id(node_idx, node_ids=node_ids, node_meta=node_meta)
    if np.isscalar(row):
        return float(row)
    if hasattr(row, "to_dict"):
        data = row.to_dict()
        for key in (node_id, str(node_id), node_idx, str(node_idx)):
            if key in data:
                return float(data[key])
        vals = [float(value) for value in data.values() if value is not None]
        if vals:
            return float(np.mean(vals))
    return None
