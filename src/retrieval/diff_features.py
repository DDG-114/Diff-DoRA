"""
src/retrieval/diff_features.py
-------------------------------
Compute differential features between a query sample and its retrieved
neighbours.

Internal diff statistics:
  Diff T   : delta temperature   (current − historical mean)
  Diff P   : delta price         (current − historical mean)
  Diff Occ : delta occupancy     (current mean − historical mean)

Paper-style Diff-DoRA only injects the environmental deltas (temperature and
price) into the prompt. Occupancy gap remains an internal reasoning signal.

Note (paper alignment): the paper's Table 1 shows a typo where Diff P = +0.0
despite current=1.2 and hist=0.9. We always compute current − historical.
"""
from __future__ import annotations

import numpy as np


def _safe_mean(arr, default: float = 0.0) -> float:
    if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
        return default
    clean = [x for x in arr if x is not None]
    if not clean:
        return default
    return float(np.mean(clean))


def _first_present(mapping: dict | None, keys: tuple[str, ...]):
    if not mapping:
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


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
        "diff_ghi":   float | None,
        "diff_dni":   float | None,
        "diff_tsi":   float | None,
        "diff_wind":  float | None,
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

    # Renewable-generation differentials
    diff_ghi = None
    diff_dni = None
    diff_tsi = None
    diff_wind = None
    if weather_current is not None and weather_retrieved:
        ghi_curr = _first_present(weather_current, ("global_horizontal_irradiance", "ghi"))
        ghi_hist = _safe_mean([
            _first_present(w, ("global_horizontal_irradiance", "ghi"))
            for w in weather_retrieved if w
        ])
        if ghi_curr is not None:
            diff_ghi = float(ghi_curr) - ghi_hist

        dni_curr = _first_present(weather_current, ("direct_normal_irradiance", "dni"))
        dni_hist = _safe_mean([
            _first_present(w, ("direct_normal_irradiance", "dni"))
            for w in weather_retrieved if w
        ])
        if dni_curr is not None:
            diff_dni = float(dni_curr) - dni_hist

        tsi_curr = _first_present(weather_current, ("total_solar_irradiance", "solar_rad", "dhi"))
        tsi_hist = _safe_mean([
            _first_present(w, ("total_solar_irradiance", "solar_rad", "dhi"))
            for w in weather_retrieved if w
        ])
        if tsi_curr is not None:
            diff_tsi = float(tsi_curr) - tsi_hist

        wind_curr = _first_present(weather_current, ("wind_speed_hub", "wind_speed_50m", "wind_speed_30m", "wind_speed_10m", "wind_speed"))
        if wind_curr is not None:
            wind_hist_values = []
            for weather_item in weather_retrieved:
                if not weather_item:
                    continue
                candidate = _first_present(
                    weather_item,
                    ("wind_speed_hub", "wind_speed_50m", "wind_speed_30m", "wind_speed_10m", "wind_speed"),
                )
                if candidate is not None:
                    wind_hist_values.append(candidate)
            diff_wind = float(wind_curr) - _safe_mean(wind_hist_values)

    return {
        "diff_occ":   round(diff_occ,   4),
        "diff_temp":  round(diff_temp,  4) if diff_temp  is not None else None,
        "diff_price": round(diff_price, 4) if diff_price is not None else None,
        "diff_ghi":   round(diff_ghi,   4) if diff_ghi   is not None else None,
        "diff_dni":   round(diff_dni,   4) if diff_dni   is not None else None,
        "diff_tsi":   round(diff_tsi,   4) if diff_tsi   is not None else None,
        "diff_wind":  round(diff_wind,  4) if diff_wind  is not None else None,
    }


def format_diff_block(diff: dict | None) -> str:
    """Format only environmental differentials for prompt injection."""
    diff = diff or {}
    lines = []
    if diff.get("diff_temp") is not None:
        lines.append(f"Diff T: {float(diff['diff_temp']):+.3f}")
    if diff.get("diff_price") is not None:
        lines.append(f"Diff P: {float(diff['diff_price']):+.3f}")
    if diff.get("diff_tsi") is not None:
        lines.append(f"Diff SI: {float(diff['diff_tsi']):+.3f}")
    if diff.get("diff_dni") is not None:
        lines.append(f"Diff DNI: {float(diff['diff_dni']):+.3f}")
    if diff.get("diff_ghi") is not None:
        lines.append(f"Diff GHI: {float(diff['diff_ghi']):+.3f}")
    if diff.get("diff_wind") is not None:
        lines.append(f"Diff W: {float(diff['diff_wind']):+.3f}")
    return " | ".join(lines) if lines else "(environmental diff unavailable)"
