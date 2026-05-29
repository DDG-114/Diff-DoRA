from __future__ import annotations

import math

import numpy as np

from src.utils.history_window import weather_at_history_end


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _future_slope(sample: dict, node_idx: int) -> float:
    y = np.asarray(sample["y"][:, node_idx], dtype=np.float32)
    if y.size == 0:
        return 0.0
    return float(y[-1] - y[0])


def _history_slope(sample: dict, node_idx: int, lookback: int = 4) -> float:
    x = np.asarray(sample["x_hist"][:, node_idx], dtype=np.float32)
    if x.size == 0:
        return 0.0
    anchor = max(0, len(x) - lookback)
    return float(x[-1] - x[anchor])


def _series_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _irradiance_strength(weather_row: dict | None) -> float:
    if not weather_row:
        return 0.0
    candidates = [
        weather_row.get("total_solar_irradiance"),
        weather_row.get("ghi"),
        weather_row.get("global_horizontal_irradiance"),
        weather_row.get("dni"),
        weather_row.get("direct_normal_irradiance"),
    ]
    return max(_safe_float(value, default=0.0) for value in candidates)


def source_sample_score(
    sample: dict,
    *,
    node_idx: int,
    weather=None,
    volatility_weight: float = 1.0,
    history_ramp_weight: float = 1.0,
    future_ramp_weight: float = 1.0,
    irradiance_weight: float = 0.001,
) -> float:
    """Heuristic active-learning score for renewable source windows."""
    x = np.asarray(sample["x_hist"][:, node_idx], dtype=np.float32)
    y = np.asarray(sample["y"][:, node_idx], dtype=np.float32)
    weather_row = weather_at_history_end(weather, sample)

    volatility = _series_std(x) + _series_std(y)
    hist_ramp = abs(_history_slope(sample, node_idx=node_idx))
    future_ramp = abs(_future_slope(sample, node_idx=node_idx))
    irradiance = _irradiance_strength(weather_row)

    score = (
        volatility_weight * volatility
        + history_ramp_weight * hist_ramp
        + future_ramp_weight * future_ramp
        + irradiance_weight * irradiance
    )
    if not math.isfinite(score):
        return 0.0
    return float(score)


def price_dynamic_sample_score(
    sample: dict,
    *,
    node_idx: int,
    floor_value: float = 0.04,
) -> float:
    """Score electricity-price windows by future shape information.

    Electricity price data can contain many floor-price intervals. For a
    day-ahead task, the informative samples are usually the ones with clear
    intra-day range, peaks, or departures from the price floor.
    """
    x = np.asarray(sample["x_hist"][:, node_idx], dtype=np.float32)
    y = np.asarray(sample["y"][:, node_idx], dtype=np.float32)
    if y.size == 0:
        return 0.0

    future_std = _series_std(y)
    future_range = float(np.max(y) - np.min(y))
    floor_gap = float(np.mean(np.abs(y - floor_value)))
    peak_gap = float(max(0.0, np.max(y) - floor_value))
    hist_std = _series_std(x)
    future_ramp = abs(_future_slope(sample, node_idx=node_idx))

    score = (
        2.0 * future_std
        + 1.5 * future_range
        + 1.0 * floor_gap
        + 0.5 * peak_gap
        + 0.5 * hist_std
        + 0.5 * future_ramp
    )
    if not math.isfinite(score):
        return 0.0
    return float(score)


def select_active_source_items(
    raw_samples: list[dict],
    node_indices: list[int],
    *,
    weather=None,
    budget_ratio: float = 0.5,
    max_items: int = 0,
) -> list[dict]:
    """
    Build per-node training items, score them, and keep the top budget slice.

    The output format matches the tagged samples used by shared-adapter training.
    """
    if not raw_samples or not node_indices:
        return []

    scored: list[tuple[float, int, dict]] = []
    counter = 0
    for sample in raw_samples:
        for node_idx in node_indices:
            tagged = dict(sample, node_idx=int(node_idx))
            score = source_sample_score(sample, node_idx=int(node_idx), weather=weather)
            scored.append((score, counter, tagged))
            counter += 1

    scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    total = len(scored)
    if max_items > 0:
        keep = min(max_items, total)
    else:
        keep = max(1, int(math.ceil(total * float(budget_ratio))))
    return [tagged for _score, _order, tagged in scored[:keep]]


def select_price_dynamic_items(
    raw_samples: list[dict],
    node_indices: list[int],
    *,
    budget_ratio: float = 0.7,
    max_items: int = 0,
    floor_value: float = 0.04,
) -> list[dict]:
    """Keep price windows with the most useful day-ahead dynamics."""
    if not raw_samples or not node_indices:
        return []

    scored: list[tuple[float, int, dict]] = []
    counter = 0
    for sample in raw_samples:
        for node_idx in node_indices:
            tagged = dict(sample, node_idx=int(node_idx))
            score = price_dynamic_sample_score(
                sample,
                node_idx=int(node_idx),
                floor_value=floor_value,
            )
            scored.append((score, counter, tagged))
            counter += 1

    scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    total = len(scored)
    if max_items > 0:
        keep = min(max_items, total)
    else:
        keep = max(1, int(math.ceil(total * float(budget_ratio))))
    return [tagged for _score, _order, tagged in scored[:keep]]
