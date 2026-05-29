from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_daylevel_prediction_map(csv_path: str | Path) -> dict[tuple[str, int], float]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    if not {"day", "slot", "prediction"}.issubset(df.columns):
        raise ValueError(f"Expected columns day, slot, prediction in {path}")
    return {
        (str(row.day), int(row.slot)): float(row.prediction)
        for row in df.itertuples(index=False)
    }


def attach_candidate_curve(
    sample: dict,
    *,
    horizon: int,
    prediction_map: dict[tuple[str, int], float] | None,
) -> dict:
    """Attach a precomputed candidate curve to a sample when available."""
    if not prediction_map:
        return sample

    target_start = sample.get("target_start_timestamp")
    if target_start is None:
        return sample

    ts = pd.Timestamp(target_start)
    day_str = str(ts.normalize().date())
    start_slot = int(ts.hour * 4 + ts.minute // 15)

    candidate = []
    for step in range(horizon):
        slot = start_slot + step
        lookup_day = day_str
        if slot >= 96:
            slot = slot % 96
            lookup_day = str((ts.normalize() + pd.Timedelta(days=1)).date())
        value = prediction_map.get((lookup_day, int(slot)))
        if value is None:
            return sample
        candidate.append(float(value))

    enriched = dict(sample)
    enriched["candidate_future"] = np.asarray(candidate, dtype=np.float32)
    return enriched


def build_candidate_refine_mask(
    candidate: np.ndarray,
    *,
    peak_threshold: float = 0.45,
    ramp_threshold: float = 0.08,
    curvature_threshold: float = 0.05,
) -> np.ndarray:
    """Flag candidate steps that are worth selective LLM refinement.

    All thresholds operate on the normalized price scale used by ``gs_price_2025``.
    """
    arr = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    prev = np.concatenate([arr[:1], arr[:-1]])
    nxt = np.concatenate([arr[1:], arr[-1:]])
    ramp_prev = np.abs(arr - prev)
    ramp_next = np.abs(nxt - arr)
    curvature = np.abs(nxt - 2.0 * arr + prev)

    mask = (
        (arr >= peak_threshold)
        | (ramp_prev >= ramp_threshold)
        | (ramp_next >= ramp_threshold)
        | (curvature >= curvature_threshold)
    )

    # Dilate by one step on each side so the LLM can adjust transition zones too.
    if mask.any():
        padded = np.pad(mask.astype(np.int32), (1, 1))
        mask = (padded[:-2] | padded[1:-1] | padded[2:]).astype(bool)
    return mask.astype(np.float32)


def attach_candidate_refine_mask(sample: dict) -> dict:
    candidate = sample.get("candidate_future")
    if candidate is None:
        return sample
    enriched = dict(sample)
    enriched["candidate_refine_mask"] = build_candidate_refine_mask(np.asarray(candidate, dtype=np.float32))
    return enriched


def combine_candidate_prediction(
    sample: dict,
    parsed: np.ndarray,
    *,
    mode: str = "absolute",
    residual_clip: float | None = None,
    value_clip: tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert a parsed model output into the final forecast sequence."""
    arr = np.asarray(parsed, dtype=np.float32).reshape(-1)
    if mode == "absolute":
        if value_clip is not None:
            arr = np.clip(arr, float(value_clip[0]), float(value_clip[1]))
        return arr
    if mode not in {"residual", "selective_residual", "chunk_offset"}:
        raise ValueError(f"Unsupported candidate combination mode: {mode!r}")

    candidate = sample.get("candidate_future")
    if candidate is None:
        raise ValueError("Residual mode requires sample['candidate_future'].")
    candidate_arr = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if len(candidate_arr) < len(arr):
        raise ValueError("Candidate curve shorter than parsed residual output.")
    residual = arr
    if residual_clip is not None:
        residual = np.clip(residual, -float(residual_clip), float(residual_clip))
    if mode == "chunk_offset":
        offset = float(np.mean(residual))
        residual = np.full_like(residual, offset)
    if mode == "selective_residual":
        mask = sample.get("candidate_refine_mask")
        if mask is None:
            raise ValueError("selective_residual mode requires sample['candidate_refine_mask'].")
        mask_arr = np.asarray(mask, dtype=np.float32).reshape(-1)[: len(residual)]
        residual = residual * mask_arr
    out = candidate_arr[: len(arr)] + residual
    if value_clip is not None:
        out = np.clip(out, float(value_clip[0]), float(value_clip[1]))
    return out
