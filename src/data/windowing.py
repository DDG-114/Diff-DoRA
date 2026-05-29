from __future__ import annotations

from pathlib import Path

AUTO_WINDOW_STRIDE = 0
OVERLAPPING_WINDOW_STRIDE = 1


def non_overlapping_window_stride(horizon: int) -> int:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    return horizon


def resolve_window_stride(
    window_stride: int | None,
    *,
    horizon: int,
    default: int = OVERLAPPING_WINDOW_STRIDE,
) -> int:
    if window_stride is None:
        resolved = int(default)
    else:
        resolved = int(window_stride)
        if resolved == AUTO_WINDOW_STRIDE:
            resolved = non_overlapping_window_stride(horizon)
    if resolved <= 0:
        raise ValueError(f"window_stride must be positive after resolution, got {resolved}")
    return resolved


def window_stride_token(window_stride: int) -> str:
    stride = int(window_stride)
    if stride <= 0:
        raise ValueError(f"window_stride must be positive, got {stride}")
    return str(stride)


def default_retrieval_cache_path(dataset: str, horizon: int, window_stride: int) -> Path:
    return Path("data/retrieval_cache") / (
        f"{dataset}_h{int(horizon)}_step{window_stride_token(window_stride)}.pkl"
    )
