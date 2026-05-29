"""
Sample-cache utilities for expert training.

These caches persist pre-built sliding-window samples so repeated training runs
can skip the expensive `build_samples(...)` step.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

from src.data.build_samples import build_samples
from src.data.windowing import window_stride_token

SAMPLE_CACHE_VERSION = 2
SAMPLE_CACHE_DIR = Path(os.environ.get("DIFFDORA_CACHE_ROOT", "/root/autodl-tmp/Diff-DoRA-cache")) / "sample_cache"


def _neighbor_k_token(neighbor_k: int | None) -> str:
    if neighbor_k is None or int(neighbor_k) <= 0:
        return "all"
    return str(int(neighbor_k))


def _context_history_token(history_len: int, context_history_len: int | None) -> str:
    if context_history_len is None or int(context_history_len) <= int(history_len):
        return ""
    return f"_ctx{int(context_history_len)}"


def default_expert_sample_cache_path(
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int | None,
    window_stride: int,
    *,
    context_history_len: int | None = None,
    include_test: bool = False,
) -> Path:
    suffix = "_with_test" if include_test else ""
    return SAMPLE_CACHE_DIR / (
        f"train_experts_{dataset}_h{horizon}_hist{history_len}_"
        f"nbr{_neighbor_k_token(neighbor_k)}"
        f"{_context_history_token(history_len, context_history_len)}_"
        f"step{window_stride_token(window_stride)}{suffix}.pkl"
    )


def _validate_expert_sample_cache(
    cache: dict,
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int | None,
    window_stride: int,
    context_history_len: int | None,
    include_test: bool,
) -> None:
    expected = {
        "version": SAMPLE_CACHE_VERSION,
        "kind": "train_experts",
        "dataset": dataset,
        "horizon": int(horizon),
        "history_len": int(history_len),
        "context_history_len": int(context_history_len or history_len),
        "neighbor_k": None if neighbor_k is None else int(neighbor_k),
        "window_stride": int(window_stride),
    }
    mismatches = []
    for key, value in expected.items():
        if cache.get(key) != value:
            mismatches.append(f"{key}={cache.get(key)!r} (expected {value!r})")
    if mismatches:
        raise ValueError("cache metadata mismatch: " + ", ".join(mismatches))
    if "train_samples" not in cache or cache["train_samples"] is None:
        raise ValueError("cache is missing train_samples")
    if include_test and cache.get("test_samples") is None:
        raise ValueError("cache is missing test_samples")


def build_expert_sample_cache(
    *,
    splits: dict,
    dataset: str,
    horizon: int,
    history_len: int,
    context_history_len: int | None,
    neighbor_k: int | None,
    window_stride: int,
    include_test: bool = False,
) -> dict:
    print(
        "[sample_cache] Building expert sample cache "
        f"(dataset={dataset}, h={horizon}, history={history_len}, context_history={int(context_history_len or history_len)}, "
        f"neighbor_k={neighbor_k}, window_stride={window_stride}, include_test={include_test})"
    )
    t0 = time.time()
    train_map = build_samples(
        splits["train"],
        splits["timestamps_train"],
        adj=splits.get("adj"),
        horizons=[horizon],
        history_len=history_len,
        context_history_len=context_history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
    )
    test_samples = None
    if include_test:
        test_map = build_samples(
            splits["test"],
            splits["timestamps_test"],
            adj=splits.get("adj"),
            horizons=[horizon],
            history_len=history_len,
            context_history_len=context_history_len,
            neighbor_k=neighbor_k,
            window_stride=window_stride,
        )
        test_samples = test_map[horizon]

    elapsed = time.time() - t0
    train_samples = train_map[horizon]
    test_count = 0 if test_samples is None else len(test_samples)
    print(
        "[sample_cache] Built "
        f"train={len(train_samples)} test={test_count} "
        f"in {elapsed:.1f}s"
    )
    return {
        "version": SAMPLE_CACHE_VERSION,
        "kind": "train_experts",
        "dataset": dataset,
        "horizon": int(horizon),
        "history_len": int(history_len),
        "context_history_len": int(context_history_len or history_len),
        "neighbor_k": None if neighbor_k is None else int(neighbor_k),
        "window_stride": int(window_stride),
        "include_test": bool(include_test),
        "train_samples": train_samples,
        "test_samples": test_samples,
    }


def save_expert_sample_cache(cache: dict, path: str | Path) -> Path:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"[sample_cache] Saved {cache_path} ({size_mb:.1f} MB)")
    return cache_path


def load_expert_sample_cache(
    path: str | Path,
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    context_history_len: int | None,
    neighbor_k: int | None,
    window_stride: int,
    include_test: bool = False,
) -> dict:
    cache_path = Path(path)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    _validate_expert_sample_cache(
        cache,
        dataset=dataset,
        horizon=horizon,
        history_len=history_len,
        context_history_len=context_history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
        include_test=include_test,
    )
    test_count = 0 if cache.get("test_samples") is None else len(cache["test_samples"])
    print(
        f"[sample_cache] Loaded {cache_path} "
        f"(train={len(cache['train_samples'])}, test={test_count})"
    )
    return cache


def load_or_build_expert_sample_cache(
    *,
    splits: dict,
    dataset: str,
    horizon: int,
    history_len: int,
    context_history_len: int | None = None,
    neighbor_k: int | None,
    window_stride: int,
    cache_path: str | Path | None = None,
    include_test: bool = False,
    force_rebuild: bool = False,
) -> tuple[dict, Path]:
    resolved_path = Path(cache_path) if cache_path else default_expert_sample_cache_path(
        dataset,
        horizon,
        history_len,
        neighbor_k,
        window_stride,
        context_history_len=context_history_len,
        include_test=include_test,
    )

    if resolved_path.exists() and not force_rebuild:
        try:
            cache = load_expert_sample_cache(
                resolved_path,
                dataset=dataset,
                horizon=horizon,
                history_len=history_len,
                context_history_len=context_history_len,
                neighbor_k=neighbor_k,
                window_stride=window_stride,
                include_test=include_test,
            )
            return cache, resolved_path
        except Exception as exc:
            print(f"[sample_cache] Ignoring stale cache at {resolved_path}: {exc}")

    cache = build_expert_sample_cache(
        splits=splits,
        dataset=dataset,
        horizon=horizon,
        history_len=history_len,
        context_history_len=context_history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
        include_test=include_test,
    )
    save_expert_sample_cache(cache, resolved_path)
    return cache, resolved_path
