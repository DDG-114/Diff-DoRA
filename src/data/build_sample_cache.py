"""
Pre-build and persist sample caches for expert training.

Usage:
  python -m src.data.build_sample_cache
  python -m src.data.build_sample_cache --datasets st_evcdp --horizons 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.build_splits import build_splits
from src.data.loaders import DATASET_LOADERS
from src.data.windowing import resolve_window_stride
from src.data.sample_cache import (
    default_expert_sample_cache_path,
    load_or_build_expert_sample_cache,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["st_evcdp", "urbanev"], choices=sorted(DATASET_LOADERS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 9, 12])
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument(
        "--context_history_len",
        type=int,
        default=0,
        help="Optional long-range history length. `0` keeps the classic single-window cache.",
    )
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help="Window step size. `0` means use `horizon` (non-overlapping targets); `1` keeps classic overlapping sliding windows.",
    )
    parser.add_argument("--include_test", action="store_true",
                        help="Also cache test split samples for post-train evaluation.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild cache even if a matching cache already exists.")
    parser.add_argument(
        "--output_path",
        default="",
        help="Explicit output path. Requires exactly one dataset and one horizon.",
    )
    args = parser.parse_args()

    if args.output_path and (len(args.datasets) != 1 or len(args.horizons) != 1):
        raise ValueError("--output_path requires exactly one dataset and one horizon.")

    saved = []
    for dataset in args.datasets:
        raw = DATASET_LOADERS[dataset]()
        splits = build_splits(raw, dataset)
        for horizon in args.horizons:
            effective_window_stride = resolve_window_stride(args.window_stride, horizon=horizon)
            cache_path = (
                Path(args.output_path)
                if args.output_path
                else default_expert_sample_cache_path(
                    dataset,
                    horizon,
                    args.history_len,
                    args.neighbor_k,
                    effective_window_stride,
                    context_history_len=args.context_history_len,
                    include_test=args.include_test,
                )
            )
            _, resolved_path = load_or_build_expert_sample_cache(
                splits=splits,
                dataset=dataset,
                horizon=horizon,
                history_len=args.history_len,
                context_history_len=args.context_history_len,
                neighbor_k=args.neighbor_k,
                window_stride=effective_window_stride,
                cache_path=cache_path,
                include_test=args.include_test,
                force_rebuild=args.force,
            )
            saved.append(resolved_path)

    print(f"\n✅  Prepared {len(saved)} sample cache(s):")
    for path in saved:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
