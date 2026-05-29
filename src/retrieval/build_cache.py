"""
src/retrieval/build_cache.py
-----------------------------
Pre-build and persist KNNRetriever indices for each dataset / horizon.

Usage:
  python -m src.retrieval.build_cache
  python -m src.retrieval.build_cache --datasets st_evcdp urbanev --horizons 3 6 9 12

Output:
  data/retrieval_cache/{dataset}_h{horizon}.pkl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data.build_splits   import build_splits
from src.data.build_samples  import build_samples
from src.data.loaders import DATASET_LOADERS
from src.data.windowing import default_retrieval_cache_path, resolve_window_stride
from src.retrieval.knn_retriever import KNNRetriever

CACHE_DIR = Path("data/retrieval_cache")


def build_and_save(
    dataset: str,
    horizon: int,
    *,
    window_stride: int,
    output_path: Path | None = None,
) -> Path:
    print(f"\n[build_cache] {dataset}  horizon={horizon}  window_stride={window_stride}")
    t0 = time.time()

    raw    = DATASET_LOADERS[dataset]()
    splits = build_splits(raw, dataset)

    train_samples = build_samples(
        splits["train"],
        splits["timestamps_train"],
        adj=splits.get("adj"),
        horizons=[horizon],
        window_stride=window_stride,
    )[horizon]

    print(f"  pool size = {len(train_samples)}")

    retriever = KNNRetriever(train_samples, top_k=2)

    out_path = output_path or default_retrieval_cache_path(dataset, horizon, window_stride)
    retriever.save(out_path)

    elapsed = time.time() - t0
    vec_dim = retriever.vecs.shape[1]
    print(f"  saved → {out_path}  (vec_dim={vec_dim}, {elapsed:.1f}s)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["st_evcdp", "urbanev"], choices=sorted(DATASET_LOADERS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 9, 12])
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help="Window step size. `0` means use `horizon` (non-overlapping targets); `1` keeps classic overlapping sliding windows.",
    )
    parser.add_argument(
        "--output_path",
        default="",
        help="Explicit output path. Requires exactly one dataset and one horizon.",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_path and (len(args.datasets) != 1 or len(args.horizons) != 1):
        raise ValueError("--output_path requires exactly one dataset and one horizon.")

    saved = []
    for ds in args.datasets:
        for h in args.horizons:
            explicit_output = Path(args.output_path) if args.output_path else None
            effective_window_stride = resolve_window_stride(args.window_stride, horizon=h)
            path = build_and_save(
                ds,
                h,
                window_stride=effective_window_stride,
                output_path=explicit_output,
            )
            saved.append(path)

    print(f"\n✅  Built {len(saved)} retrieval cache(s):")
    for p in saved:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
