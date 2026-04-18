"""
src/retrieval/knn_retriever.py
-------------------------------
KNN retriever for RAG.

For a query sample at time t we search the training pool (indices < t)
for the top-k most similar historical windows, measured by L2 distance
on a compact feature vector.

Retrieval vector for each sample:
  legacy_v0:
    [ mean(x_hist), std(x_hist), mean_time_feat ]

  shapeaware_v1:
    [ mean(x_hist), std(x_hist), last(x_hist),
      last(x_hist)-prev(x_hist), last(x_hist)-x_hist[-4],
      mean_time_feat ]
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

LEGACY_FEATURE_VERSION = "legacy_v0"
SHAPEAWARE_FEATURE_VERSION = "shapeaware_v1"


def _clamped_delta(x_hist: np.ndarray, lookback: int) -> np.ndarray:
    """Return x[-1] - x[-lookback] with lookback clamped to available history."""
    anchor_idx = max(x_hist.shape[0] - lookback, 0)
    return x_hist[-1] - x_hist[anchor_idx]


def _build_legacy_vector(sample: dict) -> np.ndarray:
    x_hist = sample["x_hist"]
    time_feat = sample["time_feat"]
    return np.concatenate([
        x_hist.mean(axis=0),
        x_hist.std(axis=0),
        time_feat.mean(axis=0),
    ]).astype(np.float32)


def _build_shapeaware_vector(sample: dict) -> np.ndarray:
    x_hist = sample["x_hist"]
    time_feat = sample["time_feat"]
    return np.concatenate([
        x_hist.mean(axis=0),
        x_hist.std(axis=0),
        x_hist[-1],
        _clamped_delta(x_hist, lookback=2),
        _clamped_delta(x_hist, lookback=4),
        time_feat.mean(axis=0),
    ]).astype(np.float32)


def build_retrieval_vectors(
    samples: list[dict],
    *,
    feature_version: str = SHAPEAWARE_FEATURE_VERSION,
) -> np.ndarray:
    """
    Build (len(samples), D) feature matrix for KNN search.
    """
    vecs = []
    for s in samples:
        if feature_version == LEGACY_FEATURE_VERSION:
            vecs.append(_build_legacy_vector(s))
        elif feature_version == SHAPEAWARE_FEATURE_VERSION:
            vecs.append(_build_shapeaware_vector(s))
        else:
            raise ValueError(f"Unsupported feature_version={feature_version!r}")
    return np.stack(vecs)


class KNNRetriever:
    """
    Flat L2 KNN retriever (no external library required).

    Parameters
    ----------
    pool_samples : list[dict]   – the training samples forming the pool
    top_k        : int          – number of neighbours to return
    """

    def __init__(self, pool_samples: list[dict], top_k: int = 2):
        self.pool   = pool_samples
        self.top_k  = top_k
        self.feature_version = SHAPEAWARE_FEATURE_VERSION
        self.vecs = build_retrieval_vectors(
            pool_samples,
            feature_version=self.feature_version,
        )

    def query(self, sample: dict, exclude_t_start: int | None = None) -> list[dict]:
        """
        Return top-k most similar pool samples.

        Parameters
        ----------
        sample : query sample
        exclude_t_start : if set, exclude pool samples with t_start >= exclude_t_start
                          (prevents future leakage)
        """
        q = build_retrieval_vectors(
            [sample],
            feature_version=self.feature_version,
        )[0]
        diffs = self.vecs - q
        dists = (diffs ** 2).sum(axis=1)

        if exclude_t_start is not None:
            for i, s in enumerate(self.pool):
                if s["t_start"] >= exclude_t_start:
                    dists[i] = np.inf

        top_idx = np.argsort(dists)[:self.top_k]
        return [self.pool[i] for i in top_idx]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vecs": self.vecs,
                "pool": self.pool,
                "top_k": self.top_k,
                "feature_version": self.feature_version,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "KNNRetriever":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls.__new__(cls)
        obj.pool   = d["pool"]
        obj.top_k  = d["top_k"]
        obj.vecs   = d["vecs"]
        obj.feature_version = d.get("feature_version", LEGACY_FEATURE_VERSION)
        return obj
