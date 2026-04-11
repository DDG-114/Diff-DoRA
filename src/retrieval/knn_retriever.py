"""
src/retrieval/knn_retriever.py
-------------------------------
KNN retriever for RAG.

For a query sample at time t we search the training pool (indices < t)
for the top-k most similar historical windows, measured by L2 distance
on a compact feature vector.

Retrieval vector for each sample:
  [ mean(x_hist), std(x_hist), mean_time_feat ]
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def build_retrieval_vectors(samples: list[dict]) -> np.ndarray:
    """
    Build (len(samples), D) feature matrix for KNN search.
    Features: mean_occ (N,), std_occ (N,), mean_time (4,)
    """
    vecs = []
    for s in samples:
        x  = s["x_hist"]                   # (12, N)
        tf = s["time_feat"]                # (12, 4)
        v  = np.concatenate([
            x.mean(axis=0),                # (N,)
            x.std(axis=0),                 # (N,)
            tf.mean(axis=0),               # (4,)
        ])
        vecs.append(v.astype(np.float32))
    return np.stack(vecs)                  # (M, D)


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
        self.vecs   = build_retrieval_vectors(pool_samples)  # (M, D)

    def query(self, sample: dict, exclude_t_start: int | None = None) -> list[dict]:
        """
        Return top-k most similar pool samples.

        Parameters
        ----------
        sample : query sample
        exclude_t_start : if set, exclude pool samples with t_start >= exclude_t_start
                          (prevents future leakage)
        """
        q = build_retrieval_vectors([sample])[0]  # (D,)
        diffs = self.vecs - q                      # (M, D)
        dists = (diffs ** 2).sum(axis=1)           # (M,)

        if exclude_t_start is not None:
            for i, s in enumerate(self.pool):
                if s["t_start"] >= exclude_t_start:
                    dists[i] = np.inf

        top_idx = np.argsort(dists)[:self.top_k]
        return [self.pool[i] for i in top_idx]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vecs": self.vecs, "pool": self.pool, "top_k": self.top_k}, f)

    @classmethod
    def load(cls, path: str | Path) -> "KNNRetriever":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls.__new__(cls)
        obj.pool   = d["pool"]
        obj.top_k  = d["top_k"]
        obj.vecs   = d["vecs"]
        return obj
