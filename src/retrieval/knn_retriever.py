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

  contextaware_v2:
    shapeaware_v1 plus compact statistics from x_context and aux_context when
    long-range history is available.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

LEGACY_FEATURE_VERSION = "legacy_v0"
SHAPEAWARE_FEATURE_VERSION = "shapeaware_v1"
CONTEXTAWARE_FEATURE_VERSION = "contextaware_v2"
DEFAULT_QUERY_BATCH_SIZE = 128
DEFAULT_CUDA_CORPUS_CHUNK_SIZE = 65536


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
    parts = [
        x_hist.mean(axis=0),
        x_hist.std(axis=0),
        x_hist[-1],
        _clamped_delta(x_hist, lookback=2),
        _clamped_delta(x_hist, lookback=4),
        time_feat.mean(axis=0),
    ]
    aux_hist = sample.get("aux_hist")
    if aux_hist is not None:
        aux = np.asarray(aux_hist, dtype=np.float32)
        parts.extend([
            aux.mean(axis=0),
            aux.std(axis=0),
            aux[-1],
        ])
    aux_future = sample.get("aux_future")
    if aux_future is not None:
        aux_f = np.asarray(aux_future, dtype=np.float32)
        parts.extend([
            aux_f.mean(axis=0),
            aux_f.min(axis=0),
            aux_f.max(axis=0),
            aux_f[-1] - aux_f[0],
        ])
    return np.concatenate(parts).astype(np.float32)


def _context_stats(arr: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.size == 0:
        width = values.shape[1] if values.ndim == 2 else 1
        z = np.zeros(width, dtype=np.float32)
        return [z, z, z, z, z]

    half = max(1, values.shape[0] // 2)
    first = values[:half]
    second = values[-half:]
    return [
        values.mean(axis=0),
        values.std(axis=0),
        values.min(axis=0),
        values.max(axis=0),
        second.mean(axis=0) - first.mean(axis=0),
    ]


def _build_contextaware_vector(sample: dict) -> np.ndarray:
    parts = [_build_shapeaware_vector(sample)]
    x_context = sample.get("x_context")
    if x_context is not None:
        parts.extend(_context_stats(np.asarray(x_context, dtype=np.float32)))

    aux_context = sample.get("aux_context")
    if aux_context is not None:
        parts.extend(_context_stats(np.asarray(aux_context, dtype=np.float32)))

    return np.concatenate(parts).astype(np.float32)


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
        elif feature_version == CONTEXTAWARE_FEATURE_VERSION:
            vecs.append(_build_contextaware_vector(s))
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

    def __init__(
        self,
        pool_samples: list[dict],
        top_k: int = 2,
        *,
        query_device: str = "cpu",
        query_batch_size: int = DEFAULT_QUERY_BATCH_SIZE,
        corpus_chunk_size: int = DEFAULT_CUDA_CORPUS_CHUNK_SIZE,
    ):
        self.pool   = pool_samples
        self.top_k  = top_k
        self.feature_version = CONTEXTAWARE_FEATURE_VERSION
        self.vecs = build_retrieval_vectors(
            pool_samples,
            feature_version=self.feature_version,
        )
        self.t_starts = np.array([int(sample.get("t_start", -1)) for sample in pool_samples], dtype=np.int32)
        self.query_device = query_device or "cpu"
        self.query_batch_size = max(1, int(query_batch_size))
        self.corpus_chunk_size = max(1, int(corpus_chunk_size))
        self._resolved_query_device: str | None = None
        self._gpu_vecs = None
        self._gpu_t_starts = None
        self._gpu_vec_norms = None

    def configure_query_backend(
        self,
        *,
        query_device: str | None = None,
        query_batch_size: int | None = None,
        corpus_chunk_size: int | None = None,
    ) -> None:
        if query_device is not None:
            self.query_device = query_device or "cpu"
            self._resolved_query_device = None
            self._gpu_vecs = None
            self._gpu_t_starts = None
            self._gpu_vec_norms = None
        if query_batch_size is not None:
            self.query_batch_size = max(1, int(query_batch_size))
        if corpus_chunk_size is not None:
            self.corpus_chunk_size = max(1, int(corpus_chunk_size))

    @property
    def resolved_query_device(self) -> str:
        if self._resolved_query_device is not None:
            return self._resolved_query_device

        requested = str(self.query_device or "cpu").strip().lower()
        if requested in {"", "cpu"}:
            self._resolved_query_device = "cpu"
        elif requested == "auto":
            self._resolved_query_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif requested.startswith("cuda"):
            if torch.cuda.is_available():
                self._resolved_query_device = requested
            else:
                print(
                    f"[retrieval] Requested GPU query backend {requested}, "
                    "but CUDA is unavailable. Falling back to CPU."
                )
                self._resolved_query_device = "cpu"
        else:
            raise ValueError(f"Unsupported query_device={self.query_device!r}")
        return self._resolved_query_device

    @property
    def uses_gpu(self) -> bool:
        return self.resolved_query_device.startswith("cuda")

    def _build_query_vector(self, sample: dict) -> np.ndarray:
        if self.feature_version == LEGACY_FEATURE_VERSION:
            return _build_legacy_vector(sample)
        if self.feature_version == SHAPEAWARE_FEATURE_VERSION:
            return _build_shapeaware_vector(sample)
        if self.feature_version == CONTEXTAWARE_FEATURE_VERSION:
            return _build_contextaware_vector(sample)
        raise ValueError(f"Unsupported feature_version={self.feature_version!r}")

    def _ensure_gpu_query_state(self) -> bool:
        if not self.uses_gpu:
            return False
        if self._gpu_vecs is not None and self._gpu_t_starts is not None and self._gpu_vec_norms is not None:
            return True

        try:
            device = torch.device(self.resolved_query_device)
            self._gpu_vecs = torch.from_numpy(self.vecs).to(device=device, dtype=torch.float32)
            self._gpu_t_starts = torch.from_numpy(self.t_starts).to(device=device, dtype=torch.int32)
            self._gpu_vec_norms = (self._gpu_vecs * self._gpu_vecs).sum(dim=1)
            return True
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    f"[retrieval] Failed to move retrieval bank to {self.resolved_query_device} "
                    f"({exc}). Falling back to CPU."
                )
                self._resolved_query_device = "cpu"
                self._gpu_vecs = None
                self._gpu_t_starts = None
                self._gpu_vec_norms = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            raise

    def _query_indices_from_vector_cpu(self, q: np.ndarray, exclude_t_start: int | None = None) -> np.ndarray:
        diffs = self.vecs - q
        dists = (diffs ** 2).sum(axis=1)
        rank_dists = np.round(dists.astype(np.float64), 6)
        rank_dists += np.arange(len(rank_dists), dtype=np.float64) * 1e-12

        if exclude_t_start is not None:
            rank_dists = rank_dists.copy()
            rank_dists[self.t_starts >= int(exclude_t_start)] = np.inf

        k = min(int(self.top_k), len(rank_dists))
        if k <= 0:
            return np.empty((0,), dtype=np.int64)

        top_idx = np.argpartition(rank_dists, k - 1)[:k]
        order = np.argsort(rank_dists[top_idx], kind="stable")
        return top_idx[order]

    def _query_indices_batch_gpu(
        self,
        query_vecs: np.ndarray,
        exclude_t_starts: list[int | None] | None = None,
    ) -> list[np.ndarray]:
        if not self._ensure_gpu_query_state():
            return [
                self._query_indices_from_vector_cpu(query_vecs[i], None if exclude_t_starts is None else exclude_t_starts[i])
                for i in range(len(query_vecs))
            ]

        device = torch.device(self.resolved_query_device)
        k = min(int(self.top_k), len(self.pool))
        if k <= 0:
            return [np.empty((0,), dtype=np.int64) for _ in range(len(query_vecs))]

        outputs: list[np.ndarray] = []
        for q_start in range(0, len(query_vecs), self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, len(query_vecs))
            query_batch = torch.from_numpy(query_vecs[q_start:q_end]).to(device=device, dtype=torch.float32)
            query_norms = (query_batch * query_batch).sum(dim=1)
            batch_size = query_batch.shape[0]
            fallback_no_history = [False] * batch_size

            best_rank_dists = torch.full((batch_size, k), float("inf"), device=device, dtype=torch.float64)
            best_indices = torch.full((batch_size, k), -1, device=device, dtype=torch.long)
            exclude_batch = None
            if exclude_t_starts is not None:
                exclude_values = [
                    -1 if exclude_t_starts[idx] is None else int(exclude_t_starts[idx])
                    for idx in range(q_start, q_end)
                ]
                fallback_no_history = [
                    exclude_values[row_idx] >= 0 and not np.any(self.t_starts < exclude_values[row_idx])
                    for row_idx in range(batch_size)
                ]
                exclude_batch = torch.tensor(exclude_values, device=device, dtype=torch.int32)

            for corpus_start in range(0, len(self.pool), self.corpus_chunk_size):
                corpus_end = min(corpus_start + self.corpus_chunk_size, len(self.pool))
                bank_chunk = self._gpu_vecs[corpus_start:corpus_end]
                bank_norms = self._gpu_vec_norms[corpus_start:corpus_end]
                dists = bank_norms.unsqueeze(0) + query_norms.unsqueeze(1) - 2.0 * (query_batch @ bank_chunk.T)
                dists = torch.clamp(dists, min=0.0)
                rank_dists = torch.round(dists.to(torch.float64) * 1_000_000.0) / 1_000_000.0
                rank_bias = (
                    torch.arange(corpus_start, corpus_end, device=device, dtype=torch.float64) * 1e-12
                ).unsqueeze(0)
                rank_dists = rank_dists + rank_bias

                if exclude_batch is not None:
                    invalid_mask = self._gpu_t_starts[corpus_start:corpus_end].unsqueeze(0) >= exclude_batch.unsqueeze(1)
                    rank_dists = rank_dists.masked_fill(invalid_mask, float("inf"))

                chunk_k = min(k, corpus_end - corpus_start)
                chunk_rank_dists, chunk_local_idx = torch.topk(rank_dists, k=chunk_k, dim=1, largest=False)
                chunk_indices = chunk_local_idx + corpus_start

                if chunk_k < k:
                    pad_cols = k - chunk_k
                    chunk_rank_dists = torch.cat(
                        [chunk_rank_dists, torch.full((batch_size, pad_cols), float("inf"), device=device, dtype=torch.float64)],
                        dim=1,
                    )
                    chunk_indices = torch.cat(
                        [chunk_indices, torch.full((batch_size, pad_cols), -1, device=device, dtype=torch.long)],
                        dim=1,
                    )

                merged_dists = torch.cat([best_rank_dists, chunk_rank_dists], dim=1)
                merged_indices = torch.cat([best_indices, chunk_indices], dim=1)
                keep = torch.argsort(merged_dists, dim=1)[:, :k]
                best_rank_dists = torch.gather(merged_dists, 1, keep)
                best_indices = torch.gather(merged_indices, 1, keep)

            batch_indices = best_indices.detach().cpu().numpy()
            batch_dists = best_rank_dists.detach().cpu().numpy()
            fallback_indices = np.arange(k, dtype=np.int64)
            for row_idx, (idx_row, dist_row) in enumerate(zip(batch_indices, batch_dists)):
                if fallback_no_history[row_idx]:
                    outputs.append(fallback_indices.copy())
                    continue
                valid = idx_row >= 0
                idx_valid = idx_row[valid].astype(np.int64)
                dist_valid = dist_row[valid]
                order = np.lexsort((idx_valid, dist_valid))
                outputs.append(idx_valid[order])

        return outputs

    def query_indices_batch(
        self,
        samples: list[dict],
        exclude_t_starts: list[int | None] | None = None,
    ) -> list[np.ndarray]:
        if not samples:
            return []
        if exclude_t_starts is not None and len(exclude_t_starts) != len(samples):
            raise ValueError("exclude_t_starts must have the same length as samples")

        query_vecs = build_retrieval_vectors(samples, feature_version=self.feature_version)
        if self.uses_gpu:
            return self._query_indices_batch_gpu(query_vecs, exclude_t_starts=exclude_t_starts)

        outputs = []
        for idx, q in enumerate(query_vecs):
            exclude_t_start = None if exclude_t_starts is None else exclude_t_starts[idx]
            outputs.append(self._query_indices_from_vector_cpu(q, exclude_t_start=exclude_t_start))
        return outputs

    def query_indices(self, sample: dict, exclude_t_start: int | None = None) -> np.ndarray:
        return self.query_indices_batch([sample], exclude_t_starts=[exclude_t_start])[0]

    def query(self, sample: dict, exclude_t_start: int | None = None) -> list[dict]:
        """
        Return top-k most similar pool samples.

        Parameters
        ----------
        sample : query sample
        exclude_t_start : if set, exclude pool samples with t_start >= exclude_t_start
                          (prevents future leakage)
        """
        top_idx = self.query_indices(sample, exclude_t_start=exclude_t_start)
        return [self.pool[int(i)] for i in top_idx]

    def query_batch(
        self,
        samples: list[dict],
        exclude_t_starts: list[int | None] | None = None,
    ) -> list[list[dict]]:
        return [
            [self.pool[int(i)] for i in top_idx]
            for top_idx in self.query_indices_batch(samples, exclude_t_starts=exclude_t_starts)
        ]

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
        obj.t_starts = np.array([int(sample.get("t_start", -1)) for sample in obj.pool], dtype=np.int32)
        obj.query_device = "cpu"
        obj.query_batch_size = DEFAULT_QUERY_BATCH_SIZE
        obj.corpus_chunk_size = DEFAULT_CUDA_CORPUS_CHUNK_SIZE
        obj._resolved_query_device = None
        obj._gpu_vecs = None
        obj._gpu_t_starts = None
        obj._gpu_vec_norms = None
        return obj
