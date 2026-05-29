"""
Persistent retrieval-result caches for expert training.

These caches store, for each expanded expert sample:
- top-k retrieved pool indices
- precomputed diff features

This lets tokenization/pretraining reuse expensive retrieval work across runs.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

from src.data.windowing import window_stride_token
from src.train.tokenized_cache import _cap_token, _neighbor_k_token

RETRIEVAL_RESULT_CACHE_VERSION = 2
DEFAULT_RETRIEVAL_CACHE_SHARD_SIZE = 25000
RETRIEVAL_RESULT_CACHE_DIR = (
    Path(os.environ.get("DIFFDORA_CACHE_ROOT", "/root/autodl-tmp/Diff-DoRA-cache"))
    / "retrieval_result_cache"
)


def default_expert_retrieval_cache_dir(
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int | None,
    window_stride: int,
    prompt_style: str,
    variant: str,
    max_samples_per_expert: int,
    retrieval_bank_max_samples_per_expert: int,
    context_history_len: int | None = None,
) -> Path:
    context_token = "" if context_history_len is None or int(context_history_len) <= int(history_len) else f"_ctx{int(context_history_len)}"
    return RETRIEVAL_RESULT_CACHE_DIR / (
        f"train_experts_{dataset}_h{horizon}_hist{history_len}{context_token}_"
        f"nbr{_neighbor_k_token(neighbor_k)}_step{window_stride_token(window_stride)}_"
        f"{prompt_style}_"
        f"samples{_cap_token(max_samples_per_expert)}_"
        f"bank{_cap_token(retrieval_bank_max_samples_per_expert)}"
    ) / variant


def expert_split_retrieval_cache_path(
    cache_dir: str | Path,
    expert_id: int,
    split: str,
) -> Path:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split={split!r}")
    return Path(cache_dir) / f"expert_{int(expert_id)}_{split}.pkl"


def retrieval_result_cache_metadata(
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int | None,
    window_stride: int,
    prompt_style: str,
    variant: str,
    expert_id: int,
    split: str,
    top_k: int,
    use_rag: bool,
    include_env_diff: bool,
    max_samples_per_expert: int,
    retrieval_bank_max_samples_per_expert: int,
    sample_count: int | None = None,
    context_history_len: int | None = None,
) -> dict:
    metadata = {
        "version": RETRIEVAL_RESULT_CACHE_VERSION,
        "kind": "train_experts_retrieval",
        "dataset": dataset,
        "horizon": int(horizon),
        "history_len": int(history_len),
        "context_history_len": int(context_history_len or history_len),
        "neighbor_k": None if neighbor_k is None else int(neighbor_k),
        "window_stride": int(window_stride),
        "prompt_style": prompt_style,
        "variant": variant,
        "expert_id": int(expert_id),
        "split": split,
        "top_k": int(top_k),
        "use_rag": bool(use_rag),
        "include_env_diff": bool(include_env_diff),
        "max_samples_per_expert": int(max_samples_per_expert),
        "retrieval_bank_max_samples_per_expert": int(retrieval_bank_max_samples_per_expert),
    }
    if sample_count is not None:
        metadata["sample_count"] = int(sample_count)
    return metadata


def retrieval_result_cache_key(sample: dict, node_idx: int) -> str:
    return f"{int(sample.get('t_start', -1))}:{int(node_idx)}"


def _metadata_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_suffix(path.suffix + ".meta.json")


def _part_path(cache_path: str | Path, part_idx: int) -> Path:
    path = Path(cache_path)
    return path.with_name(f"{path.stem}.part{int(part_idx):03d}{path.suffix}")


def _validate_metadata(actual: dict, expected: dict) -> None:
    mismatches = []
    for key, value in expected.items():
        if actual.get(key) != value:
            mismatches.append(f"{key}={actual.get(key)!r} (expected {value!r})")
    if mismatches:
        raise ValueError("retrieval-cache metadata mismatch: " + ", ".join(mismatches))


def can_use_retrieval_result_cache(
    cache_path: str | Path,
    *,
    expected_metadata: dict,
) -> bool:
    path = Path(cache_path)
    meta_path = _metadata_path(path)
    if not path.exists() or not meta_path.exists():
        return False
    try:
        with open(meta_path, "r") as f:
            actual = json.load(f)
        _validate_metadata(actual, expected_metadata)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and payload.get("format") == "sharded":
            for part_name in payload.get("parts", []):
                part_path = path.parent / part_name
                if not part_path.exists():
                    return False
        return True
    except Exception:
        return False


class RetrievalResultCacheWriter:
    """Incrementally write retrieval cache shards and finalize a manifest."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        metadata: dict,
        shard_size: int = DEFAULT_RETRIEVAL_CACHE_SHARD_SIZE,
    ):
        self.cache_path = Path(cache_path)
        self.metadata = metadata
        self.shard_size = max(1, int(shard_size))
        self.parts: list[str] = []
        self.entry_count = 0
        self.part_idx = 0
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def add_entries(self, entries: dict[str, dict]) -> None:
        if not entries:
            return
        items = list(entries.items())
        for start in range(0, len(items), self.shard_size):
            chunk = dict(items[start:start + self.shard_size])
            part_path = _part_path(self.cache_path, self.part_idx)
            tmp_part = part_path.with_suffix(part_path.suffix + ".tmp")
            with open(tmp_part, "wb") as f:
                pickle.dump({"entries": chunk}, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_part.replace(part_path)
            self.parts.append(part_path.name)
            self.entry_count += len(chunk)
            size_mb = part_path.stat().st_size / 1024 / 1024
            print(f"[retrieval_cache] Saved shard {part_path} ({size_mb:.1f} MB)")
            self.part_idx += 1

    def finalize(self) -> Path:
        manifest = {
            "format": "sharded",
            "version": RETRIEVAL_RESULT_CACHE_VERSION,
            "metadata": self.metadata,
            "parts": self.parts,
            "entry_count": self.entry_count,
        }
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.cache_path)

        meta = {**self.metadata, "sample_count": self.entry_count, "format": "sharded", "part_count": len(self.parts)}
        meta_path = _metadata_path(self.cache_path)
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2)
        tmp_meta.replace(meta_path)

        size_mb = self.cache_path.stat().st_size / 1024 / 1024
        print(f"[retrieval_cache] Saved manifest {self.cache_path} ({size_mb:.1f} MB) with {len(self.parts)} part(s)")
        return self.cache_path


def save_retrieval_result_cache_part(
    cache_path: str | Path,
    part_idx: int,
    entries: dict[str, dict],
) -> Path:
    path = _part_path(cache_path, part_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump({"entries": entries}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[retrieval_cache] Saved shard {path} ({size_mb:.1f} MB)")
    return path


def finalize_retrieval_result_cache(
    cache_path: str | Path,
    *,
    metadata: dict,
    part_names: list[str],
    entry_count: int,
) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "sharded",
        "version": RETRIEVAL_RESULT_CACHE_VERSION,
        "metadata": metadata,
        "parts": part_names,
        "entry_count": entry_count,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)

    meta = {**metadata, "sample_count": entry_count, "format": "sharded", "part_count": len(part_names)}
    meta_path = _metadata_path(path)
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.replace(meta_path)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[retrieval_cache] Saved manifest {path} ({size_mb:.1f} MB) with {len(part_names)} part(s)")
    return path


def load_retrieval_result_cache(
    cache_path: str | Path,
    *,
    expected_metadata: dict | None = None,
) -> dict[str, dict]:
    path = Path(cache_path)
    if expected_metadata is not None and not can_use_retrieval_result_cache(path, expected_metadata=expected_metadata):
        raise ValueError(f"Invalid or stale retrieval cache: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "entries" in payload:
        return payload["entries"]
    if isinstance(payload, dict) and payload.get("format") == "sharded":
        entries: dict[str, dict] = {}
        for part_name in payload.get("parts", []):
            part_path = path.parent / part_name
            with open(part_path, "rb") as f:
                part_payload = pickle.load(f)
            if isinstance(part_payload, dict) and "entries" in part_payload:
                entries.update(part_payload["entries"])
            else:
                entries.update(part_payload)
        return entries
    return payload


def save_retrieval_result_cache(
    entries: dict[str, dict],
    cache_path: str | Path,
    *,
    metadata: dict,
    shard_size: int | None = None,
) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shard_size is not None and int(shard_size) > 0 and len(entries) > int(shard_size):
        writer = RetrievalResultCacheWriter(path, metadata=metadata, shard_size=int(shard_size))
        writer.add_entries(entries)
        return writer.finalize()
    payload = {
        "metadata": metadata,
        "entries": entries,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)

    meta_path = _metadata_path(path)
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    tmp_meta.replace(meta_path)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[retrieval_cache] Saved {path} ({size_mb:.1f} MB)")
    return path
