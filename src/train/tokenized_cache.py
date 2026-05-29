"""
Tokenized dataset cache utilities for expert training.

These caches persist fully materialized `input_ids / attention_mask / labels`
items so training can skip expert expansion, expert-local retrieval building,
and prompt/tokenization on subsequent runs.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.windowing import window_stride_token

TOKENIZED_CACHE_VERSION = 2
TOKENIZED_CACHE_DIR = Path(os.environ.get("DIFFDORA_CACHE_ROOT", "/root/autodl-tmp/Diff-DoRA-cache")) / "tokenized_cache"
DEFAULT_TOKENIZED_CACHE_SHARD_SIZE = 25000


def _cap_token(value: int | None) -> str:
    if value is None or int(value) <= 0:
        return "full"
    return str(int(value))


def _neighbor_k_token(neighbor_k: int | None) -> str:
    if neighbor_k is None or int(neighbor_k) <= 0:
        return "all"
    return str(int(neighbor_k))


def default_expert_tokenized_cache_dir(
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int | None,
    window_stride: int,
    max_length: int,
    prompt_style: str,
    variant: str,
    max_samples_per_expert: int,
    retrieval_bank_max_samples_per_expert: int,
    context_history_len: int | None = None,
) -> Path:
    context_token = "" if context_history_len is None or int(context_history_len) <= int(history_len) else f"_ctx{int(context_history_len)}"
    return TOKENIZED_CACHE_DIR / (
        f"train_experts_{dataset}_h{horizon}_hist{history_len}{context_token}_"
        f"nbr{_neighbor_k_token(neighbor_k)}_step{window_stride_token(window_stride)}_"
        f"len{max_length}_{prompt_style}_"
        f"samples{_cap_token(max_samples_per_expert)}_"
        f"bank{_cap_token(retrieval_bank_max_samples_per_expert)}"
    ) / variant


def expert_split_tokenized_cache_path(
    cache_dir: str | Path,
    expert_id: int,
    split: str,
) -> Path:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split={split!r}")
    return Path(cache_dir) / f"expert_{int(expert_id)}_{split}.pkl"


def tokenized_cache_metadata(
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int | None,
    window_stride: int,
    max_length: int,
    prompt_style: str,
    variant: str,
    expert_id: int,
    split: str,
    use_rag: bool,
    include_env_diff: bool,
    max_samples_per_expert: int,
    retrieval_bank_max_samples_per_expert: int,
    sample_count: int | None = None,
    context_history_len: int | None = None,
) -> dict:
    metadata = {
        "version": TOKENIZED_CACHE_VERSION,
        "kind": "train_experts_tokenized",
        "dataset": dataset,
        "horizon": int(horizon),
        "history_len": int(history_len),
        "context_history_len": int(context_history_len or history_len),
        "neighbor_k": None if neighbor_k is None else int(neighbor_k),
        "window_stride": int(window_stride),
        "max_length": int(max_length),
        "prompt_style": prompt_style,
        "variant": variant,
        "expert_id": int(expert_id),
        "split": split,
        "use_rag": bool(use_rag),
        "include_env_diff": bool(include_env_diff),
        "max_samples_per_expert": int(max_samples_per_expert),
        "retrieval_bank_max_samples_per_expert": int(retrieval_bank_max_samples_per_expert),
    }
    if sample_count is not None:
        metadata["sample_count"] = int(sample_count)
    return metadata


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
        raise ValueError("tokenized-cache metadata mismatch: " + ", ".join(mismatches))


def can_use_tokenized_items_cache(
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


class TokenizedItemsCacheWriter:
    """Incrementally write tokenized cache shards and finalize a manifest."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        metadata: dict,
        shard_size: int = DEFAULT_TOKENIZED_CACHE_SHARD_SIZE,
    ):
        self.cache_path = Path(cache_path)
        self.metadata = metadata
        self.shard_size = max(1, int(shard_size))
        self.parts: list[str] = []
        self.item_count = 0
        self.part_idx = 0
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def add_items(self, items: list[dict]) -> None:
        if not items:
            return
        for start in range(0, len(items), self.shard_size):
            chunk = items[start:start + self.shard_size]
            part_path = _part_path(self.cache_path, self.part_idx)
            tmp_part_path = part_path.with_suffix(part_path.suffix + ".tmp")
            with open(tmp_part_path, "wb") as f:
                pickle.dump({"items": chunk}, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_part_path.replace(part_path)
            self.parts.append(part_path.name)
            self.item_count += len(chunk)
            size_mb = part_path.stat().st_size / 1024 / 1024
            print(f"[tokenized_cache] Saved shard {part_path} ({size_mb:.1f} MB)")
            self.part_idx += 1

    def finalize(self) -> Path:
        manifest = {
            "format": "sharded",
            "version": TOKENIZED_CACHE_VERSION,
            "metadata": self.metadata,
            "parts": self.parts,
            "item_count": self.item_count,
        }
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.cache_path)

        meta = {**self.metadata, "sample_count": self.item_count, "format": "sharded", "part_count": len(self.parts)}
        meta_path = _metadata_path(self.cache_path)
        tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with open(tmp_meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        tmp_meta_path.replace(meta_path)

        size_mb = self.cache_path.stat().st_size / 1024 / 1024
        print(f"[tokenized_cache] Saved manifest {self.cache_path} ({size_mb:.1f} MB) with {len(self.parts)} part(s)")
        return self.cache_path


def save_tokenized_items_cache_part(
    cache_path: str | Path,
    part_idx: int,
    items: list[dict],
) -> Path:
    path = _part_path(cache_path, part_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump({"items": items}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[tokenized_cache] Saved shard {path} ({size_mb:.1f} MB)")
    return path


def finalize_tokenized_items_cache(
    cache_path: str | Path,
    *,
    metadata: dict,
    part_names: list[str],
    item_count: int,
) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "sharded",
        "version": TOKENIZED_CACHE_VERSION,
        "metadata": metadata,
        "parts": part_names,
        "item_count": item_count,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)

    meta = {**metadata, "sample_count": item_count, "format": "sharded", "part_count": len(part_names)}
    meta_path = _metadata_path(path)
    tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    tmp_meta_path.replace(meta_path)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[tokenized_cache] Saved manifest {path} ({size_mb:.1f} MB) with {len(part_names)} part(s)")
    return path


def save_tokenized_items_cache(
    items: list[dict],
    cache_path: str | Path,
    *,
    metadata: dict,
    shard_size: int | None = None,
) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shard_size is not None and int(shard_size) > 0 and len(items) > int(shard_size):
        writer = TokenizedItemsCacheWriter(path, metadata=metadata, shard_size=int(shard_size))
        writer.add_items(items)
        return writer.finalize()
    payload = {
        "metadata": metadata,
        "items": items,
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)

    meta_path = _metadata_path(path)
    tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    tmp_meta_path.replace(meta_path)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[tokenized_cache] Saved {path} ({size_mb:.1f} MB)")
    return path


def load_tokenized_items_cache(
    cache_path: str | Path,
    *,
    expected_metadata: dict | None = None,
) -> list[dict]:
    path = Path(cache_path)
    if expected_metadata is not None:
        if not can_use_tokenized_items_cache(path, expected_metadata=expected_metadata):
            raise ValueError(f"Invalid or stale tokenized cache: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]
    if isinstance(payload, dict) and payload.get("format") == "sharded":
        items = []
        for part_name in payload.get("parts", []):
            part_path = path.parent / part_name
            with open(part_path, "rb") as f:
                part_payload = pickle.load(f)
            if isinstance(part_payload, dict) and "items" in part_payload:
                items.extend(part_payload["items"])
            else:
                items.extend(part_payload)
        return items
    return payload


class TokenizedItemsDataset(Dataset):
    """Dataset wrapper for cached tokenized items."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {k: torch.tensor(v) for k, v in item.items()}
