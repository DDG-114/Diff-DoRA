from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.load_st_evcdp import (
    TRAIN_ONLY_NORMALIZATION,
    load_st_evcdp,
    validate_st_evcdp_processed,
)
from src.retrieval.knn_retriever import KNNRetriever
from src.routing.build_labels import LABEL_CBD, LABEL_RES, build_routing_labels


DEFAULT_FEWSHOT_RATIOS = (0.05, 0.10, 0.20, 0.40, 1.00)
DEFAULT_ZEROSHOT_SOURCE_RATIOS = (0.20, 0.40, 0.60, 0.80)
DEFAULT_ZEROSHOT_HALF_TRAIN_RATIOS = (0.60, 0.80)
DEFAULT_ZEROSHOT_TEST_WINDOW_DIVISOR = 10
DEFAULT_SEED = 42
DEFAULT_WINDOW_STRIDE = 6
DEFAULT_MAX_TRAIN_ITEMS = 4000
MANIFEST_VERSION = 1

SHOT_RETRIEVAL_DIR = Path("data/retrieval_cache/shot")
SHOT_SAMPLE_DIR = Path("data/sample_cache/shot")
SHOT_MANIFEST_DIR = Path("data/manifests/st_evcdp")


def parse_ratio_csv(raw: str | None, defaults: tuple[float, ...]) -> list[float]:
    if raw is None:
        return [float(v) for v in defaults]
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        return [float(v) for v in defaults]
    return values


def ratio_key(value: float) -> str:
    return f"{float(value):.2f}"


def strict_processed_path(dataset: str, horizon: int) -> Path:
    if dataset != "st_evcdp":
        raise ValueError(f"Strict shot preprocessing currently supports only st_evcdp, got {dataset!r}.")
    return Path(f"data/processed/{dataset}_trainnorm_h{int(horizon)}.pkl")


def default_manifest_path(kind: str, dataset: str, horizon: int, window_stride: int, seed: int) -> Path:
    if dataset != "st_evcdp":
        raise ValueError(f"Strict shot manifests currently support only st_evcdp, got {dataset!r}.")
    return SHOT_MANIFEST_DIR / f"{kind}_{dataset}_h{int(horizon)}_step{int(window_stride)}_seed{int(seed)}.json"


def _fewshot_retrieval_cache_path(dataset: str, horizon: int, window_stride: int, ratio: float, seed: int) -> Path:
    return SHOT_RETRIEVAL_DIR / (
        f"fewshot_{dataset}_h{int(horizon)}_step{int(window_stride)}_"
        f"ratio{ratio_key(ratio)}_seed{int(seed)}.pkl"
    )


def _zeroshot_sample_cache_path(dataset: str, horizon: int, window_stride: int, ratio: float, seed: int) -> Path:
    return SHOT_SAMPLE_DIR / (
        f"zeroshot_{dataset}_h{int(horizon)}_step{int(window_stride)}_"
        f"src{ratio_key(ratio)}_seed{int(seed)}.pkl"
    )

def stable_manifest_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_manifest(payload: dict, path: str | Path) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")
    tmp_path.replace(manifest_path)
    return manifest_path


def load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    expected_hash = payload.get("manifest_hash")
    payload_wo_hash = dict(payload)
    payload_wo_hash.pop("manifest_hash", None)
    actual_hash = stable_manifest_hash(payload_wo_hash)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Manifest hash mismatch for {manifest_path}: expected {expected_hash}, got {actual_hash}"
        )
    payload["manifest_hash"] = actual_hash
    return payload


def manifest_complete(payload: dict) -> bool:
    if payload.get("kind") == "fewshot":
        entries = payload.get("ratios", {})
        return all(Path(entry["retrieval_cache_path"]).exists() for entry in entries.values())
    if payload.get("kind") == "zeroshot_moe":
        entries = payload.get("source_ratios", {})
        return all(Path(entry["sample_cache_path"]).exists() for entry in entries.values())
    return False


def _fewshot_manifest_matches(
    payload: dict,
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int,
    window_stride: int,
    seed: int,
    normalization_source: str,
    max_train_items: int,
    ratios: list[float],
) -> bool:
    return (
        payload.get("kind") == "fewshot"
        and payload.get("dataset") == dataset
        and int(payload.get("horizon", -1)) == int(horizon)
        and int(payload.get("history_len", -1)) == int(history_len)
        and int(payload.get("neighbor_k", -1)) == int(neighbor_k)
        and int(payload.get("window_stride", -1)) == int(window_stride)
        and int(payload.get("seed", -1)) == int(seed)
        and payload.get("normalization_source") == normalization_source
        and int(payload.get("max_train_items", -1)) == int(max_train_items)
        and sorted(payload.get("ratios", {}).keys()) == sorted(ratio_key(r) for r in ratios)
    )


def _zeroshot_manifest_matches(
    payload: dict,
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int,
    window_stride: int,
    seed: int,
    normalization_source: str,
    max_train_items: int,
    source_ratios: list[float],
    half_train_ratios: list[float],
    test_window_divisor: int,
) -> bool:
    existing_source_ratio_keys = sorted(payload.get("source_ratios", {}).keys())
    requested_source_ratio_keys = sorted(ratio_key(r) for r in source_ratios)
    existing_half_train_ratio_keys = sorted(payload.get("zeroshot_half_train_ratios", []))
    requested_half_train_ratio_keys = sorted(ratio_key(r) for r in half_train_ratios)
    return (
        payload.get("kind") == "zeroshot_moe"
        and payload.get("dataset") == dataset
        and int(payload.get("horizon", -1)) == int(horizon)
        and int(payload.get("history_len", -1)) == int(history_len)
        and int(payload.get("neighbor_k", -1)) == int(neighbor_k)
        and int(payload.get("window_stride", -1)) == int(window_stride)
        and int(payload.get("seed", -1)) == int(seed)
        and payload.get("normalization_source") == normalization_source
        and int(payload.get("max_train_items", -1)) == int(max_train_items)
        and int(payload.get("zeroshot_test_window_divisor", -1)) == int(test_window_divisor)
        and requested_half_train_ratio_keys == existing_half_train_ratio_keys
        and all(key in existing_source_ratio_keys for key in requested_source_ratio_keys)
    )


def save_pickle(obj, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(out_path)
    return out_path


def load_shot_dataset(
    dataset: str,
    *,
    horizon: int,
    normalization_source: str,
    force_reprocess: bool = False,
) -> dict:
    if dataset != "st_evcdp":
        raise ValueError(f"Shot preprocessing currently supports only st_evcdp, got {dataset!r}.")

    processed_path = None
    if normalization_source == TRAIN_ONLY_NORMALIZATION:
        processed_path = strict_processed_path(dataset, horizon)

    data = load_st_evcdp(
        processed_path=processed_path,
        force_reprocess=force_reprocess,
        normalization_source=normalization_source,
    )
    if normalization_source == TRAIN_ONLY_NORMALIZATION:
        validate_st_evcdp_processed(data, strict=True)
    return data


def _mask_occ_and_adj(
    occ: np.ndarray,
    adj: np.ndarray | None,
    keep_nodes: list[int],
) -> tuple[np.ndarray, np.ndarray | None]:
    keep = np.zeros(occ.shape[1], dtype=bool)
    keep[np.array(keep_nodes, dtype=int)] = True

    occ_masked = occ.copy()
    occ_masked[:, ~keep] = 0.0

    if adj is None:
        return occ_masked, None

    adj_masked = adj.copy().astype(np.float32)
    adj_masked[~keep, :] = 0.0
    adj_masked[:, ~keep] = 0.0
    return occ_masked, adj_masked


def _select_stratified_source_nodes(labels: np.ndarray, ratio: float, seed: int) -> tuple[list[int], dict]:
    total_nodes = len(labels)
    source_total = max(1, int(total_nodes * ratio))
    if source_total >= total_nodes:
        source_total = total_nodes - 1

    domain_nodes = {
        LABEL_CBD: [int(idx) for idx, label in enumerate(labels) if int(label) == LABEL_CBD],
        LABEL_RES: [int(idx) for idx, label in enumerate(labels) if int(label) == LABEL_RES],
    }
    quotas = {}
    remainders = []
    assigned = 0
    for label, nodes in domain_nodes.items():
        exact = len(nodes) * float(ratio)
        count = min(len(nodes), math.floor(exact))
        quotas[label] = count
        assigned += count
        remainders.append((exact - count, label))

    needed = max(0, source_total - assigned)
    for _, label in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if needed <= 0:
            break
        if quotas[label] < len(domain_nodes[label]):
            quotas[label] += 1
            needed -= 1

    source_nodes = []
    for label, nodes in domain_nodes.items():
        rng = random.Random(seed + int(round(ratio * 1000)) + int(label))
        picked = sorted(rng.sample(nodes, quotas[label]))
        source_nodes.extend(picked)

    source_nodes = sorted(source_nodes)
    target_nodes = sorted(idx for idx in range(total_nodes) if idx not in set(source_nodes))
    domain_counts = {
        "total": {
            "CBD": len(domain_nodes[LABEL_CBD]),
            "Residential": len(domain_nodes[LABEL_RES]),
        },
        "source": {
            "CBD": sum(1 for idx in source_nodes if int(labels[idx]) == LABEL_CBD),
            "Residential": sum(1 for idx in source_nodes if int(labels[idx]) == LABEL_RES),
        },
        "target": {
            "CBD": sum(1 for idx in target_nodes if int(labels[idx]) == LABEL_CBD),
            "Residential": sum(1 for idx in target_nodes if int(labels[idx]) == LABEL_RES),
        },
    }
    return source_nodes, domain_counts


def _build_common_raw_samples(
    splits: dict,
    *,
    horizon: int,
    history_len: int,
    neighbor_k: int,
    window_stride: int,
) -> tuple[list[dict], list[dict]]:
    train_raw = build_samples(
        splits["train"],
        splits["timestamps_train"],
        adj=splits.get("adj"),
        horizons=[horizon],
        history_len=history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
    )[horizon]
    test_raw = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[horizon],
        history_len=history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
    )[horizon]
    return train_raw, test_raw


def build_strict_fewshot_manifest(
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int,
    window_stride: int,
    seed: int,
    normalization_source: str,
    max_train_items: int,
    ratios: list[float],
    force_reprocess: bool = False,
    manifest_path: str | Path | None = None,
) -> dict:
    data = load_shot_dataset(
        dataset,
        horizon=horizon,
        normalization_source=normalization_source,
        force_reprocess=force_reprocess,
    )
    splits = build_splits(data, dataset)
    train_raw, test_raw = _build_common_raw_samples(
        splits,
        horizon=horizon,
        history_len=history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
    )
    node_count = splits["train"].shape[1]

    entries = {}
    for ratio in ratios:
        key = ratio_key(ratio)
        n_windows = max(1, int(len(train_raw) * ratio))
        selected_train_raw = train_raw[:n_windows]
        retrieval_cache_path = _fewshot_retrieval_cache_path(dataset, horizon, window_stride, ratio, seed)
        KNNRetriever(selected_train_raw, top_k=2).save(retrieval_cache_path)

        train_item_count_full = n_windows * node_count
        train_item_count = train_item_count_full if max_train_items <= 0 else min(max_train_items, train_item_count_full)
        entries[key] = {
            "ratio": float(ratio),
            "window_indices": list(range(n_windows)),
            "t_start_indices": [int(sample["t_start"]) for sample in selected_train_raw],
            "train_window_count": int(n_windows),
            "test_window_count": int(len(test_raw)),
            "node_count": int(node_count),
            "train_item_count_full": int(train_item_count_full),
            "train_item_count": int(train_item_count),
            "retrieval_cache_path": retrieval_cache_path.as_posix(),
        }

    payload = {
        "manifest_version": MANIFEST_VERSION,
        "kind": "fewshot",
        "dataset": dataset,
        "horizon": int(horizon),
        "history_len": int(history_len),
        "neighbor_k": int(neighbor_k),
        "window_stride": int(window_stride),
        "seed": int(seed),
        "max_train_items": int(max_train_items),
        "normalization_source": normalization_source,
        "processed_path": strict_processed_path(dataset, horizon).as_posix(),
        "train_window_count": int(len(train_raw)),
        "test_window_count": int(len(test_raw)),
        "node_count": int(node_count),
        "ratios": entries,
    }
    payload["manifest_hash"] = stable_manifest_hash(payload)
    save_manifest(payload, manifest_path or default_manifest_path("fewshot", dataset, horizon, window_stride, seed))
    return payload


def build_strict_zeroshot_manifest(
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int,
    window_stride: int,
    seed: int,
    normalization_source: str,
    max_train_items: int,
    source_ratios: list[float],
    half_train_ratios: list[float],
    test_window_divisor: int,
    force_reprocess: bool = False,
    manifest_path: str | Path | None = None,
) -> dict:
    data = load_shot_dataset(
        dataset,
        horizon=horizon,
        normalization_source=normalization_source,
        force_reprocess=force_reprocess,
    )
    splits = build_splits(data, dataset)
    full_train_raw, full_test_raw = _build_common_raw_samples(
        splits,
        horizon=horizon,
        history_len=history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
    )
    if int(test_window_divisor) <= 0:
        raise ValueError(f"test_window_divisor must be positive, got {test_window_divisor}")
    selected_test_window_count = max(1, len(full_test_raw) // int(test_window_divisor))
    selected_test_window_indices = list(range(selected_test_window_count))
    selected_test_raw = [full_test_raw[idx] for idx in selected_test_window_indices]
    half_train_ratio_keys = {ratio_key(value) for value in half_train_ratios}
    labels = build_routing_labels(splits["train"], splits.get("node_meta"))

    entries = {}
    for ratio in source_ratios:
        key = ratio_key(ratio)
        source_nodes, domain_counts = _select_stratified_source_nodes(labels, ratio, seed)
        source_set = set(source_nodes)
        target_nodes = sorted(idx for idx in range(len(labels)) if idx not in source_set)
        source_occ_train, source_adj_train = _mask_occ_and_adj(
            splits["train"],
            splits.get("adj"),
            source_nodes,
        )
        train_raw_full = build_samples(
            source_occ_train,
            splits["timestamps_train"],
            adj=source_adj_train,
            horizons=[horizon],
            history_len=history_len,
            neighbor_k=neighbor_k,
            window_stride=window_stride,
        )[horizon]
        selected_train_window_count = len(train_raw_full)
        if key in half_train_ratio_keys:
            selected_train_window_count = max(1, len(train_raw_full) // 2)
        selected_train_window_indices = list(range(selected_train_window_count))
        train_raw = [train_raw_full[idx] for idx in selected_train_window_indices]

        sample_cache_path = _zeroshot_sample_cache_path(dataset, horizon, window_stride, ratio, seed)

        cache_payload = {
            "version": MANIFEST_VERSION,
            "kind": "zeroshot_moe_masked_train",
            "dataset": dataset,
            "horizon": int(horizon),
            "history_len": int(history_len),
            "neighbor_k": int(neighbor_k),
            "window_stride": int(window_stride),
            "seed": int(seed),
            "source_ratio": float(ratio),
            "normalization_source": normalization_source,
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "train_window_indices": selected_train_window_indices,
            "train_t_start_indices": [int(sample["t_start"]) for sample in train_raw],
            "train_samples": train_raw,
        }
        save_pickle(cache_payload, sample_cache_path)

        train_item_count_full = len(train_raw) * len(source_nodes)
        train_item_count = train_item_count_full if max_train_items <= 0 else min(max_train_items, train_item_count_full)
        entries[key] = {
            "source_ratio": float(ratio),
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "eval_target_nodes": target_nodes,
            "domain_counts": domain_counts,
            "train_window_indices": selected_train_window_indices,
            "train_t_start_indices": [int(sample["t_start"]) for sample in train_raw],
            "masked_train_window_count": int(len(train_raw)),
            "test_window_indices": selected_test_window_indices,
            "test_t_start_indices": [int(sample["t_start"]) for sample in selected_test_raw],
            "test_window_count": int(len(selected_test_raw)),
            "train_item_count_full": int(train_item_count_full),
            "train_item_count": int(train_item_count),
            "sample_cache_path": sample_cache_path.as_posix(),
        }

    payload = {
        "manifest_version": MANIFEST_VERSION,
        "kind": "zeroshot_moe",
        "dataset": dataset,
        "horizon": int(horizon),
        "history_len": int(history_len),
        "neighbor_k": int(neighbor_k),
        "window_stride": int(window_stride),
        "seed": int(seed),
        "max_train_items": int(max_train_items),
        "normalization_source": normalization_source,
        "processed_path": strict_processed_path(dataset, horizon).as_posix(),
        "training_architecture": "hard_routing_moe",
        "zeroshot_half_train_ratios": sorted(half_train_ratio_keys),
        "zeroshot_test_window_divisor": int(test_window_divisor),
        "train_window_count": int(len(full_train_raw)),
        "test_window_count_full": int(len(full_test_raw)),
        "test_window_count": int(len(selected_test_raw)),
        "test_window_indices": selected_test_window_indices,
        "test_t_start_indices": [int(sample["t_start"]) for sample in selected_test_raw],
        "node_count": int(len(labels)),
        "source_ratios": entries,
    }
    payload["manifest_hash"] = stable_manifest_hash(payload)
    save_manifest(payload, manifest_path or default_manifest_path("zeroshot", dataset, horizon, window_stride, seed))
    return payload


def ensure_strict_fewshot_manifest(
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int,
    window_stride: int,
    seed: int,
    normalization_source: str,
    max_train_items: int,
    ratios: list[float] | None = None,
    manifest_path: str | Path | None = None,
    force_rebuild: bool = False,
) -> tuple[dict, Path]:
    resolved_path = Path(manifest_path) if manifest_path else default_manifest_path(
        "fewshot", dataset, horizon, window_stride, seed
    )
    if resolved_path.exists() and not force_rebuild:
        payload = load_manifest(resolved_path)
        if manifest_complete(payload) and _fewshot_manifest_matches(
            payload,
            dataset=dataset,
            horizon=horizon,
            history_len=history_len,
            neighbor_k=neighbor_k,
            window_stride=window_stride,
            seed=seed,
            normalization_source=normalization_source,
            max_train_items=max_train_items,
            ratios=ratios or [float(v) for v in DEFAULT_FEWSHOT_RATIOS],
        ):
            return payload, resolved_path
    payload = build_strict_fewshot_manifest(
        dataset=dataset,
        horizon=horizon,
        history_len=history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
        seed=seed,
        normalization_source=normalization_source,
        max_train_items=max_train_items,
        ratios=ratios or [float(v) for v in DEFAULT_FEWSHOT_RATIOS],
        force_reprocess=force_rebuild,
        manifest_path=resolved_path,
    )
    return payload, resolved_path


def ensure_strict_zeroshot_manifest(
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    neighbor_k: int,
    window_stride: int,
    seed: int,
    normalization_source: str,
    max_train_items: int,
    source_ratios: list[float] | None = None,
    half_train_ratios: list[float] | None = None,
    test_window_divisor: int = DEFAULT_ZEROSHOT_TEST_WINDOW_DIVISOR,
    manifest_path: str | Path | None = None,
    force_rebuild: bool = False,
) -> tuple[dict, Path]:
    resolved_path = Path(manifest_path) if manifest_path else default_manifest_path(
        "zeroshot", dataset, horizon, window_stride, seed
    )
    if resolved_path.exists() and not force_rebuild:
        payload = load_manifest(resolved_path)
        if manifest_complete(payload) and _zeroshot_manifest_matches(
            payload,
            dataset=dataset,
            horizon=horizon,
            history_len=history_len,
            neighbor_k=neighbor_k,
            window_stride=window_stride,
            seed=seed,
            normalization_source=normalization_source,
            max_train_items=max_train_items,
            source_ratios=source_ratios or [float(v) for v in DEFAULT_ZEROSHOT_SOURCE_RATIOS],
            half_train_ratios=half_train_ratios or [float(v) for v in DEFAULT_ZEROSHOT_HALF_TRAIN_RATIOS],
            test_window_divisor=test_window_divisor,
        ):
            return payload, resolved_path
    payload = build_strict_zeroshot_manifest(
        dataset=dataset,
        horizon=horizon,
        history_len=history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
        seed=seed,
        normalization_source=normalization_source,
        max_train_items=max_train_items,
        source_ratios=source_ratios or [float(v) for v in DEFAULT_ZEROSHOT_SOURCE_RATIOS],
        half_train_ratios=half_train_ratios or [float(v) for v in DEFAULT_ZEROSHOT_HALF_TRAIN_RATIOS],
        test_window_divisor=test_window_divisor,
        force_reprocess=force_rebuild,
        manifest_path=resolved_path,
    )
    return payload, resolved_path
