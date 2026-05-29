"""
Pre-build fully tokenized expert-training caches.

This script performs the expensive stages ahead of training:
1. Expand raw samples into per-expert node-tagged pools
2. Build expert-local retrievers
3. Materialize prompt/tokenized train+val datasets

Usage:
  python -m src.train.build_tokenized_expert_cache --dataset st_evcdp --horizon 6 --variant full
  python -m src.train.build_tokenized_expert_cache --dataset st_evcdp --horizon 6 --variant wo_diffdora
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from src.data.build_splits import build_splits
from src.data.loaders import DATASET_CHOICES, load_dataset
from src.data.sample_cache import default_expert_sample_cache_path, load_or_build_expert_sample_cache
from src.data.windowing import resolve_window_stride
from src.utils.history_window import price_at_history_end, weather_at_history_end
from src.models.qwen_peft import load_tokenizer
from src.retrieval.knn_retriever import KNNRetriever
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.retrieval_result_cache import (
    can_use_retrieval_result_cache,
    default_expert_retrieval_cache_dir,
    expert_split_retrieval_cache_path,
    retrieval_result_cache_metadata,
    finalize_retrieval_result_cache,
    retrieval_result_cache_key,
    save_retrieval_result_cache,
    save_retrieval_result_cache_part,
)
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router import HardRouter
from src.train.train_single import EVDataset
from src.utils.node_context import resolve_node_id
from src.train.tokenized_cache import (
    DEFAULT_TOKENIZED_CACHE_SHARD_SIZE,
    default_expert_tokenized_cache_dir,
    expert_split_tokenized_cache_path,
    finalize_tokenized_items_cache,
    save_tokenized_items_cache,
    save_tokenized_items_cache_part,
    tokenized_cache_metadata,
)


def _is_within_sample_cap(current_len: int, cap: int) -> bool:
    return cap <= 0 or current_len < cap


def _split_train_val(samples_with_node: list[dict]) -> tuple[list[dict], list[dict]]:
    val_split_idx = int(len(samples_with_node) * 0.85)
    return samples_with_node[:val_split_idx], samples_with_node[val_split_idx:]


def _batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start:start + max(1, batch_size)]


def _weather_at(weather, sample: dict) -> dict | None:
    return weather_at_history_end(weather, sample)


def _price_at(price, sample: dict, sample_node_idx: int, *, node_ids=None, node_meta=None) -> float | None:
    return price_at_history_end(price, sample, sample_node_idx, node_ids=node_ids, node_meta=node_meta)


def _build_chunk_retrieval_entries(
    split_samples: list[dict],
    retriever: KNNRetriever | None,
    splits: dict,
) -> dict[str, dict]:
    if retriever is None or not split_samples:
        return {}

    exclude_t_starts = [int(sample.get("t_start", -1)) for sample in split_samples]
    retrieved_indices_batch = retriever.query_indices_batch(
        split_samples,
        exclude_t_starts=exclude_t_starts,
    )

    retrieval_entries: dict[str, dict] = {}
    for sample, retrieved_idx in zip(split_samples, retrieved_indices_batch):
        sample_node_idx = int(sample.get("node_idx", 0))
        retrieved = [retriever.pool[int(i)] for i in retrieved_idx]
        weather_current = _weather_at(splits.get("weather"), sample)
        weather_retrieved = [_weather_at(splits.get("weather"), rs) for rs in retrieved]
        price_current = _price_at(
            splits.get("price"),
            sample,
            sample_node_idx,
            node_ids=splits.get("node_ids"),
            node_meta=splits.get("node_meta"),
        )
        price_retrieved = [
            _price_at(
                splits.get("price"),
                rs,
                sample_node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )
            for rs in retrieved
        ]
        diff = compute_diff_features(
            query_sample=sample,
            retrieved_samples=retrieved,
            weather_current=weather_current,
            weather_retrieved=weather_retrieved,
            price_current=price_current,
            price_retrieved=price_retrieved,
            node_idx=sample_node_idx,
        )
        retrieval_entries[retrieval_result_cache_key(sample, sample_node_idx)] = {
            "retrieved_indices": [int(i) for i in retrieved_idx],
            "diff": diff,
        }
    return retrieval_entries


_SHARD_CONTEXT = {}
_SHARD_TOKENIZER = None


def _set_shard_context(**kwargs) -> None:
    global _SHARD_CONTEXT
    _SHARD_CONTEXT = kwargs


def _init_shard_worker() -> None:
    global _SHARD_TOKENIZER
    _SHARD_TOKENIZER = load_tokenizer()


def _materialize_chunk_task(task: tuple[int, int, int]):
    global _SHARD_CONTEXT, _SHARD_TOKENIZER
    part_idx, start_idx, end_idx = task
    split_samples = _SHARD_CONTEXT["split_samples"][start_idx:end_idx]
    retrieval_entries = _build_chunk_retrieval_entries(
        split_samples,
        _SHARD_CONTEXT["retriever"],
        _SHARD_CONTEXT["splits"],
    )
    dataset_obj = EVDataset(
        split_samples,
        _SHARD_TOKENIZER,
        _SHARD_CONTEXT["horizon"],
        max_length=_SHARD_CONTEXT["max_length"],
        node_idx=0,
        use_rag=True,
        retriever=_SHARD_CONTEXT["retriever"],
        weather=_SHARD_CONTEXT["splits"].get("weather"),
        price=_SHARD_CONTEXT["splits"].get("price"),
        node_meta=_SHARD_CONTEXT["splits"].get("node_meta"),
        node_ids=_SHARD_CONTEXT["splits"].get("node_ids"),
        poi=_SHARD_CONTEXT["splits"].get("poi"),
        include_env_diff=_SHARD_CONTEXT["include_env_diff"],
        prompt_style=_SHARD_CONTEXT["prompt_style"],
        retrieval_cache_entries=retrieval_entries,
        precomputed_retrieval_entries=retrieval_entries,
    )
    if _SHARD_CONTEXT.get("disable_cache_sharding", False):
        return {
            "tokenized_part_name": None,
            "retrieval_part_name": None,
            "item_count": len(dataset_obj.items),
            "entry_count": len(retrieval_entries),
            "items": dataset_obj.items,
            "entries": retrieval_entries,
        }

    tokenized_part = save_tokenized_items_cache_part(
        _SHARD_CONTEXT["cache_path"],
        part_idx,
        dataset_obj.items,
    )
    retrieval_part = save_retrieval_result_cache_part(
        _SHARD_CONTEXT["retrieval_path"],
        part_idx,
        retrieval_entries,
    )
    return {
        "tokenized_part_name": tokenized_part.name,
        "retrieval_part_name": retrieval_part.name,
        "item_count": len(dataset_obj.items),
        "entry_count": len(retrieval_entries),
        "items": None,
        "entries": None,
    }


def _materialize_expert_caches(
    *,
    dataset: str,
    horizon: int,
    history_len: int,
    context_history_len: int | None,
    neighbor_k: int,
    window_stride: int,
    max_length: int,
    prompt_style: str,
    variant: str,
    include_env_diff: bool,
    expert_id: int,
    split_payloads: list[tuple[str, list[dict]]],
    retriever,
    splits: dict,
    tokenized_cache_dir: Path,
    retrieval_cache_dir: Path,
    max_samples_per_expert: int,
    retrieval_bank_max_samples_per_expert: int,
    force: bool,
    materialize_chunk_size: int,
    tokenized_cache_shard_size: int,
    num_workers: int,
    disable_cache_sharding: bool,
) -> None:
    for split_name, split_samples in split_payloads:
        cache_path = expert_split_tokenized_cache_path(tokenized_cache_dir, expert_id, split_name)
        retrieval_path = expert_split_retrieval_cache_path(retrieval_cache_dir, expert_id, split_name)
        if cache_path.exists() and not force:
            print(f"[tokenized_cache] Exists, skipping: {cache_path}")
            continue

        retrieval_meta = retrieval_result_cache_metadata(
            dataset=dataset,
            horizon=horizon,
            history_len=history_len,
            context_history_len=context_history_len,
            neighbor_k=neighbor_k,
            window_stride=window_stride,
            prompt_style=prompt_style,
            variant=variant,
            expert_id=expert_id,
            split=split_name,
            top_k=retriever.top_k,
            use_rag=True,
            include_env_diff=include_env_diff,
            max_samples_per_expert=max_samples_per_expert,
            retrieval_bank_max_samples_per_expert=retrieval_bank_max_samples_per_expert,
            sample_count=len(split_samples),
        )
        print(
            f"[tokenized_cache] Materializing {variant} "
            f"expert_{expert_id} {split_name} ({len(split_samples)} samples) …"
        )
        tokenized_meta = tokenized_cache_metadata(
            dataset=dataset,
            horizon=horizon,
            history_len=history_len,
            context_history_len=context_history_len,
            neighbor_k=neighbor_k,
            window_stride=window_stride,
            max_length=max_length,
            prompt_style=prompt_style,
            variant=variant,
            expert_id=expert_id,
            split=split_name,
            use_rag=True,
            include_env_diff=include_env_diff,
            max_samples_per_expert=max_samples_per_expert,
            retrieval_bank_max_samples_per_expert=retrieval_bank_max_samples_per_expert,
            sample_count=len(split_samples),
        )
        effective_chunk = min(max(1, materialize_chunk_size), max(1, tokenized_cache_shard_size))
        tasks = []
        part_idx = 0
        for start_idx in range(0, len(split_samples), effective_chunk):
            end_idx = min(start_idx + effective_chunk, len(split_samples))
            tasks.append((part_idx, start_idx, end_idx))
            part_idx += 1

        _set_shard_context(
            split_samples=split_samples,
            horizon=horizon,
            max_length=max_length,
            retriever=retriever,
            splits=splits,
            include_env_diff=include_env_diff,
            prompt_style=prompt_style,
            cache_path=cache_path,
            retrieval_path=retrieval_path,
            disable_cache_sharding=disable_cache_sharding,
        )

        use_process_pool = (
            num_workers > 1
            and os.name != "nt"
            and len(tasks) > 1
            and not getattr(retriever, "uses_gpu", False)
            and not disable_cache_sharding
        )
        if getattr(retriever, "uses_gpu", False) and num_workers > 1:
            print(
                "[tokenized_cache] GPU retrieval enabled; processing materialization chunks "
                "serially per variant to avoid duplicating the retrieval bank across worker processes."
            )
        if disable_cache_sharding:
            print(
                "[tokenized_cache] Cache sharding disabled; aggregating chunk results into a single "
                "cache file per split."
            )

        if use_process_pool:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=min(num_workers, len(tasks)), initializer=_init_shard_worker) as pool:
                results = pool.map(_materialize_chunk_task, tasks)
        else:
            _init_shard_worker()
            results = [_materialize_chunk_task(task) for task in tasks]

        item_count = sum(row["item_count"] for row in results)
        entry_count = sum(row["entry_count"] for row in results)
        if disable_cache_sharding:
            all_items = []
            all_entries: dict[str, dict] = {}
            for row in results:
                all_items.extend(row["items"] or [])
                all_entries.update(row["entries"] or {})
            save_tokenized_items_cache(
                all_items,
                cache_path,
                metadata={**tokenized_meta, "sample_count": item_count},
                shard_size=0,
            )
            save_retrieval_result_cache(
                all_entries,
                retrieval_path,
                metadata={**retrieval_meta, "sample_count": entry_count},
                shard_size=0,
            )
        else:
            tokenized_parts = [row["tokenized_part_name"] for row in results]
            retrieval_parts = [row["retrieval_part_name"] for row in results]
            finalize_tokenized_items_cache(
                cache_path,
                metadata=tokenized_meta,
                part_names=tokenized_parts,
                item_count=item_count,
            )
            finalize_retrieval_result_cache(
                retrieval_path,
                metadata=retrieval_meta,
                part_names=retrieval_parts,
                entry_count=entry_count,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="st_evcdp", choices=DATASET_CHOICES)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--context_history_len", type=int, default=0,
                        help="Optional long-context history window. 0 means use history_len only.")
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help="Window step size. `0` means use `horizon` (non-overlapping targets); `1` keeps classic overlapping sliding windows.",
    )
    parser.add_argument("--max_length", type=int, default=2560)
    parser.add_argument("--prompt_style", choices=sorted(EVDataset.PROMPT_STYLES), default="cot")
    parser.add_argument("--variant", choices=["full", "wo_diffdora"], required=True)
    parser.add_argument("--sample_cache", default="")
    parser.add_argument("--output_dir", default="",
                        help="Optional explicit tokenized-cache dir; defaults to data/tokenized_cache/...")
    parser.add_argument("--retrieval_cache_dir", default="",
                        help="Optional explicit retrieval-result-cache dir; defaults to data/retrieval_result_cache/...")
    parser.add_argument("--max_samples_per_expert", type=int, default=0)
    parser.add_argument("--retrieval_bank_max_samples_per_expert", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of expert workers for tokenized cache materialization. On Linux, 2 runs expert_0 and expert_1 in parallel.")
    parser.add_argument("--materialize_chunk_size", type=int, default=5000,
                        help="Number of expanded samples to tokenize per chunk before flushing shard files.")
    parser.add_argument("--tokenized_cache_shard_size", type=int, default=DEFAULT_TOKENIZED_CACHE_SHARD_SIZE,
                        help="Maximum number of items per tokenized cache shard file.")
    parser.add_argument("--retrieval_device", default="cpu",
                        help="Retrieval query backend: cpu, auto, or a CUDA device like cuda:0.")
    parser.add_argument("--retrieval_query_batch_size", type=int, default=128,
                        help="Number of query samples to score per GPU retrieval batch.")
    parser.add_argument("--retrieval_corpus_chunk_size", type=int, default=65536,
                        help="Number of bank vectors to compare per GPU retrieval chunk.")
    parser.add_argument("--disable_cache_sharding", action="store_true",
                        help="Aggregate chunk results into a single cache file per split instead of part files.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    include_env_diff = args.variant == "full"
    if include_env_diff and args.prompt_style == "vanilla":
        raise ValueError("variant=full is incompatible with prompt_style=vanilla")

    effective_window_stride = resolve_window_stride(args.window_stride, horizon=args.horizon)
    raw = load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)

    sample_cache_path = (
        Path(args.sample_cache)
        if args.sample_cache
        else default_expert_sample_cache_path(
            args.dataset,
            args.horizon,
            args.history_len,
            args.neighbor_k,
            effective_window_stride,
            context_history_len=args.context_history_len,
            include_test=False,
        )
    )
    sample_cache, resolved_sample_cache_path = load_or_build_expert_sample_cache(
        splits=splits,
        dataset=args.dataset,
        horizon=args.horizon,
        history_len=args.history_len,
        context_history_len=args.context_history_len,
        neighbor_k=args.neighbor_k,
        window_stride=effective_window_stride,
        cache_path=sample_cache_path,
        include_test=False,
        force_rebuild=False,
    )
    print(f"[tokenized_cache] Using sample cache: {resolved_sample_cache_path}")

    labels = build_routing_labels(splits["train"], raw.get("node_meta"))
    router = HardRouter(labels)
    n_nodes = splits["train"].shape[1]

    raw_samples = sample_cache["train_samples"]
    tagged: dict[int, list] = {0: [], 1: []}
    print("[tokenized_cache] Expanding per-node expert samples …")
    for sample in raw_samples:
        for node_idx in range(n_nodes):
            expert_id = router.route(node_idx)
            if _is_within_sample_cap(len(tagged[expert_id]), args.max_samples_per_expert):
                tagged[expert_id].append(dict(sample, node_idx=node_idx))
    print(f"[tokenized_cache] Expert sample counts: expert_0={len(tagged[0])}, expert_1={len(tagged[1])}")

    retrievers = {}
    for expert_id in (0, 1):
        if args.retrieval_bank_max_samples_per_expert > 0:
            expert_samples = tagged[expert_id][:args.retrieval_bank_max_samples_per_expert]
        else:
            expert_samples = tagged[expert_id]
        retrievers[expert_id] = KNNRetriever(
            expert_samples,
            top_k=2,
            query_device=args.retrieval_device,
            query_batch_size=args.retrieval_query_batch_size,
            corpus_chunk_size=args.retrieval_corpus_chunk_size,
        )
        print(
            f"[tokenized_cache] Built retriever expert_{expert_id} "
            f"with {len(expert_samples)} reference samples "
            f"(query_device={retrievers[expert_id].resolved_query_device})"
        )

    cache_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_expert_tokenized_cache_dir(
            args.dataset,
            args.horizon,
            args.history_len,
            args.neighbor_k,
            effective_window_stride,
            args.max_length,
            args.prompt_style,
            args.variant,
            args.max_samples_per_expert,
            args.retrieval_bank_max_samples_per_expert,
            context_history_len=args.context_history_len,
        )
    )
    retrieval_cache_dir = (
        Path(args.retrieval_cache_dir)
        if args.retrieval_cache_dir
        else default_expert_retrieval_cache_dir(
            args.dataset,
            args.horizon,
            args.history_len,
            args.neighbor_k,
            effective_window_stride,
            args.prompt_style,
            args.variant,
            args.max_samples_per_expert,
            args.retrieval_bank_max_samples_per_expert,
            context_history_len=args.context_history_len,
        )
    )
    tokenizer = load_tokenizer()

    for expert_id in (0, 1):
        train_samples, val_samples = _split_train_val(tagged[expert_id])
        _materialize_expert_caches(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            context_history_len=args.context_history_len,
            neighbor_k=args.neighbor_k,
            window_stride=effective_window_stride,
            max_length=args.max_length,
            prompt_style=args.prompt_style,
            variant=args.variant,
            include_env_diff=include_env_diff,
            expert_id=expert_id,
            split_payloads=[("train", train_samples), ("val", val_samples)],
            retriever=retrievers[expert_id],
            splits=splits,
            tokenized_cache_dir=cache_dir,
            retrieval_cache_dir=retrieval_cache_dir,
            max_samples_per_expert=args.max_samples_per_expert,
            retrieval_bank_max_samples_per_expert=args.retrieval_bank_max_samples_per_expert,
            force=args.force,
            materialize_chunk_size=args.materialize_chunk_size,
            tokenized_cache_shard_size=args.tokenized_cache_shard_size,
            num_workers=args.num_workers,
            disable_cache_sharding=args.disable_cache_sharding,
        )

    print(f"\n✅  Prepared tokenized caches under: {cache_dir}")
    print(f"✅  Retrieval caches under: {retrieval_cache_dir}")


if __name__ == "__main__":
    main()
