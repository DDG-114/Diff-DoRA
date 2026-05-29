from __future__ import annotations

import argparse

from src.data.load_st_evcdp import TRAIN_ONLY_NORMALIZATION
from src.data.shot_manifests import (
    DEFAULT_FEWSHOT_RATIOS,
    DEFAULT_MAX_TRAIN_ITEMS,
    DEFAULT_SEED,
    DEFAULT_WINDOW_STRIDE,
    DEFAULT_ZEROSHOT_HALF_TRAIN_RATIOS,
    DEFAULT_ZEROSHOT_SOURCE_RATIOS,
    DEFAULT_ZEROSHOT_TEST_WINDOW_DIVISOR,
    ensure_strict_fewshot_manifest,
    ensure_strict_zeroshot_manifest,
    parse_ratio_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="st_evcdp", choices=["st_evcdp"])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)
    parser.add_argument("--window_stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_train_items", type=int, default=DEFAULT_MAX_TRAIN_ITEMS)
    parser.add_argument(
        "--normalization_source",
        choices=["train_only"],
        default=TRAIN_ONLY_NORMALIZATION,
    )
    parser.add_argument("--fewshot_ratios", default=None)
    parser.add_argument("--source_ratios", default=None)
    parser.add_argument("--zeroshot_half_train_ratios", default=None)
    parser.add_argument("--zeroshot_test_window_divisor", type=int, default=DEFAULT_ZEROSHOT_TEST_WINDOW_DIVISOR)
    parser.add_argument("--skip_fewshot", action="store_true")
    parser.add_argument("--skip_zeroshot", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.skip_fewshot:
        ratios = parse_ratio_csv(args.fewshot_ratios, DEFAULT_FEWSHOT_RATIOS)
        manifest, path = ensure_strict_fewshot_manifest(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.window_stride,
            seed=args.seed,
            normalization_source=args.normalization_source,
            max_train_items=args.max_train_items,
            ratios=ratios,
            force_rebuild=args.force,
        )
        print(f"[fewshot] manifest: {path}")
        print(f"[fewshot] manifest_hash: {manifest['manifest_hash']}")

    if not args.skip_zeroshot:
        source_ratios = parse_ratio_csv(args.source_ratios, DEFAULT_ZEROSHOT_SOURCE_RATIOS)
        half_train_ratios = parse_ratio_csv(args.zeroshot_half_train_ratios, DEFAULT_ZEROSHOT_HALF_TRAIN_RATIOS)
        manifest, path = ensure_strict_zeroshot_manifest(
            dataset=args.dataset,
            horizon=args.horizon,
            history_len=args.history_len,
            neighbor_k=args.neighbor_k,
            window_stride=args.window_stride,
            seed=args.seed,
            normalization_source=args.normalization_source,
            max_train_items=args.max_train_items,
            source_ratios=source_ratios,
            half_train_ratios=half_train_ratios,
            test_window_divisor=args.zeroshot_test_window_divisor,
            force_rebuild=args.force,
        )
        print(f"[zeroshot] manifest: {path}")
        print(f"[zeroshot] manifest_hash: {manifest['manifest_hash']}")


if __name__ == "__main__":
    main()
