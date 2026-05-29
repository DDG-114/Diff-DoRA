#!/usr/bin/env python3
"""Prepare GS(1).csv for electricity price forecasting.

Target:
  Price

Auxiliary features:
  发电总出力预测, 竞价空间, 统一负荷预测, 抽蓄, 统一新能源预测, 联络线计划

Missing values are repaired by time interpolation. Outliers are kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").exists() and (parent / "data").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_gs_price import load_gs_price

FREQ = "15min"
DEFAULT_SOURCE = Path("data/GS(1).csv")
DEFAULT_OUTPUT = Path("data/raw/gs_price")
DEFAULT_PROCESSED = Path("data/processed/gs_price.pkl")
DATE_COL = "Date"
TARGET_COL = "Price"
AUX_COLUMNS = [
    "发电总出力预测",
    "竞价空间",
    "统一负荷预测",
    "抽蓄",
    "统一新能源预测",
    "联络线计划",
]
ALL_VALUE_COLUMNS = [TARGET_COL, *AUX_COLUMNS]


def _read_csv(path: Path) -> pd.DataFrame:
    last_exc: Exception | None = None
    for encoding in (None, "utf-8-sig", "gb18030", "gbk"):
        try:
            kwargs = {"low_memory": False}
            if encoding is not None:
                kwargs["encoding"] = encoding
            return pd.read_csv(path, **kwargs)
        except UnicodeDecodeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to read CSV: {path}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _regular_datetime(raw_dates: pd.Series) -> tuple[pd.Series, str]:
    parsed = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    valid = parsed.dropna()
    if len(valid) < 3:
        return parsed, "not enough valid timestamps to infer a regular grid"

    diffs = valid.sort_values().diff().dropna()
    median_step = diffs.median()
    first_pos = int(parsed.first_valid_index())
    start = parsed.iloc[first_pos] - median_step * first_pos
    expected = pd.Series(pd.date_range(start=start, periods=len(parsed), freq=median_step), index=parsed.index)
    match_rate = (parsed.dropna() == expected.loc[parsed.notna()]).mean()
    if match_rate >= 0.999:
        out = parsed.copy()
        out.loc[parsed.isna()] = expected.loc[parsed.isna()]
        return out, f"inferred missing timestamps on a regular {median_step} grid; match_rate={match_rate:.3%}"
    return parsed, f"kept parsed timestamps; regular-grid match_rate={match_rate:.3%}"


def _repair_values(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_missing = values.isna()
    repaired = values.replace([np.inf, -np.inf], np.nan)
    repaired = repaired.interpolate(method="time", limit_direction="both").ffill().bfill()
    remaining_missing = repaired.isna()
    if remaining_missing.any().any():
        repaired = repaired.fillna(0.0)
    return repaired.astype(float), raw_missing.astype(int), remaining_missing.astype(int)


def _split_indices(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    train = np.flatnonzero(index.year == 2025)
    test = np.flatnonzero(index.year == 2026)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Expected both 2025 train rows and 2026 test rows.")
    return {
        "train": train.astype(int),
        "val": np.asarray([], dtype=int),
        "test": test.astype(int),
    }


def _split_indices_2025_internal(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    year_mask = index.year == 2025
    year_positions = np.flatnonzero(year_mask)
    if len(year_positions) == 0:
        raise ValueError("No 2025 rows found.")
    n = len(year_positions)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return {
        "train": year_positions[:train_end].astype(int),
        "val": year_positions[train_end:val_end].astype(int),
        "test": year_positions[val_end:].astype(int),
    }


def _write_node_meta(price: pd.DataFrame, output_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "node_id": TARGET_COL,
                "english_name": "market_price",
                "zone_type": "Market",
                "type": "price",
                "expert_id": 0,
                "capacity": float(np.nanmax(np.abs(price[TARGET_COL].to_numpy(dtype=float)))),
                "area": "GS",
                "unit": "CNY/MWh",
                "source_column": TARGET_COL,
                "description": "electricity market clearing price",
            }
        ]
    ).to_csv(output_path, index=False)


def _write_adjacency(output_path: Path) -> None:
    pd.DataFrame({"node_id": [TARGET_COL], TARGET_COL: [0.0]}).to_csv(output_path, index=False)


def _write_time_csv(index: pd.DatetimeIndex, output_path: Path) -> None:
    pd.DataFrame(
        {
            "timestamp": index,
            "year": index.year,
            "month": index.month,
            "day": index.day,
            "hour": index.hour,
            "minute": index.minute,
            "second": index.second,
        }
    ).to_csv(output_path, index=False)


def prepare(
    source_csv: Path,
    output_dir: Path,
    processed_path: Path,
    *,
    force_reprocess: bool = True,
    split_mode: str = "2025_to_2026",
) -> dict:
    df = _normalise_columns(_read_csv(source_csv))
    missing_columns = [col for col in [DATE_COL, *ALL_VALUE_COLUMNS] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    timestamps, date_note = _regular_datetime(df[DATE_COL])
    values = df[ALL_VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values.index = pd.DatetimeIndex(timestamps)
    values.index.name = "timestamp"
    values = values.sort_index()

    repaired, raw_missing, remaining_missing = _repair_values(values)
    if split_mode == "2025_internal":
        split_indices = _split_indices_2025_internal(pd.DatetimeIndex(repaired.index))
        split_policy = "train=first 80% of 2025, val=next 10% of 2025, test=last 10% of 2025"
    elif split_mode == "2025_to_2026":
        split_indices = _split_indices(pd.DatetimeIndex(repaired.index))
        split_policy = "train=calendar year 2025, test=calendar year 2026, val empty; trainer splits expert train pool internally"
    else:
        raise ValueError(f"Unsupported split_mode={split_mode!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    occupancy_path = output_dir / "occupancy.csv"
    weather_path = output_dir / "weather.csv"
    price_path = output_dir / "price.csv"
    missing_mask_path = output_dir / "missing_mask.csv"
    remaining_missing_path = output_dir / "remaining_missing_mask.csv"
    nodes_path = output_dir / "nodes.csv"
    adjacency_path = output_dir / "adjacency.csv"
    time_path = output_dir / "time.csv"
    split_path = output_dir / "split_indices.json"
    manifest_path = output_dir / "manifest.json"

    price = repaired[[TARGET_COL]]
    aux = repaired[AUX_COLUMNS].rename(
        columns={
            "发电总出力预测": "total_generation_forecast",
            "竞价空间": "bidding_space",
            "统一负荷预测": "unified_load_forecast",
            "抽蓄": "pumped_storage",
            "统一新能源预测": "unified_renewable_forecast",
            "联络线计划": "tie_line_schedule",
        }
    )

    price.reset_index().to_csv(occupancy_path, index=False)
    price.reset_index().to_csv(price_path, index=False)
    aux.reset_index().to_csv(weather_path, index=False)
    raw_missing.reset_index().to_csv(missing_mask_path, index=False)
    remaining_missing.reset_index().to_csv(remaining_missing_path, index=False)
    _write_node_meta(price, nodes_path)
    _write_adjacency(adjacency_path)
    _write_time_csv(pd.DatetimeIndex(repaired.index), time_path)

    split_json = {key: value.astype(int).tolist() for key, value in split_indices.items()}
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=2, ensure_ascii=False)

    missing_summary = {
        col: int(raw_missing[col].sum())
        for col in raw_missing.columns
        if int(raw_missing[col].sum()) > 0
    }
    manifest = {
        "dataset": "gs_price",
        "source_csv": str(source_csv),
        "frequency": FREQ,
        "target": TARGET_COL,
        "auxiliary_features": list(aux.columns),
        "start": str(repaired.index.min()),
        "end": str(repaired.index.max()),
        "rows": int(len(repaired)),
        "date_repair": date_note,
        "missing_policy": "time interpolation with limit_direction='both', followed by ffill/bfill; no outlier clipping",
        "missing_summary": missing_summary,
        "split_mode": split_mode,
        "split_policy": split_policy,
        "split_sizes": {key: int(len(value)) for key, value in split_indices.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "occupancy": str(occupancy_path),
            "weather": str(weather_path),
            "price": str(price_path),
            "missing_mask": str(missing_mask_path),
            "remaining_missing_mask": str(remaining_missing_path),
            "nodes": str(nodes_path),
            "adjacency": str(adjacency_path),
            "time": str(time_path),
            "split_indices": str(split_path),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    processed = load_gs_price(
        raw_dir=output_dir,
        processed_path=processed_path,
        force_reprocess=force_reprocess,
    )
    return {
        "manifest": manifest,
        "processed_path": str(processed_path),
        "processed_shape": tuple(processed["occupancy"].shape),
        "raw_range": [processed["norm_min"], processed["norm_max"]],
        "split_sizes": {key: int(len(value)) for key, value in processed["split_indices"].items()},
        "node_ids": processed["node_ids"],
        "weather_shape": tuple(processed["weather"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--processed_path", default=str(DEFAULT_PROCESSED))
    parser.add_argument("--split_mode", choices=["2025_to_2026", "2025_internal"], default="2025_to_2026")
    parser.add_argument("--no_force_reprocess", action="store_true")
    args = parser.parse_args()

    summary = prepare(
        Path(args.source_csv),
        Path(args.output_dir),
        Path(args.processed_path),
        force_reprocess=not args.no_force_reprocess,
        split_mode=args.split_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
