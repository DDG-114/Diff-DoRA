#!/usr/bin/env python
"""Prepare the unpacked Wotai data into a Diff-DoRA-compatible matrix layout."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_wotai_evcdp import load_wotai_evcdp


FREQ = "15min"
SOURCE_DIR = Path("data/raw/wotai_source")
OUTPUT_DIR = Path("data/raw/wotai_evcdp")

WEATHER_RENAME = {
    "temp": "temperature",
    "rh": "humidity",
    "dewpt": "dew_point",
    "wind_spd": "wind_speed",
}

WEATHER_COLUMNS = [
    "weather_code",
    "temperature",
    "pressure",
    "wind_dir",
    "wind_speed",
    "humidity",
    "dew_point",
    "clouds",
    "dhi",
    "dni",
    "ghi",
    "solar_rad",
    "elev_angle",
    "azimuth",
    "uv",
]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    last_exc: Exception | None = None
    for encoding in (None, "utf-8-sig", "gb18030", "gbk"):
        try:
            read_kwargs = dict(kwargs)
            read_kwargs.setdefault("low_memory", False)
            if encoding is not None:
                read_kwargs["encoding"] = encoding
            return pd.read_csv(path, **read_kwargs)
        except UnicodeDecodeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to read CSV: {path}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _to_datetime(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _numeric_time_frame(path: Path, time_col: str) -> pd.DataFrame:
    df = _normalise_columns(_read_csv(path))
    if time_col not in df.columns:
        raise ValueError(f"{path} is missing time column {time_col!r}")
    ts = _to_datetime(df[time_col])
    df = df.loc[ts.notna()].copy()
    df.index = pd.DatetimeIndex(ts.loc[ts.notna()])
    df = df.drop(columns=[time_col])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.groupby(df.index).mean().sort_index()
    return df


def _find_raw_csv_by_columns(source_dir: Path, required: set[str]) -> Path | None:
    raw_dir = source_dir / "Raw Data"
    if not raw_dir.exists():
        return None
    for path in sorted(raw_dir.glob("*.csv")):
        try:
            cols = set(_normalise_columns(_read_csv(path, nrows=0)).columns)
        except Exception:
            continue
        if required.issubset(cols):
            return path
    return None


def _align_series(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    clip_lower: float | None = None,
) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    series = series.groupby(series.index).mean().sort_index()
    series = series.reindex(index)
    series = series.interpolate(method="time", limit_direction="both").ffill().bfill()
    if clip_lower is not None:
        series = series.clip(lower=clip_lower)
    return series.astype(float)


def _load_primary(source_dir: Path) -> pd.DataFrame:
    candidates = [
        source_dir / "Load Prediction" / "Preprocessed_Load_Prediction.csv",
        source_dir / "Load Prediction" / "Load_Prediction.csv",
    ]
    for path in candidates:
        if path.exists():
            return _numeric_time_frame(path, "time")
    raise FileNotFoundError("Could not find Load Prediction CSV under the source directory.")


def _load_pv_total(source_dir: Path) -> pd.DataFrame | None:
    candidates = [
        source_dir / "PV Prediction" / "Preprocessed_PV_Prediction.csv",
        source_dir / "PV Prediction" / "PV_Prediction.csv",
    ]
    for path in candidates:
        if path.exists():
            return _numeric_time_frame(path, "time")
    return None


def _load_resampled_raw_power(source_dir: Path, required: set[str], time_col: str, value_col: str) -> pd.Series | None:
    path = _find_raw_csv_by_columns(source_dir, required)
    if path is None:
        return None
    frame = _numeric_time_frame(path, time_col)
    if value_col not in frame.columns:
        return None
    return frame[value_col].resample(FREQ).mean()


def _build_matrix(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    primary = _load_primary(source_dir)
    index = pd.date_range(primary.index.min(), primary.index.max(), freq=FREQ)

    nodes: dict[str, pd.Series] = {}
    source_columns: dict[str, str] = {}

    nodes["actual_load"] = _align_series(primary["actual_load"], index, clip_lower=0.0)
    source_columns["actual_load"] = "Load Prediction.actual_load"

    pv_total = _load_pv_total(source_dir)
    if pv_total is not None and "p_pv_total" in pv_total.columns:
        nodes["pv_total_power"] = _align_series(pv_total["p_pv_total"], index, clip_lower=0.0)
        source_columns["pv_total_power"] = "PV Prediction.p_pv_total"
    elif "p_pv_total" in primary.columns:
        nodes["pv_total_power"] = _align_series(primary["p_pv_total"], index, clip_lower=0.0)
        source_columns["pv_total_power"] = "Load Prediction.p_pv_total"

    storage = _load_resampled_raw_power(
        source_dir,
        required={"time", "energy_p", "energy_n", "pac", "scada_sn", "ems_sn"},
        time_col="time",
        value_col="pac",
    )
    if storage is not None:
        nodes["storage_ac_power"] = _align_series(storage, index)
        source_columns["storage_ac_power"] = "Raw Data.storage.pac"
    elif "pac" in primary.columns:
        nodes["storage_ac_power"] = _align_series(primary["pac"], index)
        source_columns["storage_ac_power"] = "Load Prediction.pac"

    gate = _load_resampled_raw_power(
        source_dir,
        required={"ts", "e_total_active_charge", "e_total_active_discharge", "p_total_active"},
        time_col="ts",
        value_col="p_total_active",
    )
    if gate is not None:
        nodes["grid_active_power"] = _align_series(gate, index)
        source_columns["grid_active_power"] = "Raw Data.gate.p_total_active"
    elif "p_total_active" in primary.columns:
        nodes["grid_active_power"] = _align_series(primary["p_total_active"], index)
        source_columns["grid_active_power"] = "Load Prediction.p_total_active"

    matrix = pd.DataFrame(nodes, index=index)
    matrix.index.name = "timestamp"
    if matrix.shape[1] < 2:
        raise ValueError(f"Need at least two aligned nodes for MoE routing; got {matrix.shape[1]}")

    weather = primary.rename(columns=WEATHER_RENAME)
    weather = weather.reindex(index).interpolate(method="time", limit_direction="both").ffill().bfill()
    keep_weather = [col for col in WEATHER_COLUMNS if col in weather.columns]
    weather = weather[keep_weather].apply(pd.to_numeric, errors="coerce")
    weather.index.name = "timestamp"

    manifest = {
        "source_columns": source_columns,
        "source_dir": str(source_dir),
        "frequency": FREQ,
        "start": str(index.min()),
        "end": str(index.max()),
        "rows": int(len(index)),
        "nodes": list(matrix.columns),
        "price": "not generated; no electricity-price field was found in the source files",
    }
    return matrix, weather, manifest


def _build_node_meta(matrix: pd.DataFrame, source_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for node_id in matrix.columns:
        raw = matrix[node_id]
        if node_id in {"actual_load", "grid_active_power"}:
            zone_type = "CBD"
            role = "demand"
        else:
            zone_type = "Residential"
            role = "auxiliary_power"
        rows.append(
            {
                "node_id": node_id,
                "zone_type": zone_type,
                "type": role,
                "capacity": float(np.nanmax(np.abs(raw.to_numpy(dtype=float)))),
                "area": "Wotai",
                "unit": "kW",
                "source_column": source_columns.get(node_id, ""),
            }
        )
    return pd.DataFrame(rows)


def _build_adjacency(matrix: pd.DataFrame) -> pd.DataFrame:
    train_end = max(2, int(len(matrix) * 0.6))
    corr = matrix.iloc[:train_end].corr().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 0.0)
    corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
    corr = corr.reindex(index=matrix.columns, columns=matrix.columns).fillna(0.0)
    corr.insert(0, "node_id", corr.index)
    return corr.reset_index(drop=True)


def _build_time_csv(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": index.year,
            "month": index.month,
            "day": index.day,
            "hour": index.hour,
            "minute": index.minute,
            "second": index.second,
        }
    )


def prepare(source_dir: Path, output_dir: Path, *, force_reprocess: bool = True) -> dict:
    matrix, weather, manifest = _build_matrix(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    occupancy_path = output_dir / "occupancy.csv"
    weather_path = output_dir / "weather.csv"
    nodes_path = output_dir / "nodes.csv"
    adjacency_path = output_dir / "adjacency.csv"
    time_path = output_dir / "time.csv"
    manifest_path = output_dir / "manifest.json"

    matrix.reset_index().to_csv(occupancy_path, index=False)
    weather.reset_index().to_csv(weather_path, index=False)
    node_meta = _build_node_meta(matrix, manifest["source_columns"])
    node_meta.to_csv(nodes_path, index=False)
    _build_adjacency(matrix).to_csv(adjacency_path, index=False)
    _build_time_csv(matrix.index).to_csv(time_path, index=False)

    manifest.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "files": {
                "occupancy": str(occupancy_path),
                "weather": str(weather_path),
                "nodes": str(nodes_path),
                "adjacency": str(adjacency_path),
                "time": str(time_path),
            },
        }
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    processed = load_wotai_evcdp(
        raw_dir=output_dir,
        processed_path=Path("data/processed/wotai_evcdp.pkl"),
        force_reprocess=force_reprocess,
    )
    return {
        "manifest": manifest,
        "processed_shape": tuple(processed["occupancy"].shape),
        "norm_min": processed["norm_min"],
        "norm_max": processed["norm_max"],
        "weather_shape": tuple(processed["weather"].shape),
        "adj_shape": tuple(processed["adj"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=str(SOURCE_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no_force_reprocess", action="store_true")
    args = parser.parse_args()

    summary = prepare(
        Path(args.source_dir),
        Path(args.output_dir),
        force_reprocess=not args.no_force_reprocess,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
