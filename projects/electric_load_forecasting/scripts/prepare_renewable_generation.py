#!/usr/bin/env python
"""Prepare renewable generation Excel datasets into Diff-DoRA-compatible layouts."""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_renewable_generation import load_renewable_solar, load_renewable_wind

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
SOURCE_ROOT = Path("data/raw/renewable_generation/source")
OUTPUT_ROOTS = {
    "solar": Path("data/raw/renewable_solar"),
    "wind": Path("data/raw/renewable_wind"),
}
PROCESSED_LOADERS = {
    "solar": load_renewable_solar,
    "wind": load_renewable_wind,
}

SOLAR_FEATURE_ORDER = [
    "total_solar_irradiance",
    "direct_normal_irradiance",
    "global_horizontal_irradiance",
    "temperature",
    "pressure",
    "humidity",
]
WIND_FEATURE_ORDER = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_30m",
    "wind_direction_30m",
    "wind_speed_50m",
    "wind_direction_50m",
    "wind_speed_hub",
    "wind_direction_hub",
    "temperature",
    "pressure",
    "humidity",
]


def _xlsx_col_to_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    result = 0
    for ch in letters:
        result = result * 26 + ord(ch.upper()) - ord("A") + 1
    return max(0, result - 1)


def _shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall(f"{NS}si"):
        strings.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    return strings


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{NS}t"))
    v = cell.find(f"{NS}v")
    raw = "" if v is None else v.text or ""
    if cell_type == "s" and raw != "":
        try:
            return shared[int(raw)]
        except Exception:
            return raw
    return raw


def _read_xlsx_table(path: Path) -> pd.DataFrame:
    with ZipFile(path) as zf:
        shared = _shared_strings(zf)
        rows: list[dict[int, str]] = []
        max_col = 0
        with zf.open("xl/worksheets/sheet1.xml") as fh:
            for _event, elem in ET.iterparse(fh, events=("end",)):
                if elem.tag != f"{NS}row":
                    continue
                row_dict: dict[int, str] = {}
                for cell in elem.findall(f"{NS}c"):
                    col_idx = _xlsx_col_to_index(cell.attrib.get("r", "A1"))
                    row_dict[col_idx] = _cell_text(cell, shared)
                    max_col = max(max_col, col_idx + 1)
                rows.append(row_dict)
                elem.clear()

    matrix = []
    for row_dict in rows:
        row = [""] * max_col
        for idx, value in row_dict.items():
            row[idx] = value
        matrix.append(row)
    df = pd.DataFrame(matrix)
    header = [str(col).strip() for col in df.iloc[0].tolist()]
    data = df.iloc[1:].reset_index(drop=True).copy()
    data.columns = header
    data = data.loc[:, [str(col).strip() != "" for col in data.columns]]
    return data


def _normalise_header(header: str) -> str:
    header = str(header).replace("\n", " ").strip()
    lowered = header.lower()
    lowered = lowered.replace("horicontal", "horizontal")
    lowered = re.sub(r"\s+", " ", lowered)

    if "time(year-month-day" in lowered:
        return "timestamp"
    if "total solar irradiance" in lowered:
        return "total_solar_irradiance"
    if "direct normal irradiance" in lowered:
        return "direct_normal_irradiance"
    if "global horizontal irradiance" in lowered:
        return "global_horizontal_irradiance"
    if "wind speed at height of 10 meters" in lowered:
        return "wind_speed_10m"
    if "wind direction at height of 10 meters" in lowered:
        return "wind_direction_10m"
    if "wind speed at height of 30 meters" in lowered:
        return "wind_speed_30m"
    if "wind direction at height of 30 meters" in lowered:
        return "wind_direction_30m"
    if "wind speed at height of 50 meters" in lowered:
        return "wind_speed_50m"
    if "wind direction at height of 50 meters" in lowered:
        return "wind_direction_50m"
    if "wheel hub" in lowered and "m/s" in lowered:
        return "wind_speed_hub"
    if "wheel hub" in lowered and ("˚" in header or "direction" in lowered):
        return "wind_direction_hub"
    if lowered.startswith("air temperature"):
        return "temperature"
    if "atmosphere" in lowered:
        return "pressure"
    if "relative humidity" in lowered:
        return "humidity"
    if lowered.startswith("power"):
        return "power_mw"
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed_text = pd.to_datetime(raw, errors="coerce", format="mixed")
    numeric = pd.to_numeric(raw, errors="coerce")
    parsed_numeric = pd.to_datetime(
        numeric,
        errors="coerce",
        unit="D",
        origin="1899-12-30",
    )
    parsed = parsed_text.where(parsed_text.notna(), parsed_numeric)
    return parsed.dt.round("s")


def _read_station_frame(path: Path) -> pd.DataFrame:
    df = _read_xlsx_table(path)
    rename_map = {col: _normalise_header(col) for col in df.columns}
    df = df.rename(columns=rename_map)
    if "timestamp" not in df.columns or "power_mw" not in df.columns:
        raise ValueError(f"{path} is missing required timestamp/power columns after normalization.")
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.loc[df["timestamp"].notna()].copy()
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("timestamp")
    return df


def _nominal_capacity(path: Path) -> float:
    match = re.search(r"Nominal capacity-(\d+)MW", path.name)
    if not match:
        return float("nan")
    return float(match.group(1))


def _site_name(kind: str, idx: int) -> str:
    return f"{kind}_site_{idx:02d}"


def _family_feature_order(dataset: str) -> list[str]:
    return SOLAR_FEATURE_ORDER if dataset == "solar" else WIND_FEATURE_ORDER


def _collect_station_frames(dataset: str, variant: str) -> dict[str, dict]:
    folder = SOURCE_ROOT / f"data_{variant}" / ("solar_stations" if dataset == "solar" else "wind_farms")
    if not folder.exists():
        raise FileNotFoundError(f"Source folder not found: {folder}")

    stations = {}
    for idx, path in enumerate(sorted(folder.glob("*.xlsx")), start=1):
        site_id = _site_name(dataset, idx)
        df = _read_station_frame(path)
        stations[site_id] = {
            "path": path,
            "frame": df,
            "capacity_mw": _nominal_capacity(path),
            "raw_rows": int(len(df)),
        }
    if not stations:
        raise ValueError(f"No Excel files found in {folder}")
    return stations


def _infer_freq(stations: dict[str, dict]) -> pd.Timedelta:
    for station in stations.values():
        idx = station["frame"].index
        diffs = idx.to_series().diff().dropna()
        if not diffs.empty:
            mode = diffs.mode()
            return mode.iloc[0] if not mode.empty else diffs.iloc[0]
    return pd.Timedelta(minutes=15)


def _keep_station_ids(stations: dict[str, dict], min_length_ratio: float) -> list[str]:
    max_rows = max(meta["raw_rows"] for meta in stations.values())
    threshold = max(1, int(max_rows * float(min_length_ratio)))
    keep = [site_id for site_id, meta in stations.items() if meta["raw_rows"] >= threshold]
    if not keep:
        raise ValueError("No stations remained after min_length_ratio filtering.")
    return keep


def _align_station_series(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    clip_lower: float | None = None,
) -> pd.Series:
    aligned = series.reindex(index)
    aligned = aligned.interpolate(method="time", limit_direction="both").ffill().bfill()
    if clip_lower is not None:
        aligned = aligned.clip(lower=clip_lower)
    return aligned.astype(float)


def _build_dataset(dataset: str, variant: str, min_length_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    stations = _collect_station_frames(dataset, variant)
    keep_ids = _keep_station_ids(stations, min_length_ratio=min_length_ratio)
    kept = {site_id: stations[site_id] for site_id in keep_ids}

    freq = _infer_freq(kept)
    common_start = max(meta["frame"].index.min() for meta in kept.values())
    common_end = min(meta["frame"].index.max() for meta in kept.values())
    index = pd.date_range(common_start, common_end, freq=freq)
    if len(index) < 100:
        raise ValueError(f"Common overlap too short for {dataset}: {len(index)} rows")

    occupancy = {}
    weather_features: dict[str, list[pd.Series]] = {feature: [] for feature in _family_feature_order(dataset)}
    node_rows = []
    coverage = {}
    for site_id, meta in kept.items():
        frame = meta["frame"]
        power = _align_station_series(frame["power_mw"], index, clip_lower=0.0)
        occupancy[site_id] = power
        coverage[site_id] = float(frame["power_mw"].notna().sum() / max(len(index), 1))
        for feature in weather_features:
            if feature in frame.columns:
                weather_features[feature].append(_align_station_series(frame[feature], index))
        node_rows.append(
            {
                "node_id": site_id,
                "capacity": meta["capacity_mw"],
                "site_type": dataset,
                "raw_file": meta["path"].name,
                "source_variant": variant,
                "coverage_ratio": round(coverage[site_id], 4),
            }
        )

    occupancy_df = pd.DataFrame(occupancy, index=index)
    occupancy_df.index.name = "timestamp"

    weather_df = pd.DataFrame(index=index)
    for feature, series_list in weather_features.items():
        if series_list:
            weather_df[feature] = pd.concat(series_list, axis=1).mean(axis=1)
    weather_df.index.name = "timestamp"

    node_meta = pd.DataFrame(node_rows)
    train_end = max(2, int(len(occupancy_df) * 0.6))
    corr = occupancy_df.iloc[:train_end].corr().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 0.0)
    adj = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
    adj.insert(0, "node_id", adj.index)
    adjacency = adj.reset_index(drop=True)

    manifest = {
        "dataset": f"renewable_{dataset}",
        "source_root": str(SOURCE_ROOT),
        "source_variant": variant,
        "kept_sites": keep_ids,
        "dropped_sites": [site_id for site_id in stations if site_id not in keep_ids],
        "rows": int(len(index)),
        "frequency": str(freq),
        "start": str(index.min()),
        "end": str(index.max()),
        "features": list(weather_df.columns),
        "coverage_ratio": {site_id: coverage[site_id] for site_id in keep_ids},
    }
    return occupancy_df, weather_df, node_meta, adjacency, manifest


def _write_time_csv(index: pd.DatetimeIndex, out_path: Path) -> None:
    df = pd.DataFrame(
        {
            "year": index.year,
            "month": index.month,
            "day": index.day,
            "hour": index.hour,
            "minute": index.minute,
            "second": index.second,
        }
    )
    df.to_csv(out_path, index=False)


def prepare_one(dataset: str, *, variant: str, min_length_ratio: float, force_reprocess: bool) -> dict:
    output_dir = OUTPUT_ROOTS[dataset]
    occupancy_df, weather_df, node_meta, adjacency, manifest = _build_dataset(
        dataset,
        variant,
        min_length_ratio=min_length_ratio,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    occupancy_path = output_dir / "occupancy.csv"
    weather_path = output_dir / "weather.csv"
    nodes_path = output_dir / "nodes.csv"
    adj_path = output_dir / "adjacency.csv"
    time_path = output_dir / "time.csv"
    manifest_path = output_dir / "manifest.json"

    occupancy_df.reset_index().to_csv(occupancy_path, index=False)
    weather_df.reset_index().to_csv(weather_path, index=False)
    node_meta.to_csv(nodes_path, index=False)
    adjacency.to_csv(adj_path, index=False)
    _write_time_csv(occupancy_df.index, time_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **manifest,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "files": {
                    "occupancy": str(occupancy_path),
                    "weather": str(weather_path),
                    "nodes": str(nodes_path),
                    "adjacency": str(adj_path),
                    "time": str(time_path),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    loader = PROCESSED_LOADERS[dataset]
    processed = loader(force_reprocess=force_reprocess)
    return {
        "dataset": dataset,
        "output_dir": str(output_dir),
        "processed_shape": tuple(processed["occupancy"].shape),
        "weather_shape": tuple(processed["weather"].shape),
        "adj_shape": tuple(processed["adj"].shape),
        "kept_sites": manifest["kept_sites"],
        "dropped_sites": manifest["dropped_sites"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["solar", "wind", "all"], default="all")
    parser.add_argument("--source_variant", choices=["processed", "original"], default="processed")
    parser.add_argument("--min_length_ratio", type=float, default=0.9)
    parser.add_argument(
        "--keep_all_sites",
        action="store_true",
        help="Keep every source station by disabling the coverage-ratio filter.",
    )
    parser.add_argument("--no_force_reprocess", action="store_true")
    args = parser.parse_args()

    min_length_ratio = 0.0 if args.keep_all_sites else args.min_length_ratio

    targets = ["solar", "wind"] if args.dataset == "all" else [args.dataset]
    results = []
    for dataset in targets:
        results.append(
            prepare_one(
                dataset,
                variant=args.source_variant,
                min_length_ratio=min_length_ratio,
                force_reprocess=not args.no_force_reprocess,
            )
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
