#!/usr/bin/env python3
"""Inspect suspicious records in the 2026 slice of GS CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COL = "Date"


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def infer_regular_datetime(df: pd.DataFrame, date_col: str = DATE_COL) -> tuple[pd.Series, str]:
    raw = pd.to_datetime(df[date_col], errors="coerce")
    valid = raw.dropna()
    if len(valid) < 3:
        return raw, "not enough valid timestamps to infer missing Date values"

    diffs = valid.sort_values().diff().dropna()
    median_step = diffs.median()
    if pd.isna(median_step) or median_step <= pd.Timedelta(0):
        return raw, "timestamp step could not be inferred"

    first_pos = int(raw.first_valid_index())
    start = raw.iloc[first_pos] - median_step * first_pos
    expected = pd.Series(pd.date_range(start=start, periods=len(df), freq=median_step), index=df.index)
    matches = (raw.dropna() == expected.loc[raw.notna()]).mean()

    if matches >= 0.999:
        inferred = raw.copy()
        inferred.loc[raw.isna()] = expected.loc[raw.isna()]
        return inferred, f"inferred missing Date values from regular {median_step} interval; known matches={matches:.3%}"

    return raw, f"kept raw Date values; regular-grid match rate={matches:.3%}"


def robust_zscore(values: pd.Series, center: float, scale: float) -> pd.Series:
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.nan, index=values.index)
    return 0.6745 * (values - center) / scale


def robust_stats(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna()
    if clean.empty:
        return np.nan, np.nan
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    return median, mad


def format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.3f}"


def add_markdown_table(lines: list[str], df: pd.DataFrame, max_rows: int = 30) -> None:
    if df.empty:
        lines.append("_none_")
        return
    out = df.head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]) or pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].map(format_float)
    out = out.astype(str).replace({"NaT": "", "nan": "", "NaN": "", "None": ""})
    headers = [str(c) for c in out.columns]
    rows = out.values.tolist()
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        safe_row = [str(v).replace("\n", " ").replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(safe_row) + " |")
    if len(df) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(df)} rows.")


def contiguous_ranges(indexes: list[int]) -> list[tuple[int, int]]:
    if not indexes:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = indexes[0]
    for idx in indexes[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/GS(1).csv", help="Path to GS CSV file.")
    parser.add_argument("--year", type=int, default=2026, help="Year to inspect.")
    parser.add_argument("--output-dir", default="outputs/gs_2026_quality", help="Directory for report files.")
    parser.add_argument("--z-threshold", type=float, default=8.0, help="Robust z-score threshold for value anomalies.")
    parser.add_argument("--jump-z-threshold", type=float, default=10.0, help="Robust z-score threshold for first-difference jumps.")
    parser.add_argument("--top-k", type=int, default=25, help="Rows per column in report tables.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(csv_path)
    if DATE_COL not in df.columns:
        raise SystemExit(f"missing required column: {DATE_COL}")

    inferred_date, date_note = infer_regular_datetime(df)
    df = df.copy()
    df["_date"] = inferred_date
    df["_raw_date_missing"] = pd.to_datetime(df[DATE_COL], errors="coerce").isna()

    numeric_cols = [c for c in df.columns if c not in {DATE_COL, "_date", "_raw_date_missing"} and pd.api.types.is_numeric_dtype(df[c])]
    target = df[df["_date"].dt.year == args.year].copy()
    baseline = df[df["_date"].dt.year < args.year].copy()

    if target.empty:
        raise SystemExit(f"no rows found for year {args.year}")

    anomalies: list[pd.DataFrame] = []
    for col in numeric_cols:
        base_median, base_mad = robust_stats(baseline[col])
        target_z = robust_zscore(target[col], base_median, base_mad)
        value_mask = target_z.abs() >= args.z_threshold
        if value_mask.any():
            part = target.loc[value_mask, ["_date", DATE_COL, col]].copy()
            part["column"] = col
            part["check"] = "value_vs_2024_2025_baseline"
            part["value"] = part[col]
            part["robust_z"] = target_z.loc[value_mask]
            part = part.drop(columns=[col])
            anomalies.append(part)

        full_diff = df[col].diff()
        diff_median, diff_mad = robust_stats(full_diff.loc[baseline.index])
        jump_z = robust_zscore(full_diff.loc[target.index], diff_median, diff_mad)
        jump_mask = jump_z.abs() >= args.jump_z_threshold
        if jump_mask.any():
            part = target.loc[jump_mask, ["_date", DATE_COL, col]].copy()
            part["column"] = col
            part["check"] = "jump_vs_2024_2025_baseline"
            part["value"] = part[col]
            part["previous_value"] = df[col].shift(1).loc[part.index].to_numpy()
            part["delta"] = full_diff.loc[part.index].to_numpy()
            part["robust_z"] = jump_z.loc[jump_mask]
            part = part.drop(columns=[col])
            anomalies.append(part)

    if anomalies:
        anomaly_df = pd.concat(anomalies, ignore_index=False)
        anomaly_df = anomaly_df.reset_index(names="row_index")
        anomaly_df = anomaly_df.sort_values(["check", "column", "robust_z"], key=lambda s: s.abs() if s.name == "robust_z" else s)
    else:
        anomaly_df = pd.DataFrame(columns=["row_index", "_date", DATE_COL, "column", "check", "value", "robust_z"])

    missing_rows = []
    for col in [DATE_COL] + numeric_cols:
        if col == DATE_COL:
            mask = target["_raw_date_missing"]
        else:
            mask = target[col].isna()
        idxs = target.index[mask].to_list()
        for start, end in contiguous_ranges(idxs):
            missing_rows.append(
                {
                    "column": col,
                    "count": end - start + 1,
                    "row_start": start,
                    "row_end": end,
                    "date_start": df.loc[start, "_date"],
                    "date_end": df.loc[end, "_date"],
                }
            )
    missing_df = pd.DataFrame(missing_rows)

    # Daily summaries help expose entire abnormal days, not just single points.
    # Expected counts are derived from the actual inspected time range, so a
    # partial boundary day such as 2026-04-29 00:00 is not treated as 95 missing
    # records.
    full_expected = pd.Series(1, index=pd.DatetimeIndex(target["_date"])).resample("D").sum()
    daily = target.set_index("_date")[numeric_cols].resample("D").agg(["count", "mean", "min", "max"])
    daily_flags = []
    for col in numeric_cols:
        missing_count = full_expected - daily[(col, "count")]
        bad_days = missing_count[missing_count > 0]
        for date, count in bad_days.items():
            daily_flags.append({"date": date.date().isoformat(), "column": col, "issue": "missing_points", "count": int(count)})
    daily_flags_df = pd.DataFrame(daily_flags)

    anomaly_csv = output_dir / f"gs_{args.year}_anomaly_rows.csv"
    missing_csv = output_dir / f"gs_{args.year}_missing_ranges.csv"
    report_md = output_dir / f"gs_{args.year}_quality_report.md"
    anomaly_df.to_csv(anomaly_csv, index=False)
    missing_df.to_csv(missing_csv, index=False)

    lines: list[str] = []
    lines.append(f"# GS {args.year} Data Quality Report")
    lines.append("")
    lines.append(f"- Source CSV: `{csv_path}`")
    lines.append(f"- Rows in source file: `{len(df)}`")
    lines.append(f"- Rows in {args.year}: `{len(target)}`")
    lines.append(f"- Date range in {args.year}: `{target['_date'].min()}` -> `{target['_date'].max()}`")
    lines.append(f"- Date handling: {date_note}")
    lines.append(f"- Numeric columns: `{', '.join(numeric_cols)}`")
    lines.append(f"- Value anomaly threshold: robust z-score >= `{args.z_threshold}`")
    lines.append(f"- Jump anomaly threshold: robust z-score >= `{args.jump_z_threshold}`")
    lines.append("")

    lines.append("## Missing Value Ranges")
    add_markdown_table(lines, missing_df, max_rows=args.top_k)
    lines.append("")

    lines.append("## Daily Missing Flags")
    add_markdown_table(lines, daily_flags_df, max_rows=args.top_k)
    lines.append("")

    lines.append("## Top Value Anomalies Compared With 2024-2025")
    value_anoms = anomaly_df[anomaly_df["check"] == "value_vs_2024_2025_baseline"].copy()
    if not value_anoms.empty:
        value_anoms["_abs_z"] = value_anoms["robust_z"].abs()
        value_anoms = value_anoms.sort_values("_abs_z", ascending=False).drop(columns=["_abs_z"])
    add_markdown_table(lines, value_anoms[["row_index", "_date", DATE_COL, "column", "value", "robust_z"]], max_rows=args.top_k)
    lines.append("")

    lines.append("## Top Jump Anomalies Compared With 2024-2025")
    jump_anoms = anomaly_df[anomaly_df["check"] == "jump_vs_2024_2025_baseline"].copy()
    if not jump_anoms.empty:
        jump_anoms["_abs_z"] = jump_anoms["robust_z"].abs()
        jump_anoms = jump_anoms.sort_values("_abs_z", ascending=False).drop(columns=["_abs_z"])
    jump_cols = ["row_index", "_date", DATE_COL, "column", "previous_value", "value", "delta", "robust_z"]
    add_markdown_table(lines, jump_anoms[jump_cols] if not jump_anoms.empty else jump_anoms, max_rows=args.top_k)
    lines.append("")

    lines.append("## 2026 Numeric Summary")
    summary = target[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    summary = summary[["count", "mean", "std", "min", "1%", "5%", "50%", "95%", "99%", "max"]]
    add_markdown_table(lines, summary.reset_index(names="column"), max_rows=len(summary))
    lines.append("")

    lines.append("## Output Files")
    lines.append(f"- Anomaly rows: `{anomaly_csv}`")
    lines.append(f"- Missing ranges: `{missing_csv}`")

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"report: {report_md}")
    print(f"anomaly rows: {anomaly_csv} ({len(anomaly_df)} rows)")
    print(f"missing ranges: {missing_csv} ({len(missing_df)} ranges)")


if __name__ == "__main__":
    main()
