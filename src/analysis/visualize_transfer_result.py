from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from xml.sax.saxutils import escape

import numpy as np

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.loaders import load_dataset
from src.data.windowing import resolve_window_stride


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _prediction_range(pred: np.ndarray) -> float:
    return float(np.max(pred) - np.min(pred))


def _sample_series(arr: np.ndarray, max_points: int = 28) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32).reshape(-1)
    if values.size <= max_points:
        return values
    indices = np.linspace(0, values.size - 1, num=max_points, dtype=int)
    return values[indices]


def _constant_ratio(records: list[dict], *, norm_min: float, norm_max: float) -> float:
    if not records:
        return 0.0
    flat = 0
    for rec in records:
        pred = _denormalize(np.asarray(rec["parsed_prediction"], dtype=np.float32), norm_min, norm_max)
        if float(np.max(pred) - np.min(pred)) < 1e-6:
            flat += 1
    return flat / len(records)


def _quantiles(records: list[dict], key: str) -> dict[str, float]:
    vals = sorted(float(rec[key]) for rec in records)
    return {
        "min": vals[0],
        "p25": float(np.percentile(vals, 25)),
        "median": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "max": vals[-1],
    }


def _select_cases(records: list[dict]) -> dict[str, list[dict]]:
    enriched = []
    for rec in records:
        pred = np.asarray(rec["parsed_prediction"], dtype=np.float32)
        target = np.asarray(rec["target"], dtype=np.float32)
        rec2 = dict(rec)
        rec2["case_mae"] = _mae(pred, target)
        rec2["case_rmse"] = _rmse(pred, target)
        rec2["pred_range"] = _prediction_range(pred)
        rec2["target_range"] = _prediction_range(target)
        enriched.append(rec2)
    best = sorted(enriched, key=lambda r: (r["case_mae"], r["case_rmse"]))[:20]
    dynamic_good = [
        rec
        for rec in sorted(
            enriched,
            key=lambda r: (r["case_mae"], -r["pred_range"], r["case_rmse"]),
        )
        if rec["pred_range"] > 1e-6
    ][:20]
    good_and_dynamic = [
        rec
        for rec in sorted(
            enriched,
            key=lambda r: (r["case_mae"], r["case_rmse"], -r["pred_range"]),
        )
        if rec["pred_range"] > 1e-6 and rec["target_range"] > 1e-6
    ][:20]
    representative = sorted(
        enriched,
        key=lambda r: (abs(r["case_mae"] - np.median([row["case_mae"] for row in enriched])), -r["pred_range"]),
    )[:20]
    return {
        "best": best,
        "dynamic_good": dynamic_good,
        "good_and_dynamic": good_and_dynamic,
        "representative": representative,
    }


def _svg_text(x, y, text, *, size=12, fill="#111827", anchor="start", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-weight="{weight}">{escape(str(text))}</text>'
    )


def _svg_line(x1, y1, x2, y2, *, stroke="#9ca3af", stroke_width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{dash_attr} />'
    )


def _svg_rect(x, y, w, h, *, fill="#ffffff", stroke="#d1d5db", stroke_width=1):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" rx="8" ry="8" />'
    )


def _svg_polyline(points, *, stroke, stroke_width=2.2):
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" />'
    )


def _case_tags(cases: dict[str, list[dict]]) -> dict[tuple[int, int], dict[str, int]]:
    tagged: dict[tuple[int, int], dict[str, int]] = {}
    for label, rows in cases.items():
        for rank, rec in enumerate(rows, start=1):
            key = (int(rec["sample_index"]), int(rec["t_start"]))
            tagged.setdefault(key, {})
            tagged[key][label] = rank
    return tagged


def _denormalize(arr: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    return arr * (norm_max - norm_min) + norm_min


def _extract_case_series(
    rec: dict,
    sample: dict,
    *,
    norm_min: float,
    norm_max: float,
) -> dict:
    node_idx = int(rec["node_idx"])
    history = _denormalize(np.asarray(sample["x_hist"][:, node_idx], dtype=np.float32), norm_min, norm_max)
    target = _denormalize(np.asarray(rec["target"], dtype=np.float32), norm_min, norm_max)
    prediction = _denormalize(np.asarray(rec["parsed_prediction"], dtype=np.float32), norm_min, norm_max)

    context = sample.get("x_context")
    if context is not None:
        context_arr = _denormalize(np.asarray(context[:, node_idx], dtype=np.float32), norm_min, norm_max)
        context_steps = int(context_arr.shape[0])
        long_history_anchors = _sample_series(context_arr, max_points=28)
    else:
        context_steps = int(history.shape[0])
        long_history_anchors = history

    return {
        "sample_index": int(rec["sample_index"]),
        "t_start": int(rec["t_start"]),
        "node_idx": node_idx,
        "history_steps": int(history.shape[0]),
        "context_steps": context_steps,
        "history": [float(v) for v in history],
        "long_history_anchors": [float(v) for v in np.asarray(long_history_anchors, dtype=np.float32)],
        "target": [float(v) for v in target],
        "prediction": [float(v) for v in prediction],
        "case_mae": _mae(prediction, target),
        "case_rmse": _rmse(prediction, target),
    }


def _select_curated_cases(
    records: list[dict],
    cases: dict[str, list[dict]],
    *,
    target_count: int = 20,
    explicit_sample_indices: list[int] | None = None,
) -> list[dict]:
    if explicit_sample_indices:
        wanted = list(explicit_sample_indices)
        by_idx = {int(rec["sample_index"]): rec for rec in records}
        selected = []
        for sample_index in wanted:
            rec = by_idx.get(int(sample_index))
            if rec is not None:
                selected.append(dict(rec, curated_tags=[f"selected:{sample_index}"]))
        return selected

    wanted = {
        "best": {1, 2, 3, 4},
        "dynamic_good": {1, 2, 3, 4, 5, 6},
        "good_and_dynamic": {1, 2, 3, 4, 5, 6},
        "representative": {1, 2, 3},
    }
    tag_lookup = _case_tags(cases)
    selected: dict[tuple[int, int], dict] = {}
    priority = []
    for label, ranks in (
        ("best", [1, 2, 3, 4]),
        ("dynamic_good", [1, 2, 3, 4, 5, 6]),
        ("good_and_dynamic", [1, 2, 3, 4, 5, 6]),
        ("representative", [1, 2, 3]),
    ):
        for rank in ranks:
            priority.append(f"{label}:{rank}")
    priority_order = {tag: idx for idx, tag in enumerate(priority)}

    for rec in records:
        key = (int(rec["sample_index"]), int(rec["t_start"]))
        tags = tag_lookup.get(key, {})
        matched = [f"{label}:{rank}" for label, rank in tags.items() if label in wanted and rank in wanted[label]]
        if matched:
            selected[key] = dict(rec, curated_tags=sorted(matched, key=lambda x: priority_order.get(x, 10**9)))

    curated = sorted(
        selected.values(),
        key=lambda rec: (
            min(priority_order.get(tag, 10**9) for tag in rec["curated_tags"]),
            rec.get("case_mae", 0.0),
            int(rec["sample_index"]),
        ),
    )
    if len(curated) >= target_count:
        return curated[:target_count]

    existing_keys = {(int(rec["sample_index"]), int(rec["t_start"])) for rec in curated}
    fallback_pool = []
    for rec in records:
        key = (int(rec["sample_index"]), int(rec["t_start"]))
        if key in existing_keys:
            continue
        pred = np.asarray(rec["parsed_prediction"], dtype=np.float32)
        target = np.asarray(rec["target"], dtype=np.float32)
        pred_range = _prediction_range(pred)
        target_range = _prediction_range(target)
        fallback_pool.append(
            dict(
                rec,
                curated_tags=["fallback"],
                case_mae=_mae(pred, target),
                case_rmse=_rmse(pred, target),
                pred_range=pred_range,
                target_range=target_range,
            )
        )
    fallback_pool.sort(key=lambda rec: (rec["case_mae"], -rec["pred_range"], -rec["target_range"], int(rec["sample_index"])))
    needed = max(0, target_count - len(curated))
    curated.extend(fallback_pool[:needed])
    return curated


def _build_svg_report(payload: dict, out_path: Path) -> None:
    metrics = payload["metrics"]

    width = 1200
    curated_cases = payload["curated_cases"]
    per_row = 3
    row_height = 340
    rows_total = max(1, math.ceil(len(curated_cases) / per_row))
    height = 420 + rows_total * row_height + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        _svg_text(40, 36, "Transfer Evaluation Visualization", size=24, weight="bold"),
        _svg_text(
            40,
            60,
            (
                f"{payload['dataset']} / {payload['node_id']} / {payload['prompt_style']} | "
                f"horizon={payload['horizon']} history={payload['history_len']} context={payload['context_history_len']}"
            ),
            size=13,
            fill="#475569",
        ),
    ]

    # Summary cards
    card_y = 90
    card_w = 250
    summary_items = [
        ("Overall MAE (kW)", f"{metrics['overall']['mae']:.2f}"),
        ("Overall RMSE (kW)", f"{metrics['overall']['rmse']:.2f}"),
        ("Median Case MAE (kW)", f"{payload['case_mae_quantiles']['median']:.2f}"),
        ("Median Case RMSE (kW)", f"{payload['case_rmse_quantiles']['median']:.2f}"),
    ]
    for idx, (label, value) in enumerate(summary_items):
        x = 40 + idx * (card_w + 18)
        parts.append(_svg_rect(x, card_y, card_w, 88, fill="#ffffff"))
        parts.append(_svg_text(x + 18, card_y + 30, label, size=12, fill="#64748b"))
        parts.append(_svg_text(x + 18, card_y + 62, value, size=26, weight="bold"))

    # Horizon lines
    plot_x = 60
    plot_y = 225
    plot_w = 500
    plot_h = 230
    parts.append(_svg_rect(plot_x - 20, plot_y - 30, plot_w + 40, plot_h + 60, fill="#ffffff"))
    parts.append(_svg_text(plot_x, plot_y - 8, "Per-Horizon Errors", size=16, weight="bold"))

    horizons = sorted(int(key) for key in metrics.keys() if str(key) != "overall")
    maes = [metrics[str(h)]["mae"] for h in horizons]
    rmses = [metrics[str(h)]["rmse"] for h in horizons]
    y_max = max(rmses) * 1.08
    for tick in range(5):
        frac = tick / 4
        y = plot_y + plot_h - frac * plot_h
        val = frac * y_max
        parts.append(_svg_line(plot_x, y, plot_x + plot_w, y, stroke="#e2e8f0"))
        parts.append(_svg_text(plot_x - 10, y + 4, f"{val:.0f}", size=10, anchor="end", fill="#64748b"))
    if len(horizons) == 1:
        x_positions = [plot_x + plot_w / 2]
    else:
        x_positions = [plot_x + i * plot_w / (len(horizons) - 1) for i in range(len(horizons))]
    for x, h in zip(x_positions, horizons):
        parts.append(_svg_line(x, plot_y, x, plot_y + plot_h, stroke="#f1f5f9"))
        parts.append(_svg_text(x, plot_y + plot_h + 18, h, size=10, anchor="middle", fill="#64748b"))

    def scale_y(val: float) -> float:
        return plot_y + plot_h - (val / y_max) * plot_h

    mae_pts = [(x, scale_y(v)) for x, v in zip(x_positions, maes)]
    rmse_pts = [(x, scale_y(v)) for x, v in zip(x_positions, rmses)]
    parts.append(_svg_polyline(mae_pts, stroke="#2563eb", stroke_width=2.8))
    parts.append(_svg_polyline(rmse_pts, stroke="#dc2626", stroke_width=2.8))
    parts.append(_svg_text(plot_x + 18, plot_y + 18, "Per-horizon MAE (kW)", size=11, fill="#2563eb", weight="bold"))
    parts.append(_svg_text(plot_x + 170, plot_y + 18, "Per-horizon RMSE (kW)", size=11, fill="#dc2626", weight="bold"))

    # Behaviour stats
    stats_x = 640
    stats_y = 225
    stats_w = 500
    stats_h = 230
    parts.append(_svg_rect(stats_x - 20, stats_y - 30, stats_w + 40, stats_h + 60, fill="#ffffff"))
    parts.append(_svg_text(stats_x, stats_y - 8, "Prediction Behaviour", size=16, weight="bold"))
    behaviour_lines = [
        f"Constant prediction ratio: {payload['constant_prediction_ratio']:.3f}",
        f"Mean prediction range (kW): {payload['mean_prediction_range']:.2f}",
        f"Median prediction range (kW): {payload['median_prediction_range']:.2f}",
        f"Case MAE median (kW): {payload['case_mae_quantiles']['median']:.2f}",
        f"Case RMSE median (kW): {payload['case_rmse_quantiles']['median']:.2f}",
    ]
    for idx, line in enumerate(behaviour_lines):
        parts.append(_svg_text(stats_x, stats_y + 26 + idx * 28, line, size=13))

    # Case plots
    row_top = 540
    panel_w = 1120
    plot_left = 70
    colors = {"hist": "#94a3b8", "target": "#111827", "pred": "#16a34a"}
    cursor_y = row_top
    block_rows = max(1, math.ceil(len(curated_cases) / per_row))
    block_h = block_rows * row_height - 20
    parts.append(_svg_rect(40, cursor_y - 24, panel_w, block_h, fill="#ffffff"))
    parts.append(_svg_text(60, cursor_y, "Cases", size=16, weight="bold"))
    for case_idx, case in enumerate(curated_cases):
        row_idx = case_idx // per_row
        col_idx = case_idx % per_row
        left = plot_left + col_idx * 360
        panel_top = cursor_y + 26 + row_idx * row_height
        context_top = panel_top + 18
        context_h = 44
        plot_top = context_top + context_h + 34
        plot_w_case = 300
        plot_h_case = 150
        hist = np.asarray(case["history"], dtype=np.float32)
        long_hist = np.asarray(case.get("long_history_anchors") or [], dtype=np.float32)
        target = np.asarray(case["target"], dtype=np.float32)
        pred = np.asarray(case["prediction"], dtype=np.float32)
        values = np.concatenate([arr for arr in (hist, long_hist, target, pred) if arr.size > 0])
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0
        pad = 0.08 * (vmax - vmin)
        vmin -= pad
        vmax += pad
        if vmin < 0.0:
            vmin = 0.0
            if vmax - vmin < 1e-6:
                vmax = vmin + 1.0

        def scale_y_case(v: float) -> float:
            return plot_top + plot_h_case - ((v - vmin) / (vmax - vmin)) * plot_h_case

        parts.append(
            _svg_text(
                left,
                panel_top,
                (
                    f"sample={case['sample_index']} t={case['t_start']} | "
                    f"MAE={case['case_mae']:.2f} | hist={case['history_steps']} ctx={case['context_steps']}"
                ),
                size=11,
                weight="bold",
            )
        )

        if long_hist.size > 0:
            ctx_min = float(long_hist.min())
            ctx_max = float(long_hist.max())
            if ctx_max - ctx_min < 1e-6:
                ctx_max = ctx_min + 1.0
            ctx_pad = 0.08 * (ctx_max - ctx_min)
            ctx_min -= ctx_pad
            ctx_max += ctx_pad

            def scale_y_context(v: float) -> float:
                return context_top + context_h - ((v - ctx_min) / (ctx_max - ctx_min)) * context_h

            parts.append(_svg_rect(left, context_top, plot_w_case, context_h, fill="#fcfdff", stroke="#dbeafe"))
            ctx_x = [left + i * (plot_w_case / max(len(long_hist) - 1, 1)) for i in range(len(long_hist))]
            ctx_pts = [(x, scale_y_context(v)) for x, v in zip(ctx_x, long_hist)]
            parts.append(_svg_polyline(ctx_pts, stroke="#93c5fd", stroke_width=1.8))
            parts.append(_svg_text(left, context_top - 6, f"long context sampled ({len(long_hist)} anchors)", size=9, fill="#2563eb"))

        parts.append(_svg_text(left, plot_top + plot_h_case + 20, f"RMSE={case['case_rmse']:.2f}", size=10, fill="#475569"))
        parts.append(_svg_rect(left, plot_top, plot_w_case, plot_h_case, fill="#ffffff", stroke="#e2e8f0"))
        for tick in range(5):
            frac = tick / 4
            y = plot_top + plot_h_case - frac * plot_h_case
            parts.append(_svg_line(left, y, left + plot_w_case, y, stroke="#f1f5f9"))
            value = vmin + frac * (vmax - vmin)
            parts.append(_svg_text(left - 8, y + 4, f"{value:.0f}", size=9, anchor="end", fill="#64748b"))
        total_segments = max(len(hist) + len(target) - 1, 1)
        hist_x = [left + i * (plot_w_case / total_segments) for i in range(len(hist))]
        fut_x = [left + (len(hist) - 1 + i) * (plot_w_case / total_segments) for i in range(1, len(target) + 1)]
        hist_pts = [(x, scale_y_case(v)) for x, v in zip(hist_x, hist)]
        tgt_pts = [(x, scale_y_case(v)) for x, v in zip(fut_x, target)]
        pred_pts = [(x, scale_y_case(v)) for x, v in zip(fut_x, pred)]
        if hist_x:
            parts.append(_svg_line(hist_x[-1], plot_top, hist_x[-1], plot_top + plot_h_case, stroke="#cbd5e1", stroke_width=1, dash="4 4"))
        parts.append(_svg_polyline(hist_pts, stroke=colors["hist"], stroke_width=2.0))
        parts.append(_svg_polyline(tgt_pts, stroke=colors["target"], stroke_width=2.4))
        parts.append(_svg_polyline(pred_pts, stroke=colors["pred"], stroke_width=2.4))
        parts.append(_svg_text(left, plot_top + plot_h_case + 38, "history", size=10, fill=colors["hist"]))
        parts.append(_svg_text(left + 60, plot_top + plot_h_case + 38, "target", size=10, fill=colors["target"]))
        parts.append(_svg_text(left + 112, plot_top + plot_h_case + 38, "prediction", size=10, fill=colors["pred"]))

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _build_html_report(payload: dict, svg_name: str, csv_name: str, out_path: Path) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Transfer Visualization</title>
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      background: #f8fafc;
      color: #0f172a;
    }}
    .meta {{
      margin-bottom: 16px;
      color: #475569;
      font-size: 14px;
    }}
    .links a {{
      margin-right: 16px;
      color: #2563eb;
      text-decoration: none;
    }}
    .panel {{
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 16px;
      margin-top: 16px;
    }}
    iframe {{
      width: 100%;
      height: 1780px;
      border: 0;
      background: white;
    }}
  </style>
</head>
<body>
  <h1>Transfer Experiment Visualization</h1>
  <div class="meta">
    dataset={escape(payload['dataset'])}, node_id={escape(payload['node_id'])}, prompt_style={escape(payload['prompt_style'])},
    requested={payload['requested_samples']}, parse_success_rate={payload['parse_success_rate']:.3f}
  </div>
  <div class="links">
    <a href="{escape(svg_name)}">SVG report</a>
    <a href="{escape(csv_name)}">CSV summary</a>
  </div>
  <div class="panel">
    <iframe src="{escape(svg_name)}"></iframe>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize transfer-evaluation JSON results.")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--case-sample-indices",
        default="",
        help="Optional comma-separated sample_index list to use as the case gallery, e.g. 246,224,282,343",
    )
    args = parser.parse_args()

    result_path = Path(args.result_json)
    output_dir = Path(args.output_dir)
    data = _load_json(result_path)
    records = [rec for rec in data.get("generations", []) if rec.get("parse_ok")]
    if not records:
        raise ValueError("No parseable generations found in result JSON.")

    raw = load_dataset(data["dataset"])
    norm_min = float(raw.get("norm_min", 0.0))
    norm_max = float(raw.get("norm_max", 1.0))
    horizon = int(data.get("horizon") or max(int(key) for key in data.get("metrics", {}).keys() if str(key) != "overall"))
    history_len = int(data.get("history_len", 12))
    context_history_len = int(data.get("context_history_len", 0) or 0)
    neighbor_k = int(data.get("neighbor_k", 7))
    window_stride = resolve_window_stride(int(data.get("window_stride", 0) or 0), horizon=horizon)

    splits = build_splits(raw, data["dataset"])
    test_map = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[horizon],
        history_len=history_len,
        context_history_len=context_history_len,
        neighbor_k=neighbor_k,
        window_stride=window_stride,
    )
    sample_lookup = {int(sample["t_start"]): sample for sample in test_map[horizon]}

    cases = _select_cases(records)
    tag_lookup = _case_tags(cases)

    csv_rows = []
    for rec in records:
        pred = _denormalize(np.asarray(rec["parsed_prediction"], dtype=np.float32), norm_min, norm_max)
        target = _denormalize(np.asarray(rec["target"], dtype=np.float32), norm_min, norm_max)
        key = (int(rec["sample_index"]), int(rec["t_start"]))
        tags = tag_lookup.get(key, {})
        csv_rows.append(
            {
                "sample_index": rec["sample_index"],
                "t_start": rec["t_start"],
                "node_idx": rec["node_idx"],
                "case_mae": _mae(pred, target),
                "case_rmse": _rmse(pred, target),
                "prediction_range": _prediction_range(pred),
                "is_constant_prediction": float(_prediction_range(pred) < 1e-6),
                "tags": "|".join(f"{label}:{rank}" for label, rank in sorted(tags.items())),
                "prediction": json.dumps([round(float(v), 6) for v in pred.tolist()]),
                "target": json.dumps([round(float(v), 6) for v in target.tolist()]),
            }
        )
    payload = {
        "dataset": data["dataset"],
        "node_id": data.get("node_id"),
        "prompt_style": data.get("prompt_style"),
        "horizon": horizon,
        "history_len": history_len,
        "context_history_len": context_history_len,
        "requested_samples": int(data.get("requested_samples", len(records))),
        "evaluated_samples": int(data.get("evaluated_samples", len(records))),
        "parse_success_rate": float(data.get("parse_success_rate", 1.0)),
        "metrics": data["metrics"],
        "constant_prediction_ratio": _constant_ratio(records, norm_min=norm_min, norm_max=norm_max),
        "mean_prediction_range": float(np.mean([_prediction_range(_denormalize(np.asarray(rec["parsed_prediction"], dtype=np.float32), norm_min, norm_max)) for rec in records])),
        "median_prediction_range": float(median([_prediction_range(_denormalize(np.asarray(rec["parsed_prediction"], dtype=np.float32), norm_min, norm_max)) for rec in records])),
        "case_mae_quantiles": _quantiles([
            {"case_mae": _mae(_denormalize(np.asarray(rec["parsed_prediction"], dtype=np.float32), norm_min, norm_max), _denormalize(np.asarray(rec["target"], dtype=np.float32), norm_min, norm_max))}
            for rec in records
        ], "case_mae"),
        "case_rmse_quantiles": _quantiles([
            {"case_rmse": _rmse(_denormalize(np.asarray(rec["parsed_prediction"], dtype=np.float32), norm_min, norm_max), _denormalize(np.asarray(rec["target"], dtype=np.float32), norm_min, norm_max))}
            for rec in records
        ], "case_rmse"),
        "records": records,
    }

    case_payload = {}
    for key, rows in cases.items():
        out_rows = []
        for rec in rows:
            tag_key = (int(rec["sample_index"]), int(rec["t_start"]))
            tags = tag_lookup.get(tag_key, {})
            sample = sample_lookup.get(int(rec["t_start"]))
            if sample is None:
                raise KeyError(f"Unable to reconstruct test sample for t_start={rec['t_start']}")
            case = _extract_case_series(rec, sample, norm_min=norm_min, norm_max=norm_max)
            case["tags"] = [f"{label}:{rank}" for label, rank in sorted(tags.items())]
            out_rows.append(case)
        case_payload[key] = out_rows
    payload["cases"] = case_payload

    explicit_case_indices = []
    if args.case_sample_indices.strip():
        explicit_case_indices = [int(part.strip()) for part in args.case_sample_indices.split(",") if part.strip()]

    curated_records = _select_curated_cases(
        records,
        cases,
        explicit_sample_indices=explicit_case_indices or None,
    )
    curated_payload = []
    for rec in curated_records:
        sample = sample_lookup.get(int(rec["t_start"]))
        if sample is None:
            raise KeyError(f"Unable to reconstruct curated test sample for t_start={rec['t_start']}")
        case = _extract_case_series(rec, sample, norm_min=norm_min, norm_max=norm_max)
        case["tags"] = rec["curated_tags"]
        curated_payload.append(case)
    payload["curated_cases"] = curated_payload

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "transfer_cases.csv"
    svg_path = output_dir / "transfer_report.svg"
    html_path = output_dir / "transfer_report.html"

    _write_csv(
        csv_path,
        csv_rows,
        fieldnames=[
            "sample_index",
            "t_start",
            "node_idx",
            "case_mae",
            "case_rmse",
            "prediction_range",
            "is_constant_prediction",
            "tags",
            "prediction",
            "target",
        ],
    )
    _build_svg_report(payload, svg_path)
    _build_html_report(payload, svg_path.name, csv_path.name, html_path)
    print(f"Saved HTML: {html_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
