from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False

VARIANT_ORDER = ["wo_diffdora", "full", "wo_cot", "wo_rag", "wo_dora", "wo_moe", "base_model"]
VARIANT_LABELS = {
    "wo_diffdora": "Full",
    "full": "w/o DiffDoRA",
    "wo_cot": "w/o CoT",
    "wo_rag": "w/o RAG",
    "wo_dora": "w/o DoRA",
    "wo_moe": "w/o MoE",
    "base_model": "Base",
}
VARIANT_COLORS = {
    "full": "#2d6a4f",
    "wo_diffdora": "#1d4ed8",
    "wo_cot": "#d97706",
    "wo_rag": "#be185d",
    "wo_dora": "#2563eb",
    "wo_moe": "#7c3aed",
    "base_model": "#6b7280",
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _variant_sort_key(name: str):
    if name in VARIANT_ORDER:
        return (0, VARIANT_ORDER.index(name))
    return (1, name)


def _metric_value(metric_block: dict | None, *keys, default=math.nan):
    cur = metric_block
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            if key in cur:
                cur = cur[key]
            elif str(key) in cur:
                cur = cur[str(key)]
            else:
                return default
        else:
            return default
    if cur is None:
        return default
    return float(cur)


def _summary_rows(results: dict, horizon: int) -> list[dict]:
    rows = []
    for name in sorted(results.keys(), key=_variant_sort_key):
        payload = results[name]
        row = {
            "variant": name,
            "label": VARIANT_LABELS.get(name, name),
            "overall_rmse": _metric_value(payload.get("metrics"), "overall", "rmse"),
            "overall_mae": _metric_value(payload.get("metrics"), "overall", "mae"),
            "parse_success_rate": float(payload.get("parse_success_rate", math.nan)),
            "requested_predictions": int(payload.get("requested_predictions", 0)),
            "parsed_predictions": int(payload.get("parsed_predictions", 0)),
            "cbd_mae": _metric_value(payload.get("domain_metrics", {}).get("CBD"), "overall", "mae"),
            "residential_mae": _metric_value(payload.get("domain_metrics", {}).get("Residential"), "overall", "mae"),
            "max_new_tokens": int(payload.get("max_new_tokens", 0)),
        }
        for h in range(1, horizon + 1):
            row[f"h{h}_rmse"] = _metric_value(payload.get("metrics"), h, "rmse")
            row[f"h{h}_mae"] = _metric_value(payload.get("metrics"), h, "mae")
        rows.append(row)
    return rows


def _write_summary_csv(rows: list[dict], output_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: float, default: float = 0.0) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return float(value)


def _display_cap(values: list[float], *, pad: float = 1.15, outlier_ratio: float = 4.0) -> tuple[float, bool]:
    clean = sorted(_clean(v) for v in values)
    if not clean:
        return 1.0, False
    if len(clean) == 1:
        ymax = clean[-1] * pad
        return (ymax if ymax > 0 else 1.0), False
    max_v = clean[-1]
    second_v = clean[-2]
    if second_v > 0 and max_v > second_v * outlier_ratio:
        ymax = second_v * pad
        return (ymax if ymax > 0 else 1.0), True
    ymax = max_v * pad
    return (ymax if ymax > 0 else 1.0), False


def _svg_text(x, y, text, size=12, fill="#111827", anchor="middle", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-weight="{weight}">{escape(str(text))}</text>'
    )


def _svg_rect(x, y, w, h, fill, stroke="none", stroke_width=1, rx=0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}" ry="{rx}" />'
    )


def _svg_line(x1, y1, x2, y2, stroke="#9ca3af", stroke_width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{dash_attr} />'
    )


def _svg_polyline(points, stroke, stroke_width=2.5):
    coords = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" />'
    )


def _bar_chart_svg(title, labels, values, colors, width=390, height=320, y_max=None, percent=False):
    values = [_clean(v) for v in values]
    clipped = False
    if y_max is None:
        y_max, clipped = _display_cap(values)
    if y_max <= 0:
        y_max = 1.0
    if percent:
        y_max = max(y_max, 1.0)
    margin_left, margin_right, margin_top, margin_bottom = 48, 18, 30, 58
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    step = plot_w / max(len(values), 1)
    bar_w = step * 0.6
    parts = [
        _svg_text(width / 2, 18, title, size=15, weight="bold"),
        _svg_line(margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h, stroke="#374151", stroke_width=1.2),
        _svg_line(margin_left, margin_top, margin_left, margin_top + plot_h, stroke="#374151", stroke_width=1.2),
    ]
    for i in range(5):
        frac = i / 4
        y = margin_top + plot_h - frac * plot_h
        tick_val = frac * y_max
        parts.append(_svg_line(margin_left, y, margin_left + plot_w, y, stroke="#e5e7eb", stroke_width=1))
        label = f"{tick_val:.2f}" if percent else f"{tick_val:.1f}"
        parts.append(_svg_text(margin_left - 8, y + 4, label, size=10, anchor="end", fill="#4b5563"))
    for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
        x_center = margin_left + idx * step + step / 2
        display_value = min(value, y_max)
        bar_h = (display_value / y_max) * plot_h if y_max else 0
        parts.append(_svg_rect(x_center - bar_w / 2, margin_top + plot_h - bar_h, bar_w, bar_h, color, rx=2))
        parts.append(_svg_text(x_center, margin_top + plot_h + 18, label, size=11))
        label_text = f"{value:.2f}" if value < 1000 else f"{value:.1f}"
        parts.append(_svg_text(x_center, margin_top + plot_h - bar_h - 6, label_text, size=10))
        if clipped and value > y_max:
            parts.append(_svg_text(x_center, margin_top + 10, "clipped", size=9, fill="#7c2d12"))
    return parts


def _plot_overview_svg(rows: list[dict], output_path: Path):
    labels = [row["label"] for row in rows]
    variants = [row["variant"] for row in rows]
    colors = [VARIANT_COLORS.get(v, "#4b5563") for v in variants]
    width, height = 840, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, "#ffffff"),
        _svg_text(width / 2, 24, "Ablation Overview", size=19, weight="bold"),
    ]
    rmse_vals = [row["overall_rmse"] for row in rows]
    mae_vals = [row["overall_mae"] for row in rows]
    for idx, chart in enumerate([
        _bar_chart_svg("Overall RMSE", labels, rmse_vals, colors),
        _bar_chart_svg("Overall MAE", labels, mae_vals, colors),
    ]):
        parts.append(f'<g transform="translate({20 + idx * 400},50)">')
        parts.extend(chart)
        parts.append('</g>')
    parts.append('</svg>')
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _plot_detail_svg(rows: list[dict], horizon: int, output_path: Path):
    labels = [row["label"] for row in rows]
    variants = [row["variant"] for row in rows]
    colors = [VARIANT_COLORS.get(v, "#4b5563") for v in variants]
    width, height = 1260, 440
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, "#ffffff"),
        _svg_text(width / 2, 24, "Expert-Only Ablation Details", size=19, weight="bold"),
    ]
    parts.append('<g transform="translate(20,60)">')
    domain_colors = ["#60a5fa", "#f59e0b"] * len(rows)
    domain_labels = []
    domain_values = []
    domain_bar_colors = []
    for row in rows:
        domain_labels.extend([f"{row['label']} C", f"{row['label']} R"])
        domain_values.extend([row["cbd_mae"], row["residential_mae"]])
        domain_bar_colors.extend(["#60a5fa", "#f59e0b"])
    parts.extend(_bar_chart_svg("Domain MAE", domain_labels, domain_values, domain_bar_colors, width=430, height=340))
    parts.append('</g>')

    plot_x, plot_y, plot_w, plot_h = 500, 95, 720, 270
    parts.append(_svg_text(860, 78, "Per-Horizon RMSE", size=15, weight="bold"))
    parts.append(_svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#374151", stroke_width=1.2))
    parts.append(_svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#374151", stroke_width=1.2))
    horizons = list(range(1, horizon + 1))
    max_rmse, clipped_rmse = _display_cap([_clean(row[f"h{h}_rmse"]) for row in rows for h in horizons])
    for i in range(5):
        frac = i / 4
        y = plot_y + plot_h - frac * plot_h
        parts.append(_svg_line(plot_x, y, plot_x + plot_w, y, stroke="#e5e7eb", stroke_width=1))
        parts.append(_svg_text(plot_x - 8, y + 4, f"{(frac * max_rmse):.1f}", size=10, anchor="end", fill="#4b5563"))
    for idx, h in enumerate(horizons):
        x = plot_x + (idx / max(len(horizons) - 1, 1)) * plot_w if len(horizons) > 1 else plot_x + plot_w / 2
        parts.append(_svg_line(x, plot_y, x, plot_y + plot_h, stroke="#f3f4f6", stroke_width=1))
        parts.append(_svg_text(x, plot_y + plot_h + 20, h, size=11))
    for row, color in zip(rows, colors):
        pts = []
        for idx, h in enumerate(horizons):
            x = plot_x + (idx / max(len(horizons) - 1, 1)) * plot_w if len(horizons) > 1 else plot_x + plot_w / 2
            y_val = _clean(row[f"h{h}_rmse"])
            y = plot_y + plot_h - (min(y_val, max_rmse) / max_rmse) * plot_h
            pts.append((x, y))
        parts.append(_svg_polyline(pts, color))
        for idx, (x, y) in enumerate(pts):
            parts.append(_svg_rect(x - 3, y - 3, 6, 6, color, rx=3))
            if clipped_rmse:
                real_val = _clean(row[f"h{horizons[idx]}_rmse"])
                if real_val > max_rmse:
                    label_text = f"{real_val:.1f}" if real_val < 1000 else f"{real_val:.0f}"
                    parts.append(_svg_text(x, plot_y + 12, label_text, size=9, fill=color))
    for idx, row in enumerate(rows):
        x = 540 + idx * 165
        parts.append(_svg_rect(x, 388, 18, 10, colors[idx], rx=2))
        parts.append(_svg_text(x + 26, 397, row["label"], size=11, anchor="start"))
    parts.append('</svg>')
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _plot_overview_matplotlib(rows: list[dict], output_path: Path):
    labels = [row["label"] for row in rows]
    variants = [row["variant"] for row in rows]
    colors = [VARIANT_COLORS.get(name, "#4b5563") for name in variants]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6), dpi=170)
    metrics = [
        ("overall_rmse", "Overall RMSE"),
        ("overall_mae", "Overall MAE"),
    ]
    x = np.arange(len(rows))
    for ax, (key, title) in zip(axes, metrics):
        vals = [row[key] for row in rows]
        ymax, clipped = _display_cap(vals)
        disp = [min(v, ymax) for v in vals]
        ax.bar(x, disp, color=colors, width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, ymax if ymax > 0 else 1.0)
        if clipped:
            for xi, raw, shown in zip(x, vals, disp):
                if raw > ymax:
                    txt = f"{raw:.1f}" if raw < 1000 else f"{raw:.0f}"
                    ax.text(xi, shown, txt, ha="center", va="bottom", fontsize=8, rotation=90)
    fig.suptitle("Expert-Only Ablation Overview", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_detail_matplotlib(rows: list[dict], horizon: int, output_path: Path):
    labels = [row["label"] for row in rows]
    variants = [row["variant"] for row in rows]
    colors = [VARIANT_COLORS.get(name, "#4b5563") for name in variants]
    fig = plt.figure(figsize=(12.8, 5.2), dpi=170)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(rows))
    width = 0.35
    cbd_vals = [row["cbd_mae"] for row in rows]
    res_vals = [row["residential_mae"] for row in rows]
    domain_vals = cbd_vals + res_vals
    ymax, clipped = _display_cap(domain_vals)
    ax1.bar(x - width / 2, [min(v, ymax) for v in cbd_vals], width, label="CBD", color="#60a5fa")
    ax1.bar(x + width / 2, [min(v, ymax) for v in res_vals], width, label="Residential", color="#f59e0b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.set_title("Domain MAE")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False)
    ax1.set_ylim(0, ymax if ymax > 0 else 1.0)
    if clipped:
        for xi, raw in zip(x - width / 2, cbd_vals):
            if raw > ymax:
                ax1.text(xi, ymax, f"{raw:.1f}" if raw < 1000 else f"{raw:.0f}", ha="center", va="bottom", fontsize=8, rotation=90)
        for xi, raw in zip(x + width / 2, res_vals):
            if raw > ymax:
                ax1.text(xi, ymax, f"{raw:.1f}" if raw < 1000 else f"{raw:.0f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax2 = fig.add_subplot(gs[0, 1])
    horizons = np.arange(1, horizon + 1)
    rmse_vals = [row[f"h{h}_rmse"] for row in rows for h in horizons]
    ymax, clipped = _display_cap(rmse_vals)
    for row, color in zip(rows, colors):
        vals = [row[f"h{h}_rmse"] for h in horizons]
        ax2.plot(horizons, [min(v, ymax) for v in vals], marker="o", linewidth=2.0, color=color, label=row["label"])
    ax2.set_xticks(horizons)
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Per-Horizon RMSE")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    ax2.set_ylim(0, ymax if ymax > 0 else 1.0)
    if clipped:
        for row, color in zip(rows, colors):
            vals = [row[f"h{h}_rmse"] for h in horizons]
            for hx, raw in zip(horizons, vals):
                if raw > ymax:
                    ax2.text(hx, ymax, f"{raw:.1f}" if raw < 1000 else f"{raw:.0f}", color=color, fontsize=8, ha="center", va="bottom")
    fig.suptitle("Expert-Only Ablation Details", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize expert-only ablation results.")
    parser.add_argument("--ablation-json", required=True,
                        help="Path to the expert ablation JSON file.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for summary files and figures.")
    args = parser.parse_args()

    ablation_path = Path(args.ablation_json)
    if not ablation_path.exists():
        raise FileNotFoundError(f"Ablation JSON not found: {ablation_path}")
    payload = _load_json(ablation_path)
    results = payload.get("results", {})
    horizon = int(payload.get("horizon", 0))
    rows = _summary_rows(results, horizon)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    ext = ".png" if HAS_MATPLOTLIB else ".svg"
    overview_path = out_dir / f"overview{ext}"
    detail_path = out_dir / f"domain_horizon{ext}"

    _write_summary_csv(rows, summary_csv)
    if HAS_MATPLOTLIB:
        _plot_overview_matplotlib(rows, overview_path)
        _plot_detail_matplotlib(rows, horizon, detail_path)
        backend = "matplotlib"
    else:
        _plot_overview_svg(rows, overview_path)
        _plot_detail_svg(rows, horizon, detail_path)
        backend = "svg_fallback"

    summary_payload = {
        "input": str(ablation_path),
        "dataset": payload.get("dataset"),
        "horizon": horizon,
        "render_backend": backend,
        "variants": rows,
        "figures": {
            "overview": str(overview_path),
            "domain_horizon": str(detail_path),
        },
        "summary_csv": str(summary_csv),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved overview figure: {overview_path}")
    print(f"Saved detail figure: {detail_path}")


if __name__ == "__main__":
    main()
