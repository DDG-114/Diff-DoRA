from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _svg_polyline(points, *, stroke, stroke_width=2.2):
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" />'
    )


def _plot_detail_window(domain: str, payloads: list[dict], output_path: Path, title_suffix: str):
    width = 1500
    row_h = 260
    height = 70 + row_h * len(payloads)
    left = 70
    plot_w = 1360
    plot_h = 150
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        _svg_text(width / 2, 28, f"{domain} | {title_suffix}", size=18, anchor="middle", weight="bold"),
    ]

    for row, payload in enumerate(payloads):
        top = 60 + row * row_h
        truth = payload["truth"]
        full_pred = payload["full_prediction"]
        wo_pred = payload["wo_prediction"]
        timestamps = payload["timestamps"]
        node_idx = payload["node_idx"]
        all_vals = np.array(truth + full_pred + wo_pred, dtype=np.float32)
        y_min = float(all_vals.min())
        y_max = float(all_vals.max())
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad

        def scale_y(v: float) -> float:
            return top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

        xs = [left + (i / max(len(truth) - 1, 1)) * plot_w for i in range(len(truth))]
        parts.append(_svg_text(
            left,
            top - 12,
            (
                f"node={node_idx} | {timestamps[0].replace('T',' ')[:16]} -> "
                f"{timestamps[-1].replace('T',' ')[:16]}"
            ),
            size=12,
            weight="bold",
        ))
        parts.append(_svg_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#374151", stroke_width=1.2))
        parts.append(_svg_line(left, top, left, top + plot_h, stroke="#374151", stroke_width=1.2))
        for tick in range(5):
            frac = tick / 4
            y = top + plot_h - frac * plot_h
            v = y_min + frac * (y_max - y_min)
            parts.append(_svg_line(left, y, left + plot_w, y, stroke="#e5e7eb", stroke_width=1))
            parts.append(_svg_text(left - 8, y + 4, f"{v:.1f}", size=10, anchor="end", fill="#4b5563"))
        for i in range(len(xs)):
            if i % max(1, len(xs) // 4) == 0 or i == len(xs) - 1:
                parts.append(_svg_text(xs[i], top + plot_h + 18, timestamps[i].replace("T", " ")[11:16], size=10, anchor="middle", fill="#4b5563"))

        series = [("Ground Truth", "#111827", truth), ("Full", "#2d6a4f", full_pred), ("w/o DiffDoRA", "#1d4ed8", wo_pred)]
        for _, color, vals in series:
            pts = [(x, scale_y(v)) for x, v in zip(xs, vals)]
            parts.append(_svg_polyline(pts, stroke=color, stroke_width=2.4 if color == "#111827" else 2.1))
        legend_y = top + plot_h + 42
        legend_x = left
        for label, color in [("Ground Truth", "#111827"), ("Full", "#2d6a4f"), ("w/o DiffDoRA", "#1d4ed8")]:
            parts.append(_svg_line(legend_x, legend_y - 4, legend_x + 18, legend_y - 4, stroke=color, stroke_width=3))
            parts.append(_svg_text(legend_x + 24, legend_y, label, size=11))
            legend_x += 180

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _best_gap_segments(full_node: dict, wo_node: dict, *, segment_len: int) -> tuple[int, float]:
    full_pred = full_node["prediction"]
    wo_pred = wo_node["prediction"]
    best_start = 0
    best_gap = -1.0
    for start in range(0, len(full_pred) - segment_len + 1):
        gap = 0.0
        valid = 0
        for i in range(start, start + segment_len):
            fp = full_pred[i]
            wp = wo_pred[i]
            if fp is None or wp is None:
                continue
            gap += abs(fp - wp)
            valid += 1
        if valid == 0:
            continue
        gap /= valid
        if gap > best_gap:
            best_gap = gap
            best_start = start
    return best_start, best_gap


def main():
    parser = argparse.ArgumentParser(description="Create zoomed local detail figures from existing curve/window JSONs.")
    parser.add_argument("--daily-full-json", required=True)
    parser.add_argument("--daily-wo-json", required=True)
    parser.add_argument("--concat-full-json", required=True)
    parser.add_argument("--concat-wo-json", required=True)
    parser.add_argument("--window-summary-json", required=True)
    parser.add_argument("--window-full-json", required=True)
    parser.add_argument("--window-wo-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    daily_full = _load_json(Path(args.daily_full_json))
    daily_wo = _load_json(Path(args.daily_wo_json))
    concat_full = _load_json(Path(args.concat_full_json))
    concat_wo = _load_json(Path(args.concat_wo_json))
    window_summary = _load_json(Path(args.window_summary_json))
    patch_full = _load_json(Path(args.window_full_json))
    patch_wo = _load_json(Path(args.window_wo_json))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"figures": {}}

    # Daily/detail and concat/detail: 1-hour local segments (12 points)
    for mode, full_json, wo_json, prefix in [
        ("daily", daily_full, daily_wo, "daily"),
        ("concat", concat_full, concat_wo, "concat"),
    ]:
        for domain in ("CBD", "Residential"):
            payloads = []
            for node_idx in full_json["domains"][domain].keys():
                full_node = full_json["domains"][domain][node_idx]
                wo_node = wo_json["domains"][domain][node_idx]
                start, gap = _best_gap_segments(full_node, wo_node, segment_len=12)
                end = start + 12
                if "timestamps" in full_node:
                    timestamps = full_node["timestamps"][start:end]
                else:
                    timestamps = full_json["timestamps"][start:end]
                payloads.append({
                    "node_idx": int(node_idx),
                    "timestamps": timestamps,
                    "truth": full_node["truth"][start:end],
                    "full_prediction": full_node["prediction"][start:end],
                    "wo_prediction": wo_node["prediction"][start:end],
                    "gap": gap,
                })
            fig_path = out_dir / f"{domain.lower()}_{prefix}_detail.svg"
            _plot_detail_window(domain, payloads, fig_path, f"{mode} local detail (largest 1-hour variant gap)")
            summary["figures"][f"{domain.lower()}_{prefix}_detail"] = str(fig_path)

    # Scheme A future-only zoom: 6-step target/prediction region
    full_lookup = {(c["domain"], c["node_idx"], c["sample_dataset_index"]): c for c in patch_full["cases"]}
    wo_lookup = {(c["domain"], c["node_idx"], c["sample_dataset_index"]): c for c in patch_wo["cases"]}
    for domain in ("CBD", "Residential"):
        payloads = []
        for row in window_summary["domains"][domain]["single_windows"]:
            key = (domain, row["node_idx"], row["sample_dataset_index"])
            f = full_lookup[key]
            w = wo_lookup[key]
            payloads.append({
                "node_idx": row["node_idx"],
                "timestamps": f["future_timestamps"],
                "truth": f["target"],
                "full_prediction": f["prediction"],
                "wo_prediction": w["prediction"],
            })
        fig_path = out_dir / f"{domain.lower()}_scheme_a_future_zoom.svg"
        _plot_detail_window(domain, payloads, fig_path, "scheme A future-horizon zoom")
        summary["figures"][f"{domain.lower()}_scheme_a_future_zoom"] = str(fig_path)

    summary_path = out_dir / "detail_clips_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
