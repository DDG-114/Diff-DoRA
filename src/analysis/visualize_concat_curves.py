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


def _plot_domain(domain: str, node_payloads: list[dict], output_path: Path):
    width = 1500
    row_h = 280
    height = 70 + row_h * len(node_payloads)
    left = 70
    plot_w = 1360
    plot_h = 180

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        _svg_text(
            width / 2,
            28,
            f"{domain} | 24-hour concatenated non-overlapping windows | 3 nodes",
            size=18,
            anchor="middle",
            weight="bold",
        ),
    ]

    for row, payload in enumerate(node_payloads):
        top = 60 + row * row_h
        truth = payload["truth"]
        full_pred = payload["full_prediction"]
        wo_pred = payload["wo_prediction"]
        timestamps = payload["timestamps"]
        node_idx = payload["node_idx"]
        full_metrics = payload["full_metrics"]
        wo_metrics = payload["wo_metrics"]

        valid_full = np.array([v for v in full_pred if v is not None], dtype=np.float32)
        valid_wo = np.array([v for v in wo_pred if v is not None], dtype=np.float32)
        values = np.concatenate([np.array(truth, dtype=np.float32), valid_full, valid_wo])
        y_min = float(values.min())
        y_max = float(values.max())
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad

        def scale_y(v: float) -> float:
            return top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

        xs = [left + (i / max(len(truth) - 1, 1)) * plot_w for i in range(len(truth))]

        def build_points(series):
            points = []
            segments = []
            for x, y in zip(xs, series):
                if y is None:
                    if points:
                        segments.append(points)
                        points = []
                    continue
                points.append((x, scale_y(float(y))))
            if points:
                segments.append(points)
            return segments

        parts.append(_svg_text(
            left,
            top - 12,
            (
                f"node={node_idx} | "
                f"Full: MAE={full_metrics['mae']:.4f}, RMSE={full_metrics['rmse']:.4f} | "
                f"w/o DiffDoRA: MAE={wo_metrics['mae']:.4f}, RMSE={wo_metrics['rmse']:.4f}"
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

        for i in range(5):
            idx = int(round(i * max(len(xs) - 1, 1) / 4))
            x = xs[idx]
            parts.append(_svg_line(x, top, x, top + plot_h, stroke="#f3f4f6", stroke_width=1))
            label = timestamps[idx].replace("T", " ")[:16]
            parts.append(_svg_text(x, top + plot_h + 18, label, size=10, anchor="middle", fill="#4b5563"))

        for seg in build_points(truth):
            parts.append(_svg_polyline(seg, stroke="#111827", stroke_width=2.6))
        for seg in build_points(full_pred):
            parts.append(_svg_polyline(seg, stroke="#2d6a4f", stroke_width=2.2))
        for seg in build_points(wo_pred):
            parts.append(_svg_polyline(seg, stroke="#1d4ed8", stroke_width=2.2))

        legend_y = top + plot_h + 42
        legend_x = left
        for label, color in [("Ground Truth", "#111827"), ("Full", "#2d6a4f"), ("w/o DiffDoRA", "#1d4ed8")]:
            parts.append(_svg_line(legend_x, legend_y - 4, legend_x + 18, legend_y - 4, stroke=color, stroke_width=3))
            parts.append(_svg_text(legend_x + 24, legend_y, label, size=11))
            legend_x += 180

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Visualize concatenated non-overlapping window curves.")
    parser.add_argument("--full-json", required=True)
    parser.add_argument("--wo-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    full = _load_json(Path(args.full_json))
    wo = _load_json(Path(args.wo_json))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "full_json": args.full_json,
        "wo_json": args.wo_json,
        "domains": {},
    }

    for domain in ("CBD", "Residential"):
        node_payloads = []
        for node_idx, full_node in full["domains"][domain].items():
            wo_node = wo["domains"][domain][node_idx]
            node_payloads.append({
                "node_idx": int(node_idx),
                "timestamps": full_node["timestamps"],
                "truth": full_node["truth"],
                "full_prediction": full_node["prediction"],
                "wo_prediction": wo_node["prediction"],
                "full_metrics": full_node["metrics"],
                "wo_metrics": wo_node["metrics"],
            })
        fig_path = out_dir / f"{domain.lower()}_concat_curve.svg"
        _plot_domain(domain, node_payloads, fig_path)
        summary["domains"][domain] = {
            "figure": str(fig_path),
            "nodes": {
                str(item["node_idx"]): {
                    "full_metrics": item["full_metrics"],
                    "wo_metrics": item["wo_metrics"],
                }
                for item in node_payloads
            },
        }

    summary_path = out_dir / "concat_curve_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
