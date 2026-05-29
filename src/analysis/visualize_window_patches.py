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


def _case_metrics(pred, target):
    pred_arr = np.array(pred, dtype=np.float32)
    true_arr = np.array(target, dtype=np.float32)
    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    return {"mae": mae, "rmse": rmse}


def _draw_single_case(case, *, top, left, plot_w, plot_h):
    history = case["history"]
    target = case["target"]
    full = case["full_prediction"]
    wo = case["wo_prediction"]
    all_vals = np.array(history + target + full + wo, dtype=np.float32)
    y_min = float(all_vals.min())
    y_max = float(all_vals.max())
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    def scale_y(v: float) -> float:
        return top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

    xs = [left + (i / 17) * plot_w for i in range(18)]
    hist_pts = [(xs[i], scale_y(v)) for i, v in enumerate(history)]
    target_pts = [(xs[12 + i], scale_y(v)) for i, v in enumerate(target)]
    full_pts = [(xs[12 + i], scale_y(v)) for i, v in enumerate(full)]
    wo_pts = [(xs[12 + i], scale_y(v)) for i, v in enumerate(wo)]

    parts = [
        _svg_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#374151", stroke_width=1.1),
        _svg_line(left, top, left, top + plot_h, stroke="#374151", stroke_width=1.1),
        _svg_line(xs[11], top, xs[11], top + plot_h, stroke="#9ca3af", stroke_width=1.2, dash="4 4"),
    ]
    for tick in range(5):
        frac = tick / 4
        y = top + plot_h - frac * plot_h
        v = y_min + frac * (y_max - y_min)
        parts.append(_svg_line(left, y, left + plot_w, y, stroke="#e5e7eb", stroke_width=1))
        parts.append(_svg_text(left - 6, y + 4, f"{v:.1f}", size=9, anchor="end", fill="#4b5563"))

    parts.append(_svg_polyline(hist_pts, stroke="#6b7280", stroke_width=2.0))
    parts.append(_svg_polyline(target_pts, stroke="#111827", stroke_width=2.5))
    parts.append(_svg_polyline(full_pts, stroke="#2d6a4f", stroke_width=2.1))
    parts.append(_svg_polyline(wo_pts, stroke="#1d4ed8", stroke_width=2.1))
    return parts


def _plot_single_windows(domain: str, cases: list[dict], output_path: Path):
    width = 1400
    row_h = 260
    height = 70 + row_h * len(cases)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        _svg_text(width / 2, 28, f"{domain} | 方案A：单窗口放大图", size=18, anchor="middle", weight="bold"),
    ]

    for row, case in enumerate(cases):
        top = 60 + row * row_h
        left = 70
        plot_w = 1260
        plot_h = 150
        parts.append(_svg_text(
            left,
            top - 12,
            (
                f"node={case['node_idx']} sample={case['sample_dataset_index']} t={case['t_start']} | "
                f"Full MAE={case['full_metrics']['mae']:.4f}, w/o DiffDoRA MAE={case['wo_metrics']['mae']:.4f}"
            ),
            size=12,
            weight="bold",
        ))
        parts.extend(_draw_single_case(case, top=top, left=left, plot_w=plot_w, plot_h=plot_h))

        legend_y = top + plot_h + 36
        legend_x = left
        for label, color in [("History", "#6b7280"), ("Ground Truth", "#111827"), ("Full", "#2d6a4f"), ("w/o DiffDoRA", "#1d4ed8")]:
            parts.append(_svg_line(legend_x, legend_y - 4, legend_x + 18, legend_y - 4, stroke=color, stroke_width=3))
            parts.append(_svg_text(legend_x + 24, legend_y, label, size=10))
            legend_x += 150

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _plot_collage(domain: str, node_to_cases: dict[int, list[dict]], output_path: Path):
    width = 1500
    height = 820
    left0 = 70
    top0 = 80
    col_w = 430
    row_h = 220
    plot_w = 360
    plot_h = 110
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        _svg_text(width / 2, 28, f"{domain} | 方案B：多窗口拼贴图", size=18, anchor="middle", weight="bold"),
    ]
    node_items = list(node_to_cases.items())
    for row, (node_idx, cases) in enumerate(node_items):
        for col, case in enumerate(cases):
            left = left0 + col * col_w
            top = top0 + row * row_h
            parts.append(_svg_text(
                left,
                top - 10,
                f"node={node_idx} sample={case['sample_dataset_index']} t={case['t_start']}",
                size=11,
                weight="bold",
            ))
            parts.extend(_draw_single_case(case, top=top, left=left, plot_w=plot_w, plot_h=plot_h))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Render Scheme-A/B sliding-window patch figures.")
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--full-json", required=True)
    parser.add_argument("--wo-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    selection = _load_json(Path(args.selection_json))
    full = _load_json(Path(args.full_json))
    wo = _load_json(Path(args.wo_json))

    full_lookup = {(c["domain"], c["node_idx"], c["sample_dataset_index"]): c for c in full["cases"] if c["prediction"] is not None}
    wo_lookup = {(c["domain"], c["node_idx"], c["sample_dataset_index"]): c for c in wo["cases"] if c["prediction"] is not None}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "selection_json": args.selection_json,
        "full_json": args.full_json,
        "wo_json": args.wo_json,
        "domains": {},
    }

    for domain in ("CBD", "Residential"):
        single_cases = []
        for row in selection["single_windows"][domain]:
            key = (domain, row["node_idx"], row["sample_dataset_index"])
            full_case = full_lookup[key]
            wo_case = wo_lookup[key]
            merged = {
                "domain": domain,
                "node_idx": row["node_idx"],
                "sample_dataset_index": row["sample_dataset_index"],
                "t_start": full_case["t_start"],
                "history": full_case["history"],
                "target": full_case["target"],
                "full_prediction": full_case["prediction"],
                "wo_prediction": wo_case["prediction"],
            }
            merged["full_metrics"] = _case_metrics(merged["full_prediction"], merged["target"])
            merged["wo_metrics"] = _case_metrics(merged["wo_prediction"], merged["target"])
            single_cases.append(merged)

        collage_map = {}
        for node_idx in selection["collage_windows"][domain]["node_indices"]:
            cases = []
            for sample_idx in selection["collage_windows"][domain]["window_indices"]:
                key = (domain, node_idx, sample_idx)
                full_case = full_lookup[key]
                wo_case = wo_lookup[key]
                merged = {
                    "domain": domain,
                    "node_idx": node_idx,
                    "sample_dataset_index": sample_idx,
                    "t_start": full_case["t_start"],
                    "history": full_case["history"],
                    "target": full_case["target"],
                    "full_prediction": full_case["prediction"],
                    "wo_prediction": wo_case["prediction"],
                }
                merged["full_metrics"] = _case_metrics(merged["full_prediction"], merged["target"])
                merged["wo_metrics"] = _case_metrics(merged["wo_prediction"], merged["target"])
                cases.append(merged)
            collage_map[node_idx] = cases

        single_path = out_dir / f"{domain.lower()}_scheme_a.svg"
        collage_path = out_dir / f"{domain.lower()}_scheme_b.svg"
        _plot_single_windows(domain, single_cases, single_path)
        _plot_collage(domain, collage_map, collage_path)

        summary["domains"][domain] = {
            "scheme_a_figure": str(single_path),
            "scheme_b_figure": str(collage_path),
            "single_windows": [
                {
                    "node_idx": case["node_idx"],
                    "sample_dataset_index": case["sample_dataset_index"],
                    "t_start": case["t_start"],
                    "full_metrics": case["full_metrics"],
                    "wo_metrics": case["wo_metrics"],
                }
                for case in single_cases
            ],
            "collage_windows": {
                str(node_idx): [
                    {
                        "sample_dataset_index": case["sample_dataset_index"],
                        "t_start": case["t_start"],
                        "full_metrics": case["full_metrics"],
                        "wo_metrics": case["wo_metrics"],
                    }
                    for case in cases
                ]
                for node_idx, cases in collage_map.items()
            },
        }

    summary_path = out_dir / "window_patch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
