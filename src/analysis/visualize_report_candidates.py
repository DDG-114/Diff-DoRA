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


def _segment_metrics(pred, target):
    pred_arr = np.array(pred, dtype=np.float32)
    true_arr = np.array(target, dtype=np.float32)
    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    return {"mae": mae, "rmse": rmse}


def _plot_clip_svg(candidate: dict, output_path: Path) -> None:
    width = 1200
    height = 330
    left = 70
    top = 70
    plot_w = 1080
    plot_h = 150

    truth = candidate["truth"]
    full_pred = candidate["full_prediction"]
    wo_pred = candidate["wo_prediction"]
    timestamps = candidate["timestamps"]
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
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        _svg_text(
            width / 2,
            28,
            f"{candidate['label']} | {candidate['domain']} node={candidate['node_idx']}",
            size=18,
            anchor="middle",
            weight="bold",
        ),
        _svg_text(
            width / 2,
            50,
            (
                f"{candidate['start_ts']} -> {candidate['end_ts']} | "
                f"Full MAE={candidate['full_metrics']['mae']:.4f}, "
                f"w/o DiffDoRA MAE={candidate['wo_metrics']['mae']:.4f}, "
                f"adv={candidate['full_advantage']:+.4f}"
            ),
            size=12,
            anchor="middle",
        ),
        _svg_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#374151", stroke_width=1.2),
        _svg_line(left, top, left, top + plot_h, stroke="#374151", stroke_width=1.2),
    ]

    for tick in range(5):
        frac = tick / 4
        y = top + plot_h - frac * plot_h
        v = y_min + frac * (y_max - y_min)
        parts.append(_svg_line(left, y, left + plot_w, y, stroke="#e5e7eb", stroke_width=1))
        parts.append(_svg_text(left - 8, y + 4, f"{v:.1f}", size=10, anchor="end", fill="#4b5563"))

    for i, x in enumerate(xs):
        if i == 0 or i == len(xs) - 1 or i % 3 == 0:
            parts.append(_svg_text(x, top + plot_h + 18, timestamps[i].replace("T", " ")[11:16], size=10, anchor="middle", fill="#4b5563"))

    for label, color, vals in [
        ("Ground Truth", "#111827", truth),
        ("Full", "#2d6a4f", full_pred),
        ("w/o DiffDoRA", "#1d4ed8", wo_pred),
    ]:
        pts = [(x, scale_y(v)) for x, v in zip(xs, vals)]
        parts.append(_svg_polyline(pts, stroke=color, stroke_width=2.6 if color == "#111827" else 2.2))

    legend_y = top + plot_h + 42
    legend_x = left
    for label, color in [("Ground Truth", "#111827"), ("Full", "#2d6a4f"), ("w/o DiffDoRA", "#1d4ed8")]:
        parts.append(_svg_line(legend_x, legend_y - 4, legend_x + 18, legend_y - 4, stroke=color, stroke_width=3))
        parts.append(_svg_text(legend_x + 24, legend_y, label, size=11))
        legend_x += 180

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _plot_gallery_svg(candidates: list[dict], output_path: Path) -> None:
    width = 1500
    row_h = 160
    height = 60 + row_h * len(candidates)
    left = 60
    plot_w = 760
    plot_h = 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        _svg_text(width / 2, 26, "Report Candidate Gallery | Full better than w/o DiffDoRA", size=18, anchor="middle", weight="bold"),
    ]
    for row, candidate in enumerate(candidates):
        top = 50 + row * row_h
        truth = candidate["truth"]
        full_pred = candidate["full_prediction"]
        wo_pred = candidate["wo_prediction"]
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
        parts.append(_svg_text(left, top - 8, f"{candidate['label']} | {candidate['domain']} node={candidate['node_idx']}", size=12, weight="bold"))
        parts.append(_svg_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#374151", stroke_width=1))
        parts.append(_svg_line(left, top, left, top + plot_h, stroke="#374151", stroke_width=1))
        for label, color, vals in [("GT", "#111827", truth), ("F", "#2d6a4f", full_pred), ("WO", "#1d4ed8", wo_pred)]:
            pts = [(x, scale_y(v)) for x, v in zip(xs, vals)]
            parts.append(_svg_polyline(pts, stroke=color, stroke_width=2.0 if color == "#111827" else 1.8))
        parts.append(_svg_text(left + plot_w + 25, top + 18, candidate["start_ts"], size=11))
        parts.append(_svg_text(left + plot_w + 25, top + 38, candidate["end_ts"], size=11))
        parts.append(_svg_text(left + plot_w + 25, top + 58, f"Full MAE={candidate['full_metrics']['mae']:.3f}", size=11))
        parts.append(_svg_text(left + plot_w + 25, top + 78, f"WO MAE={candidate['wo_metrics']['mae']:.3f}", size=11))
        parts.append(_svg_text(left + plot_w + 25, top + 98, f"Adv={candidate['full_advantage']:+.3f}", size=11, weight="bold"))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate labeled candidate clips where full outperforms w/o DiffDoRA.")
    parser.add_argument("--full-json", required=True)
    parser.add_argument("--wo-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    full = _load_json(Path(args.full_json))
    wo = _load_json(Path(args.wo_json))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    win = 12
    for domain in ("CBD", "Residential"):
        for node_idx, full_node in full["domains"][domain].items():
            wo_node = wo["domains"][domain][node_idx]
            truth = full_node["truth"]
            full_pred = full_node["prediction"]
            wo_pred = wo_node["prediction"]
            timestamps = full_node["timestamps"]
            for s in range(0, len(truth) - win + 1):
                if any(v is None for v in full_pred[s:s+win]) or any(v is None for v in wo_pred[s:s+win]):
                    continue
                full_metrics = _segment_metrics(full_pred[s:s+win], truth[s:s+win])
                wo_metrics = _segment_metrics(wo_pred[s:s+win], truth[s:s+win])
                adv = wo_metrics["mae"] - full_metrics["mae"]
                if adv <= 0:
                    continue
                candidates.append({
                    "domain": domain,
                    "node_idx": int(node_idx),
                    "start_idx": s,
                    "end_idx": s + win - 1,
                    "start_ts": timestamps[s],
                    "end_ts": timestamps[s + win - 1],
                    "timestamps": timestamps[s:s+win],
                    "truth": truth[s:s+win],
                    "full_prediction": full_pred[s:s+win],
                    "wo_prediction": wo_pred[s:s+win],
                    "full_metrics": full_metrics,
                    "wo_metrics": wo_metrics,
                    "full_advantage": adv,
                })

    selected = []
    used = []
    for cand in sorted(candidates, key=lambda row: row["full_advantage"], reverse=True):
        overlap = any(
            cand["domain"] == prev["domain"]
            and cand["node_idx"] == prev["node_idx"]
            and not (cand["end_idx"] < prev["start_idx"] or cand["start_idx"] > prev["end_idx"])
            for prev in used
        )
        if overlap:
            continue
        used.append(cand)
        selected.append(cand)
        if len(selected) >= 12:
            break

    # Add the user-mentioned reference clip even if it is not full-better.
    ref_node = "69"
    ref_domain = "Residential"
    ref_start = "2022-07-13 14:45:00"
    ref_end = "2022-07-13 15:40:00"
    timestamps = full["domains"][ref_domain][ref_node]["timestamps"]
    si = timestamps.index(ref_start)
    ei = timestamps.index(ref_end)
    ref_truth = full["domains"][ref_domain][ref_node]["truth"][si:ei+1]
    ref_full = full["domains"][ref_domain][ref_node]["prediction"][si:ei+1]
    ref_wo = wo["domains"][ref_domain][ref_node]["prediction"][si:ei+1]
    ref_full_metrics = _segment_metrics(ref_full, ref_truth)
    ref_wo_metrics = _segment_metrics(ref_wo, ref_truth)
    selected.append({
        "domain": ref_domain,
        "node_idx": int(ref_node),
        "start_idx": si,
        "end_idx": ei,
        "start_ts": ref_start,
        "end_ts": ref_end,
        "timestamps": timestamps[si:ei+1],
        "truth": ref_truth,
        "full_prediction": ref_full,
        "wo_prediction": ref_wo,
        "full_metrics": ref_full_metrics,
        "wo_metrics": ref_wo_metrics,
        "full_advantage": ref_wo_metrics["mae"] - ref_full_metrics["mae"],
        "manual_reference": True,
    })

    labeled = []
    for idx, cand in enumerate(selected, 1):
        label = f"C{idx:02d}" if not cand.get("manual_reference") else "REF01"
        labeled.append({**cand, "label": label})
        fig_path = out_dir / f"{label}_{cand['domain'].lower()}_node{cand['node_idx']}_{cand['start_ts'][11:16].replace(':','-')}.svg"
        _plot_clip_svg({**cand, "label": label}, fig_path)

    gallery_path = out_dir / "candidate_gallery.svg"
    _plot_gallery_svg(labeled, gallery_path)

    summary = {
        "gallery": str(gallery_path),
        "candidates": [
            {
                "label": cand["label"],
                "domain": cand["domain"],
                "node_idx": cand["node_idx"],
                "start_ts": cand["start_ts"],
                "end_ts": cand["end_ts"],
                "full_metrics": cand["full_metrics"],
                "wo_metrics": cand["wo_metrics"],
                "full_advantage": cand["full_advantage"],
                "manual_reference": cand.get("manual_reference", False),
            }
            for cand in labeled
        ],
    }
    summary_path = out_dir / "report_candidate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved gallery: {gallery_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
