from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError as exc:  # pragma: no cover
    plt = None
    HAS_MATPLOTLIB = False


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pair_records(results: dict) -> list[dict]:
    full_records = results.get("full", {}).get("records", [])
    wo_records = results.get("wo_diffdora", {}).get("records", [])
    wo_lookup = {
        (rec["sample_dataset_index"], rec["node_idx"]): rec
        for rec in wo_records
        if rec.get("parse_ok")
    }

    paired = []
    for rec in full_records:
        if not rec.get("parse_ok"):
            continue
        key = (rec["sample_dataset_index"], rec["node_idx"])
        other = wo_lookup.get(key)
        if other is None:
            continue
        target = np.array(rec["target"], dtype=np.float32)
        full_pred = np.array(rec["parsed_prediction"], dtype=np.float32)
        wo_pred = np.array(other["parsed_prediction"], dtype=np.float32)
        paired.append({
            "sample_dataset_index": rec["sample_dataset_index"],
            "sample_index": rec["sample_index"],
            "t_start": rec["t_start"],
            "node_idx": rec["node_idx"],
            "domain": rec.get("domain"),
            "target": target,
            "full_pred": full_pred,
            "wo_pred": wo_pred,
            "variant_gap_mae": float(np.mean(np.abs(full_pred - wo_pred))),
            "full_mae": float(np.mean(np.abs(full_pred - target))),
            "wo_mae": float(np.mean(np.abs(wo_pred - target))),
            "full_advantage": float(np.mean(np.abs(wo_pred - target)) - np.mean(np.abs(full_pred - target))),
        })
    return paired


def _select_cases(paired: list[dict], mode: str, top_k: int) -> list[dict]:
    if mode == "largest_variant_gap":
        key_fn = lambda row: (row["variant_gap_mae"], row["full_advantage"])
    elif mode == "largest_full_advantage":
        key_fn = lambda row: (row["full_advantage"], row["variant_gap_mae"])
    elif mode == "worst_full":
        key_fn = lambda row: (row["full_mae"], row["variant_gap_mae"])
    else:
        raise ValueError(f"Unsupported mode={mode!r}")
    return sorted(paired, key=key_fn, reverse=True)[:top_k]


def _plot_cases_matplotlib(cases: list[dict], output_path: Path, title: str) -> None:
    if not cases:
        raise ValueError("No matched cases available to plot.")

    rows = len(cases)
    fig, axes = plt.subplots(rows, 1, figsize=(10.5, 3.2 * rows), dpi=180, squeeze=False)
    x = np.arange(1, len(cases[0]["target"]) + 1)

    for ax, case in zip(axes[:, 0], cases):
        ax.plot(x, case["target"], marker="o", linewidth=2.2, color="#111827", label="Ground Truth")
        ax.plot(x, case["full_pred"], marker="s", linewidth=2.0, color="#2d6a4f", label="Full")
        ax.plot(x, case["wo_pred"], marker="^", linewidth=2.0, color="#1d4ed8", label="w/o DiffDoRA")
        ax.set_xticks(x)
        ax.set_xlabel("Forecast Step")
        ax.set_ylabel("Occupancy")
        ax.grid(alpha=0.25)
        ax.set_title(
            f"sample={case['sample_dataset_index']} t={case['t_start']} "
            f"node={case['node_idx']} {case['domain']} | "
            f"full_mae={case['full_mae']:.4f} wo_mae={case['wo_mae']:.4f}"
        )
        ax.legend(frameon=False, ncol=3, loc="upper left")

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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


def _plot_cases_svg(cases: list[dict], output_path: Path, title: str) -> None:
    if not cases:
        raise ValueError("No matched cases available to plot.")

    width = 1080
    row_h = 250
    height = 80 + row_h * len(cases)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        _svg_text(width / 2, 28, title, size=18, anchor="middle", weight="bold"),
    ]

    colors = {
        "target": "#111827",
        "full": "#2d6a4f",
        "wo": "#1d4ed8",
    }

    for idx, case in enumerate(cases):
        top = 60 + idx * row_h
        left = 60
        plot_w = 920
        plot_h = 140

        values = np.concatenate([case["target"], case["full_pred"], case["wo_pred"]]).astype(np.float32)
        y_min = float(values.min())
        y_max = float(values.max())
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0
        y_pad = 0.08 * (y_max - y_min)
        y_min -= y_pad
        y_max += y_pad

        parts.append(_svg_text(
            left,
            top - 14,
            (
                f"sample={case['sample_dataset_index']} t={case['t_start']} "
                f"node={case['node_idx']} {case['domain']} | "
                f"full_mae={case['full_mae']:.4f} wo_mae={case['wo_mae']:.4f}"
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
            parts.append(_svg_text(left - 8, y + 4, f"{v:.2f}", size=10, anchor="end", fill="#4b5563"))

        steps = len(case["target"])
        xs = []
        for step in range(steps):
            x = left + (step / max(steps - 1, 1)) * plot_w if steps > 1 else left + plot_w / 2
            xs.append(x)
            parts.append(_svg_line(x, top, x, top + plot_h, stroke="#f3f4f6", stroke_width=1))
            parts.append(_svg_text(x, top + plot_h + 18, str(step + 1), size=10, anchor="middle", fill="#4b5563"))

        def scale_y(v: float) -> float:
            return top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

        target_pts = [(x, scale_y(float(v))) for x, v in zip(xs, case["target"])]
        full_pts = [(x, scale_y(float(v))) for x, v in zip(xs, case["full_pred"])]
        wo_pts = [(x, scale_y(float(v))) for x, v in zip(xs, case["wo_pred"])]

        parts.append(_svg_polyline(target_pts, stroke=colors["target"], stroke_width=2.6))
        parts.append(_svg_polyline(full_pts, stroke=colors["full"], stroke_width=2.3))
        parts.append(_svg_polyline(wo_pts, stroke=colors["wo"], stroke_width=2.3))

        legend_y = top + plot_h + 40
        legend_x = left
        for label, color in [("Ground Truth", colors["target"]), ("Full", colors["full"]), ("w/o DiffDoRA", colors["wo"])]:
            parts.append(_svg_line(legend_x, legend_y - 4, legend_x + 18, legend_y - 4, stroke=color, stroke_width=3))
            parts.append(_svg_text(legend_x + 24, legend_y, label, size=11))
            legend_x += 170

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize full vs w/o DiffDoRA prediction waves against ground truth."
    )
    parser.add_argument("--ablation-json", required=True,
                        help="Path to eval_paper_ablation JSON containing full and wo_diffdora results.")
    parser.add_argument("--output-dir", default="outputs/prediction_waves")
    parser.add_argument("--selection-mode",
                        choices=["largest_variant_gap", "largest_full_advantage", "worst_full"],
                        default="largest_full_advantage")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    ablation_path = Path(args.ablation_json)
    data = _load_json(ablation_path)
    paired = _pair_records(data.get("results", {}))
    selected = _select_cases(paired, args.selection_mode, args.top_k)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / (
        f"prediction_waves_{args.selection_mode}.png"
        if HAS_MATPLOTLIB else
        f"prediction_waves_{args.selection_mode}.svg"
    )
    title = f"Prediction Waves | {args.selection_mode} | top_k={args.top_k}"
    if HAS_MATPLOTLIB:
        _plot_cases_matplotlib(selected, fig_path, title)
    else:
        _plot_cases_svg(selected, fig_path, title)

    summary = {
        "input": str(ablation_path),
        "selection_mode": args.selection_mode,
        "top_k": args.top_k,
        "figure": str(fig_path),
        "cases": [
            {
                "sample_dataset_index": case["sample_dataset_index"],
                "sample_index": case["sample_index"],
                "t_start": case["t_start"],
                "node_idx": case["node_idx"],
                "domain": case["domain"],
                "variant_gap_mae": case["variant_gap_mae"],
                "full_mae": case["full_mae"],
                "wo_mae": case["wo_mae"],
                "full_advantage": case["full_advantage"],
            }
            for case in selected
        ],
    }
    summary_path = out_dir / f"prediction_waves_{args.selection_mode}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
