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
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


DEFAULT_RATIOS = ("0.20", "0.40", "0.60", "0.80")


def _load_ratio_result(path: Path) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not payload.get("results"):
        raise ValueError(f"No results found in {path}")
    ratio_key = list(payload["results"].keys())[0]
    return ratio_key, payload["results"][ratio_key]


def _collect_rows(run_dir: Path, ratios: tuple[str, ...]) -> list[dict]:
    rows = []
    for ratio in ratios:
        result_path = run_dir / f"ratio_{ratio}" / "zeroshot_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing zeroshot result: {result_path}")
        ratio_key, result = _load_ratio_result(result_path)
        metrics = result.get("metrics") or {}
        overall = metrics.get("overall") or {}
        rows.append(
            {
                "ratio_label": ratio,
                "ratio_key": ratio_key,
                "rmse": float(overall["rmse"]),
                "mae": float(overall["mae"]),
                "train_window_count": int(result.get("train_window_count", 0)),
                "test_window_count": int(result.get("test_window_count", 0)),
                "target_node_count": len(result.get("eval_target_nodes", [])),
            }
        )
    return rows


def _plot(rows: list[dict], output_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        _plot_svg(rows, output_path.with_suffix(".svg"))
        return

    labels = [row["ratio_label"] for row in rows]
    x = np.arange(len(labels))
    rmse = [row["rmse"] for row in rows]
    mae = [row["mae"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)
    fig.subplots_adjust(wspace=0.28, top=0.84, bottom=0.18)

    panels = [
        ("RMSE", rmse, "#5b8c5a"),
        ("MAE", mae, "#d17b49"),
    ]
    for ax, (title, values, color) in zip(axes, panels):
        bars = ax.bar(x, values, width=0.62, color=color, edgecolor="#2f2f2f", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
        ymin = 0.0
        ymax = max(values) * 1.18 if values else 1.0
        ax.set_ylim(ymin, ymax)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Zero-shot Evaluation Comparison (ST-EVCDP, h=6)", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.06, "Source Ratio", ha="center", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _svg_text(x, y, text, *, size=12, fill="#111827", anchor="middle", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-weight="{weight}">{escape(str(text))}</text>'
    )


def _svg_rect(x, y, w, h, fill, *, stroke="none", stroke_width=1, rx=0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}" ry="{rx}" />'
    )


def _svg_line(x1, y1, x2, y2, *, stroke="#9ca3af", stroke_width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{dash_attr} />'
    )


def _bar_chart_svg(title: str, labels: list[str], values: list[float], *, width=430, height=330, color="#5b8c5a") -> list[str]:
    margin_left, margin_right, margin_top, margin_bottom = 52, 18, 36, 56
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    step = plot_w / max(len(values), 1)
    bar_w = step * 0.58
    ymax = max(values) * 1.18 if values else 1.0
    ymax = ymax if ymax > 0 else 1.0

    parts = [
        _svg_text(width / 2, 20, title, size=15, weight="bold"),
        _svg_line(margin_left, margin_top, margin_left, margin_top + plot_h, stroke="#374151", stroke_width=1.2),
        _svg_line(margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h, stroke="#374151", stroke_width=1.2),
    ]
    for idx in range(5):
        frac = idx / 4
        y = margin_top + plot_h - frac * plot_h
        tick_val = frac * ymax
        parts.append(_svg_line(margin_left, y, margin_left + plot_w, y, stroke="#e5e7eb", stroke_width=1))
        parts.append(_svg_text(margin_left - 8, y + 4, f"{tick_val:.2f}", size=10, anchor="end", fill="#4b5563"))

    for idx, (label, value) in enumerate(zip(labels, values)):
        x_center = margin_left + idx * step + step / 2
        bar_h = (value / ymax) * plot_h if ymax else 0.0
        parts.append(
            _svg_rect(
                x_center - bar_w / 2,
                margin_top + plot_h - bar_h,
                bar_w,
                bar_h,
                color,
                stroke="#1f2937",
                stroke_width=0.8,
                rx=2,
            )
        )
        parts.append(_svg_text(x_center, margin_top + plot_h + 18, label, size=11))
        parts.append(_svg_text(x_center, margin_top + plot_h - bar_h - 8, f"{value:.3f}", size=10))
    return parts


def _plot_svg(rows: list[dict], output_path: Path) -> None:
    labels = [row["ratio_label"] for row in rows]
    rmse = [row["rmse"] for row in rows]
    mae = [row["mae"] for row in rows]

    width, height = 940, 400
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, "#ffffff"),
        _svg_text(width / 2, 24, "Zero-shot Evaluation Comparison (ST-EVCDP, h=6)", size=18, weight="bold"),
        '<g transform="translate(18,50)">',
    ]
    parts.extend(_bar_chart_svg("RMSE", labels, rmse, color="#5b8c5a"))
    parts.append("</g>")
    parts.append('<g transform="translate(488,50)">')
    parts.extend(_bar_chart_svg("MAE", labels, mae, color="#d17b49"))
    parts.append("</g>")
    parts.append(_svg_text(width / 2, height - 18, "Source Ratio", size=12))
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create zero-shot RMSE/MAE bar charts from a run directory.")
    parser.add_argument("--run-dir", required=True, help="Directory containing ratio_0.20/0.40/0.60/0.80 results.")
    parser.add_argument("--output-dir", default="", help="Directory for the figure and summary files.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    rows = _collect_rows(run_dir, DEFAULT_RATIOS)

    fig_path = output_dir / ("zeroshot_metrics_bar.png" if HAS_MATPLOTLIB else "zeroshot_metrics_bar.svg")
    _plot(rows, fig_path)

    summary_path = output_dir / "zeroshot_metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir), "rows": rows, "figure": str(fig_path)}, f, indent=2, ensure_ascii=False)

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
