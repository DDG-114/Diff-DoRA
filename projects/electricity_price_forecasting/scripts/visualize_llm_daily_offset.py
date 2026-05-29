#!/usr/bin/env python3
"""Render an HTML/SVG report for candidate baseline + LLM daily offset runs."""
from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from statistics import mean


PRICE_FLOOR = 40.0
PRICE_CAP = 1000.0


def _parse_dt(value: str) -> datetime:
    value = value.strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(value.replace("/", "-"))


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_truth(source_csv: Path) -> dict[str, list[dict[str, float]]]:
    days: dict[str, list[dict[str, float]]] = {}
    with source_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Date", "").strip():
                continue
            dt = _parse_dt(row["Date"])
            price = _safe_float(row["Price"])
            if price is None:
                continue
            day = dt.strftime("%Y-%m-%d")
            slot = dt.hour * 4 + dt.minute // 15
            days.setdefault(day, []).append({"slot": float(slot), "price": price})
    for rows in days.values():
        rows.sort(key=lambda item: item["slot"])
    return days


def _load_candidate(candidate_csv: Path) -> dict[tuple[str, int], float]:
    candidate: dict[tuple[str, int], float] = {}
    with candidate_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidate[(row["day"], int(row["slot"]))] = float(row["prediction"]) * 1000.0
    return candidate


def _clip_price(value: float) -> float:
    return min(max(value, PRICE_FLOOR), PRICE_CAP)


def _daily_mean_accuracy(pred: list[float], true: list[float]) -> float:
    pred_mean = mean(pred)
    true_mean = mean(true)
    return max(0.0, 1.0 - abs(pred_mean - true_mean) / max(abs(true_mean), PRICE_FLOOR))


def _relative_accuracy(pred: list[float], true: list[float]) -> float:
    errors = [abs(p - t) / max(abs(t), PRICE_FLOOR) for p, t in zip(pred, true)]
    return max(0.0, 1.0 - mean(errors))


def _mae(pred: list[float], true: list[float]) -> float:
    return mean(abs(p - t) for p, t in zip(pred, true))


def _rmse(pred: list[float], true: list[float]) -> float:
    return (mean((p - t) ** 2 for p, t in zip(pred, true))) ** 0.5


def _points(values: list[float], width: int, height: int, y_min: float, y_max: float) -> str:
    if len(values) == 1:
        xs = [width / 2.0]
    else:
        xs = [idx * width / (len(values) - 1) for idx in range(len(values))]
    span = max(y_max - y_min, 1e-6)
    coords = []
    for x, value in zip(xs, values):
        y = height - ((value - y_min) / span) * height
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def _curve_svg(day: str, truth: list[float], candidate: list[float], pred: list[float]) -> str:
    width = 900
    height = 260
    pad_left = 54
    pad_right = 16
    pad_top = 18
    pad_bottom = 34
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    y_values = truth + candidate + pred
    y_min = max(0.0, min(y_values) - 40.0)
    y_max = min(PRICE_CAP, max(y_values) + 40.0)
    if y_max - y_min < 160:
        mid = (y_min + y_max) / 2.0
        y_min = max(0.0, mid - 80)
        y_max = min(PRICE_CAP, mid + 80)
    y_ticks = [y_min, (y_min + y_max) / 2.0, y_max]
    x_ticks = [(0, "00:00"), (24, "06:00"), (48, "12:00"), (72, "18:00"), (95, "24:00")]

    def sx(slot: int) -> float:
        return pad_left + slot * plot_w / 95.0

    def sy(value: float) -> float:
        return pad_top + plot_h - ((value - y_min) / max(y_max - y_min, 1e-6)) * plot_h

    tick_parts = []
    for value in y_ticks:
        y = sy(value)
        tick_parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.0f}</text>'
        )
    for slot, label in x_ticks:
        x = sx(slot)
        tick_parts.append(
            f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{height - pad_bottom}" class="grid grid-x" />'
            f'<text x="{x:.1f}" y="{height - 10}" text-anchor="middle" class="axis">{label}</text>'
        )

    truth_pts = _points(truth, plot_w, plot_h, y_min, y_max)
    candidate_pts = _points(candidate, plot_w, plot_h, y_min, y_max)
    pred_pts = _points(pred, plot_w, plot_h, y_min, y_max)
    return f"""
    <svg viewBox="0 0 {width} {height}" class="curve" role="img" aria-label="{html.escape(day)} price curve">
      <g transform="translate({pad_left},{pad_top})">
        <polyline points="{candidate_pts}" class="line candidate" />
        <polyline points="{truth_pts}" class="line truth" />
        <polyline points="{pred_pts}" class="line llm" />
      </g>
      {''.join(tick_parts)}
      <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" class="axis-line" />
      <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line" />
    </svg>
    """


def _bar_svg(rows: list[dict[str, float]]) -> str:
    width = 900
    height = 230
    pad_left = 52
    pad_right = 18
    pad_top = 16
    pad_bottom = 50
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    n = len(rows)
    group = plot_w / max(n, 1)
    bar_w = min(28.0, group * 0.28)

    def sy(value: float) -> float:
        return pad_top + plot_h - value * plot_h

    parts = [
        f'<line x1="{pad_left}" y1="{sy(0.8):.1f}" x2="{width - pad_right}" y2="{sy(0.8):.1f}" class="threshold" />',
        f'<text x="{width - pad_right}" y="{sy(0.8) - 6:.1f}" text-anchor="end" class="threshold-label">0.80 threshold</text>',
    ]
    for tick in (0.0, 0.5, 0.8, 1.0):
        y = sy(tick)
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>'
        )
    for idx, row in enumerate(rows):
        x0 = pad_left + idx * group + group * 0.25
        day_acc = row["daily_mean_accuracy"]
        rel_acc = row["relative_accuracy"]
        y1 = sy(day_acc)
        y2 = sy(rel_acc)
        parts.append(
            f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{pad_top + plot_h - y1:.1f}" class="bar daily" />'
            f'<rect x="{x0 + bar_w + 3:.1f}" y="{y2:.1f}" width="{bar_w:.1f}" height="{pad_top + plot_h - y2:.1f}" class="bar relative" />'
        )
        if idx % 2 == 0:
            label = html.escape(str(row["day"])[5:])
            parts.append(
                f'<text x="{x0 + bar_w:.1f}" y="{height - 16}" text-anchor="middle" class="axis day-label">{label}</text>'
            )
    return f'<svg viewBox="0 0 {width} {height}" class="bar-chart" role="img">{"".join(parts)}</svg>'


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_report(args: argparse.Namespace) -> dict[str, object]:
    source_csv = Path(args.source_csv)
    candidate_csv = Path(args.candidate_csv)
    eval_json = Path(args.eval_json)
    truth_by_day = _load_truth(source_csv)
    candidate_map = _load_candidate(candidate_csv)
    eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))

    rows = []
    for record in eval_payload["records"]:
        day = record["day"]
        truth_rows = truth_by_day[day]
        truth = [item["price"] for item in truth_rows]
        slots = [int(item["slot"]) for item in truth_rows]
        candidate = [candidate_map[(day, slot)] for slot in slots]
        offset_yuan = float(record["offset"]) * 1000.0
        pred = [_clip_price(value + offset_yuan) for value in candidate]
        rows.append(
            {
                "day": day,
                "offset": float(record["offset"]),
                "offset_yuan": offset_yuan,
                "mae": _mae(pred, truth),
                "rmse": _rmse(pred, truth),
                "daily_mean_accuracy": _daily_mean_accuracy(pred, truth),
                "relative_accuracy": _relative_accuracy(pred, truth),
                "truth_mean": mean(truth),
                "candidate_mean": mean(candidate),
                "pred_mean": mean(pred),
                "truth": truth,
                "candidate": candidate,
                "pred": pred,
            }
        )
    return {"eval": eval_payload, "rows": rows}


def render_html(report: dict[str, object]) -> str:
    eval_payload = report["eval"]
    rows = report["rows"]
    metrics = eval_payload["metrics"]
    low_days = [row for row in rows if row["daily_mean_accuracy"] < 0.7]
    cards = [
        ("Mean daily mean accuracy", _fmt(metrics["mean_daily_mean_accuracy"], 3)),
        ("Days >= 0.8", _fmt(metrics["share_days_daily_mean_accuracy_ge_0_8"], 3)),
        ("Mean relative accuracy", _fmt(metrics["mean_relative_accuracy"], 3)),
        ("Mean day MAE", f'{metrics["mean_day_mae"]:.1f}'),
    ]
    card_html = "".join(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(value)}</div></div>'
        for label, value in cards
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['day'])}</td>"
        f"<td>{row['offset_yuan']:.1f}</td>"
        f"<td>{row['truth_mean']:.1f}</td>"
        f"<td>{row['candidate_mean']:.1f}</td>"
        f"<td>{row['pred_mean']:.1f}</td>"
        f"<td class=\"{'bad' if row['daily_mean_accuracy'] < 0.7 else ''}\">{row['daily_mean_accuracy']:.3f}</td>"
        f"<td>{row['relative_accuracy']:.3f}</td>"
        f"<td>{row['mae']:.1f}</td>"
        "</tr>"
        for row in rows
    )
    day_sections = "".join(
        f"""
        <section class="day-card">
          <div class="day-head">
            <div>
              <h2>{html.escape(row['day'])}</h2>
              <p>truth mean {row['truth_mean']:.1f}, candidate mean {row['candidate_mean']:.1f}, LLM mean {row['pred_mean']:.1f}</p>
            </div>
            <div class="day-metrics">
              <span>offset {row['offset_yuan']:.1f}</span>
              <span>daily {row['daily_mean_accuracy']:.3f}</span>
              <span>relative {row['relative_accuracy']:.3f}</span>
              <span>MAE {row['mae']:.1f}</span>
            </div>
          </div>
          {_curve_svg(row['day'], row['truth'], row['candidate'], row['pred'])}
        </section>
        """
        for row in rows
    )
    low_text = ", ".join(f"{row['day']} ({row['daily_mean_accuracy']:.3f})" for row in low_days) or "None"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Daily Offset Visualization</title>
  <style>
    :root {{
      --bg: #f3f1ea;
      --ink: #161a1d;
      --muted: #66706b;
      --panel: #fffdf8;
      --line: #d7d1c5;
      --truth: #151a1e;
      --candidate: #9aa3a3;
      --llm: #d95f21;
      --relative: #287b75;
      --bad: #b3261e;
      --threshold: #2d8a56;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f7f5ee 0%, var(--bg) 44%, #ebe6da 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 34px 22px 60px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0;
      font-size: 21px;
    }}
    p {{
      color: var(--muted);
      line-height: 1.5;
      margin: 8px 0 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .metric-card, .panel, .day-card {{
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 35px rgba(38, 32, 21, 0.08);
    }}
    .metric-card {{
      padding: 16px;
    }}
    .metric-label {{
      font-size: 13px;
      color: var(--muted);
    }}
    .metric-value {{
      font-size: 29px;
      margin-top: 6px;
      font-weight: 700;
    }}
    .panel {{
      padding: 18px;
      margin-bottom: 18px;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .swatch {{
      display: inline-block;
      width: 28px;
      height: 4px;
      margin-right: 8px;
      vertical-align: middle;
      border-radius: 99px;
      background: currentColor;
    }}
    .legend .truth {{ color: var(--truth); }}
    .legend .candidate {{ color: var(--candidate); }}
    .legend .llm {{ color: var(--llm); }}
    .legend .relative {{ color: var(--relative); }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .line {{
      fill: none;
      stroke-width: 2.4;
      stroke-linejoin: round;
      stroke-linecap: round;
    }}
    .line.truth {{ stroke: var(--truth); }}
    .line.candidate {{
      stroke: var(--candidate);
      stroke-width: 2;
      stroke-dasharray: 7 7;
    }}
    .line.llm {{ stroke: var(--llm); }}
    .grid {{
      stroke: #ddd7cb;
      stroke-width: 1;
    }}
    .grid-x {{
      stroke-dasharray: 4 8;
    }}
    .axis, .threshold-label {{
      fill: var(--muted);
      font-size: 12px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }}
    .axis-line {{
      stroke: #bdb6aa;
      stroke-width: 1;
    }}
    .threshold {{
      stroke: var(--threshold);
      stroke-width: 1.4;
      stroke-dasharray: 8 6;
    }}
    .threshold-label {{
      fill: var(--threshold);
      font-weight: 700;
    }}
    .bar.daily {{ fill: var(--llm); }}
    .bar.relative {{ fill: var(--relative); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .bad {{
      color: var(--bad);
      font-weight: 800;
    }}
    .day-card {{
      padding: 18px;
      margin-top: 18px;
    }}
    .day-head {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 8px;
    }}
    .day-metrics {{
      display: flex;
      gap: 8px;
      align-items: start;
      justify-content: flex-end;
      flex-wrap: wrap;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 13px;
    }}
    .day-metrics span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: #fbf7ed;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 24px 14px 44px; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .day-head {{ display: block; }}
      .day-metrics {{ justify-content: flex-start; margin-top: 10px; }}
      table {{ font-size: 12px; }}
      th, td {{ padding: 7px 4px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Candidate Baseline + LLM Daily Offset</h1>
    <p>Evaluation source: <code>gs_price_2025_llm_daily_offset_smoke_eval_fixed_v2.json</code>. Daily mean accuracy only checks the 96-slot average price; relative accuracy reflects pointwise curve errors.</p>

    <div class="metrics">{card_html}</div>

    <section class="panel">
      <h2>Daily Accuracy Overview</h2>
      <p>Days below 0.70 daily mean accuracy: {html.escape(low_text)}.</p>
      <div class="legend">
        <span class="llm"><span class="swatch"></span>daily mean accuracy</span>
        <span class="relative"><span class="swatch"></span>relative pointwise accuracy</span>
      </div>
      {_bar_svg(rows)}
    </section>

    <section class="panel">
      <h2>Daily Metrics Table</h2>
      <table>
        <thead>
          <tr>
            <th>day</th>
            <th>offset yuan</th>
            <th>true mean</th>
            <th>candidate mean</th>
            <th>LLM mean</th>
            <th>daily acc</th>
            <th>relative acc</th>
            <th>MAE</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>

    <div class="legend">
      <span class="truth"><span class="swatch"></span>true</span>
      <span class="candidate"><span class="swatch"></span>candidate baseline</span>
      <span class="llm"><span class="swatch"></span>candidate + LLM offset</span>
    </div>
    {day_sections}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--eval_json", default="outputs/gs_price_2025_llm_daily_offset_smoke_eval_fixed_v2.json")
    parser.add_argument("--output", default="outputs/gs_price_2025_llm_daily_offset_viz.html")
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(report), encoding="utf-8")
    metrics = report["eval"]["metrics"]
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
