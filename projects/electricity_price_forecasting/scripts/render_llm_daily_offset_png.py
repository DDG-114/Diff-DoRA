#!/usr/bin/env python3
"""Render a dependency-free PNG summary for candidate + LLM daily offset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from datetime import datetime
from pathlib import Path
from statistics import mean


PRICE_FLOOR = 40.0
PRICE_CAP = 1000.0


PALETTE = {
    "bg": (248, 246, 239),
    "panel": (255, 253, 248),
    "grid": (218, 212, 202),
    "axis": (110, 111, 105),
    "ink": (24, 28, 31),
    "truth": (18, 25, 30),
    "candidate": (152, 161, 161),
    "llm": (217, 95, 33),
    "relative": (40, 123, 117),
    "green": (48, 135, 83),
    "bad": (179, 38, 30),
}


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


class Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(bg * (width * height))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 3
            self.pixels[idx : idx + 3] = bytes(color)

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(color) * (x1 - x0)
        for yy in range(y0, y1):
            idx = (yy * self.width + x0) * 3
            self.pixels[idx : idx + len(row)] = row

    def line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        x0_i, y0_i = int(round(x0)), int(round(y0))
        x1_i, y1_i = int(round(x1)), int(round(y1))
        dx = abs(x1_i - x0_i)
        dy = -abs(y1_i - y0_i)
        sx = 1 if x0_i < x1_i else -1
        sy = 1 if y0_i < y1_i else -1
        err = dx + dy
        x, y = x0_i, y0_i
        radius = max(width // 2, 0)
        while True:
            for yy in range(y - radius, y + radius + 1):
                for xx in range(x - radius, x + radius + 1):
                    self.set_pixel(xx, yy, color)
            if x == x1_i and y == y1_i:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def polyline(self, points: list[tuple[float, float]], color: tuple[int, int, int], width: int = 1) -> None:
        for p0, p1 in zip(points, points[1:]):
            self.line(p0[0], p0[1], p1[0], p1[1], color, width)

    def text(self, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor = x
        for raw_ch in text:
            ch = raw_ch.upper()
            bitmap = FONT_5X7.get(ch, FONT_5X7[" "])
            for yy, row in enumerate(bitmap):
                for xx, value in enumerate(row):
                    if value == "1":
                        self.rect(cursor + xx * scale, y + yy * scale, scale, scale, color)
            cursor += 6 * scale

    def save_png(self, path: Path) -> None:
        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

        rows = []
        stride = self.width * 3
        for y in range(self.height):
            rows.append(b"\x00" + bytes(self.pixels[y * stride : (y + 1) * stride]))
        raw = b"".join(rows)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(png)


def parse_dt(value: str) -> datetime:
    value = value.strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(value.replace("/", "-"))


def load_truth(path: Path) -> dict[str, list[dict[str, float]]]:
    days: dict[str, list[dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Date", "").strip():
                continue
            price_raw = row.get("Price", "")
            try:
                price = float(price_raw)
            except ValueError:
                continue
            dt = parse_dt(row["Date"])
            day = dt.strftime("%Y-%m-%d")
            slot = dt.hour * 4 + dt.minute // 15
            days.setdefault(day, []).append({"slot": float(slot), "price": price})
    for rows in days.values():
        rows.sort(key=lambda item: item["slot"])
    return days


def load_candidate(path: Path) -> dict[tuple[str, int], float]:
    rows: dict[tuple[str, int], float] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["day"], int(row["slot"]))] = float(row["prediction"]) * 1000.0
    return rows


def clip_price(value: float) -> float:
    return min(max(value, PRICE_FLOOR), PRICE_CAP)


def daily_mean_accuracy(pred: list[float], truth: list[float]) -> float:
    pred_mean = mean(pred)
    true_mean = mean(truth)
    return max(0.0, 1.0 - abs(pred_mean - true_mean) / max(abs(true_mean), PRICE_FLOOR))


def relative_accuracy(pred: list[float], truth: list[float]) -> float:
    return max(0.0, 1.0 - mean(abs(p - t) / max(abs(t), PRICE_FLOOR) for p, t in zip(pred, truth)))


def mae(pred: list[float], truth: list[float]) -> float:
    return mean(abs(p - t) for p, t in zip(pred, truth))


def build_rows(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    eval_payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    truth_by_day = load_truth(Path(args.source_csv))
    candidate_map = load_candidate(Path(args.candidate_csv))
    rows = []
    for record in eval_payload["records"]:
        day = record["day"]
        truth_rows = truth_by_day[day]
        truth = [item["price"] for item in truth_rows]
        slots = [int(item["slot"]) for item in truth_rows]
        candidate = [candidate_map[(day, slot)] for slot in slots]
        offset_yuan = float(record["offset"]) * 1000.0
        pred = [clip_price(value + offset_yuan) for value in candidate]
        rows.append(
            {
                "day": day,
                "truth": truth,
                "candidate": candidate,
                "pred": pred,
                "offset_yuan": offset_yuan,
                "truth_mean": mean(truth),
                "candidate_mean": mean(candidate),
                "pred_mean": mean(pred),
                "daily_mean_accuracy": daily_mean_accuracy(pred, truth),
                "relative_accuracy": relative_accuracy(pred, truth),
                "mae": mae(pred, truth),
            }
        )
    return eval_payload, rows


def draw_card(canvas: Canvas, x: int, y: int, w: int, h: int, label: str, value: str) -> None:
    canvas.rect(x, y, w, h, PALETTE["panel"])
    canvas.rect(x, y, w, 2, PALETTE["grid"])
    canvas.rect(x, y + h - 2, w, 2, PALETTE["grid"])
    canvas.rect(x, y, 2, h, PALETTE["grid"])
    canvas.rect(x + w - 2, y, 2, h, PALETTE["grid"])
    canvas.text(x + 16, y + 18, label, PALETTE["axis"], 2)
    canvas.text(x + 16, y + 50, value, PALETTE["ink"], 4)


def map_points(values: list[float], x: int, y: int, w: int, h: int, ymin: float, ymax: float) -> list[tuple[float, float]]:
    span = max(ymax - ymin, 1e-6)
    points = []
    for idx, value in enumerate(values):
        px = x + idx * w / max(len(values) - 1, 1)
        py = y + h - ((value - ymin) / span) * h
        points.append((px, py))
    return points


def draw_curve_panel(canvas: Canvas, x: int, y: int, w: int, h: int, row: dict) -> None:
    canvas.rect(x, y, w, h, PALETTE["panel"])
    title = f"{row['day']} daily {row['daily_mean_accuracy']:.3f} relative {row['relative_accuracy']:.3f} mae {row['mae']:.1f}"
    canvas.text(x + 12, y + 10, title, PALETTE["ink"], 2)
    plot_x, plot_y = x + 50, y + 40
    plot_w, plot_h = w - 64, h - 68
    values = row["truth"] + row["candidate"] + row["pred"]
    ymin = max(0.0, min(values) - 40.0)
    ymax = min(PRICE_CAP, max(values) + 40.0)
    if ymax - ymin < 120.0:
        mid = (ymin + ymax) / 2.0
        ymin = max(0.0, mid - 60.0)
        ymax = min(PRICE_CAP, mid + 60.0)
    for frac in (0.0, 0.5, 1.0):
        yy = plot_y + plot_h - frac * plot_h
        canvas.line(plot_x, yy, plot_x + plot_w, yy, PALETTE["grid"], 1)
        label = f"{ymin + frac * (ymax - ymin):.0f}"
        canvas.text(x + 10, int(yy) - 6, label, PALETTE["axis"], 1)
    for slot, label in ((0, "00"), (24, "06"), (48, "12"), (72, "18"), (95, "24")):
        xx = plot_x + slot * plot_w / 95.0
        canvas.line(xx, plot_y, xx, plot_y + plot_h, PALETTE["grid"], 1)
        canvas.text(int(xx) - 6, y + h - 20, label, PALETTE["axis"], 1)
    canvas.polyline(map_points(row["candidate"], plot_x, plot_y, plot_w, plot_h, ymin, ymax), PALETTE["candidate"], 2)
    canvas.polyline(map_points(row["truth"], plot_x, plot_y, plot_w, plot_h, ymin, ymax), PALETTE["truth"], 2)
    canvas.polyline(map_points(row["pred"], plot_x, plot_y, plot_w, plot_h, ymin, ymax), PALETTE["llm"], 3)


def draw_accuracy_bars(canvas: Canvas, x: int, y: int, w: int, h: int, rows: list[dict]) -> None:
    canvas.rect(x, y, w, h, PALETTE["panel"])
    canvas.text(x + 14, y + 12, "DAILY ACCURACY BY DAY", PALETTE["ink"], 2)
    plot_x, plot_y = x + 54, y + 46
    plot_w, plot_h = w - 76, h - 82
    for tick in (0.0, 0.5, 0.8, 1.0):
        yy = plot_y + plot_h - tick * plot_h
        color = PALETTE["green"] if abs(tick - 0.8) < 1e-6 else PALETTE["grid"]
        canvas.line(plot_x, yy, plot_x + plot_w, yy, color, 1)
        canvas.text(x + 18, int(yy) - 6, f"{tick:.1f}", PALETTE["axis"], 1)
    group_w = plot_w / len(rows)
    for idx, row in enumerate(rows):
        daily = row["daily_mean_accuracy"]
        rel = row["relative_accuracy"]
        bx = int(plot_x + idx * group_w + group_w * 0.22)
        bw = max(8, int(group_w * 0.22))
        dy = int(plot_y + plot_h - daily * plot_h)
        ry = int(plot_y + plot_h - rel * plot_h)
        canvas.rect(bx, dy, bw, int(plot_y + plot_h - dy), PALETTE["llm"] if daily >= 0.7 else PALETTE["bad"])
        canvas.rect(bx + bw + 5, ry, bw, int(plot_y + plot_h - ry), PALETTE["relative"])
        canvas.text(bx - 2, y + h - 24, row["day"][5:], PALETTE["axis"], 1)
    canvas.rect(x + w - 250, y + 14, 22, 8, PALETTE["llm"])
    canvas.text(x + w - 220, y + 9, "daily mean", PALETTE["axis"], 2)
    canvas.rect(x + w - 250, y + 42, 22, 8, PALETTE["relative"])
    canvas.text(x + w - 220, y + 37, "pointwise", PALETTE["axis"], 2)


def render_png(eval_payload: dict, rows: list[dict], output: Path) -> None:
    width, height = 1900, 2200
    canvas = Canvas(width, height, PALETTE["bg"])
    canvas.text(42, 34, "CANDIDATE BASELINE + LLM DAILY OFFSET", PALETTE["ink"], 4)
    canvas.text(44, 82, "TRUE VS CANDIDATE VS OFFSET PREDICTION, 96 SLOTS PER DAY", PALETTE["axis"], 2)
    metrics = eval_payload["metrics"]
    cards = [
        ("MEAN DAILY ACC", f"{metrics['mean_daily_mean_accuracy']:.3f}"),
        ("DAYS >= 0.8", f"{metrics['share_days_daily_mean_accuracy_ge_0_8']:.3f}"),
        ("MEAN POINTWISE", f"{metrics['mean_relative_accuracy']:.3f}"),
        ("MEAN MAE", f"{metrics['mean_day_mae']:.1f}"),
    ]
    for idx, (label, value) in enumerate(cards):
        draw_card(canvas, 42 + idx * 455, 126, 420, 118, label, value)
    canvas.rect(44, 270, 26, 8, PALETTE["truth"])
    canvas.text(82, 262, "true", PALETTE["axis"], 2)
    canvas.rect(170, 270, 26, 8, PALETTE["candidate"])
    canvas.text(208, 262, "candidate", PALETTE["axis"], 2)
    canvas.rect(360, 270, 26, 8, PALETTE["llm"])
    canvas.text(398, 262, "candidate + LLM offset", PALETTE["axis"], 2)
    low_days = [row for row in rows if row["daily_mean_accuracy"] < 0.7]
    low_text = "LOW DAYS < 0.70: " + ", ".join(f"{row['day'][5:]}={row['daily_mean_accuracy']:.3f}" for row in low_days)
    canvas.text(42, 306, low_text, PALETTE["bad"], 2)
    draw_accuracy_bars(canvas, 42, 340, 1816, 310, rows)
    panel_w, panel_h = 890, 260
    start_y = 680
    for idx, row in enumerate(rows):
        col = idx % 2
        line = idx // 2
        draw_curve_panel(canvas, 42 + col * 930, start_y + line * 290, panel_w, panel_h, row)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save_png(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--candidate_csv", default="outputs/gs_price_2025_llm_candidate_map.csv")
    parser.add_argument("--eval_json", default="outputs/gs_price_2025_llm_daily_offset_smoke_eval_fixed_v2.json")
    parser.add_argument("--output", default="outputs/gs_price_2025_llm_daily_offset_viz.png")
    args = parser.parse_args()
    eval_payload, rows = build_rows(args)
    render_png(eval_payload, rows, Path(args.output))
    print(json.dumps(eval_payload["metrics"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
