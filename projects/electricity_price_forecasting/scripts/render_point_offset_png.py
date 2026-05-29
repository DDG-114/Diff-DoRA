#!/usr/bin/env python3
"""Render a PNG report for the point-offset calibrated 96-slot forecasts."""
from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


PALETTE = {
    "bg": (248, 246, 239),
    "panel": (255, 253, 248),
    "grid": (218, 212, 202),
    "axis": (110, 111, 105),
    "ink": (24, 28, 31),
    "truth": (18, 25, 30),
    "candidate": (152, 161, 161),
    "calibrated": (217, 95, 33),
    "offset": (40, 123, 117),
    "bad": (179, 38, 30),
    "green": (48, 135, 83),
}


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
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
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(color) * (x1 - x0)
        for yy in range(y0, y1):
            idx = (yy * self.width + x0) * 3
            self.pixels[idx : idx + len(row)] = row

    def line(self, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int], width: int = 1) -> None:
        x0_i, y0_i = int(round(x0)), int(round(y0))
        x1_i, y1_i = int(round(x1)), int(round(y1))
        dx, dy = abs(x1_i - x0_i), -abs(y1_i - y0_i)
        sx = 1 if x0_i < x1_i else -1
        sy = 1 if y0_i < y1_i else -1
        err = dx + dy
        radius = max(width // 2, 0)
        x, y = x0_i, y0_i
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
        for raw in text.upper():
            bitmap = FONT_5X7.get(raw, FONT_5X7[" "])
            for yy, row in enumerate(bitmap):
                for xx, val in enumerate(row):
                    if val == "1":
                        self.rect(cursor + xx * scale, y + yy * scale, scale, scale, color)
            cursor += 6 * scale

    def save_png(self, path: Path) -> None:
        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

        stride = self.width * 3
        raw = b"".join(b"\x00" + bytes(self.pixels[y * stride : (y + 1) * stride]) for y in range(self.height))
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(png)


def _metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(pred - true)))
    rel = max(0.0, 1.0 - float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 40.0))))
    daily = max(0.0, 1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), 40.0))
    return mae, rel, daily


def _load_rows(prediction_csv: Path) -> list[dict]:
    df = pd.read_csv(prediction_csv)
    rows = []
    for day, group in df.groupby("day", sort=True):
        group = group.sort_values("slot")
        pred = group["prediction"].to_numpy(dtype=float)
        true = group["target"].to_numpy(dtype=float)
        cand = group["candidate_prediction"].to_numpy(dtype=float)
        offset = group["point_offset"].to_numpy(dtype=float)
        mae, rel, daily = _metrics(pred, true)
        rows.append({"day": str(day), "prediction": pred, "truth": true, "candidate": cand, "offset": offset, "mae": mae, "rel": rel, "daily": daily})
    return rows


def _map_points(values: np.ndarray, x: int, y: int, w: int, h: int, ymin: float, ymax: float) -> list[tuple[float, float]]:
    span = max(ymax - ymin, 1e-6)
    return [(x + i * w / max(len(values) - 1, 1), y + h - ((float(v) - ymin) / span) * h) for i, v in enumerate(values)]


def _draw_card(canvas: Canvas, x: int, y: int, w: int, h: int, label: str, value: str) -> None:
    canvas.rect(x, y, w, h, PALETTE["panel"])
    canvas.rect(x, y, w, 2, PALETTE["grid"])
    canvas.rect(x, y + h - 2, w, 2, PALETTE["grid"])
    canvas.rect(x, y, 2, h, PALETTE["grid"])
    canvas.rect(x + w - 2, y, 2, h, PALETTE["grid"])
    canvas.text(x + 16, y + 18, label, PALETTE["axis"], 2)
    canvas.text(x + 16, y + 50, value, PALETTE["ink"], 4)


def _draw_bars(canvas: Canvas, x: int, y: int, w: int, h: int, rows: list[dict]) -> None:
    canvas.rect(x, y, w, h, PALETTE["panel"])
    canvas.text(x + 16, y + 12, "DAILY MEAN ACCURACY BY TEST DAY", PALETTE["ink"], 2)
    plot_x, plot_y = x + 54, y + 48
    plot_w, plot_h = w - 78, h - 84
    for tick in (0.0, 0.5, 0.8, 1.0):
        yy = plot_y + plot_h - tick * plot_h
        canvas.line(plot_x, yy, plot_x + plot_w, yy, PALETTE["green"] if tick == 0.8 else PALETTE["grid"], 1)
        canvas.text(x + 18, int(yy) - 6, f"{tick:.1f}", PALETTE["axis"], 1)
    group_w = plot_w / len(rows)
    for idx, row in enumerate(rows):
        val = row["daily"]
        bx = int(plot_x + idx * group_w + group_w * 0.22)
        bw = max(10, int(group_w * 0.48))
        by = int(plot_y + plot_h - val * plot_h)
        canvas.rect(bx, by, bw, int(plot_y + plot_h - by), PALETTE["calibrated"] if val >= 0.8 else PALETTE["bad"])
        if idx % 2 == 0:
            canvas.text(bx - 2, y + h - 24, row["day"][5:], PALETTE["axis"], 1)


def _draw_offsets(canvas: Canvas, x: int, y: int, w: int, h: int, offsets_csv: Path) -> None:
    offsets = pd.read_csv(offsets_csv)["applied_offset"].to_numpy(dtype=float)
    canvas.rect(x, y, w, h, PALETTE["panel"])
    canvas.text(x + 14, y + 12, "LEARNED POINT OFFSETS, 96 SLOTS", PALETTE["ink"], 2)
    plot_x, plot_y = x + 54, y + 48
    plot_w, plot_h = w - 78, h - 78
    ymin, ymax = min(-100.0, float(offsets.min()) - 10.0), max(100.0, float(offsets.max()) + 10.0)
    zero_y = plot_y + plot_h - (0.0 - ymin) / (ymax - ymin) * plot_h
    canvas.line(plot_x, zero_y, plot_x + plot_w, zero_y, PALETTE["grid"], 1)
    canvas.text(x + 14, int(zero_y) - 6, "0", PALETTE["axis"], 1)
    canvas.polyline(_map_points(offsets, plot_x, plot_y, plot_w, plot_h, ymin, ymax), PALETTE["offset"], 3)


def _draw_curve(canvas: Canvas, x: int, y: int, w: int, h: int, row: dict) -> None:
    canvas.rect(x, y, w, h, PALETTE["panel"])
    title = f"{row['day']} daily {row['daily']:.3f} rel {row['rel']:.3f} mae {row['mae']:.1f}"
    canvas.text(x + 12, y + 10, title, PALETTE["ink"], 2)
    plot_x, plot_y = x + 50, y + 40
    plot_w, plot_h = w - 64, h - 68
    values = np.concatenate([row["truth"], row["candidate"], row["prediction"]])
    ymin, ymax = max(0.0, float(values.min()) - 40.0), min(1000.0, float(values.max()) + 40.0)
    if ymax - ymin < 120.0:
        mid = (ymax + ymin) / 2.0
        ymin, ymax = max(0.0, mid - 60.0), min(1000.0, mid + 60.0)
    for frac in (0.0, 0.5, 1.0):
        yy = plot_y + plot_h - frac * plot_h
        canvas.line(plot_x, yy, plot_x + plot_w, yy, PALETTE["grid"], 1)
        canvas.text(x + 10, int(yy) - 6, f"{ymin + frac * (ymax - ymin):.0f}", PALETTE["axis"], 1)
    for slot, label in ((0, "00"), (24, "06"), (48, "12"), (72, "18"), (95, "24")):
        xx = plot_x + slot * plot_w / 95.0
        canvas.line(xx, plot_y, xx, plot_y + plot_h, PALETTE["grid"], 1)
        canvas.text(int(xx) - 6, y + h - 20, label, PALETTE["axis"], 1)
    canvas.polyline(_map_points(row["candidate"], plot_x, plot_y, plot_w, plot_h, ymin, ymax), PALETTE["candidate"], 2)
    canvas.polyline(_map_points(row["truth"], plot_x, plot_y, plot_w, plot_h, ymin, ymax), PALETTE["truth"], 2)
    canvas.polyline(_map_points(row["prediction"], plot_x, plot_y, plot_w, plot_h, ymin, ymax), PALETTE["calibrated"], 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_csv", default="outputs/gs_price_2025_point_offset_calibrated/test_predictions.csv")
    parser.add_argument("--summary_json", default="outputs/gs_price_2025_point_offset_calibrated/summary.json")
    parser.add_argument("--offsets_csv", default="outputs/gs_price_2025_point_offset_calibrated/point_offsets.csv")
    parser.add_argument("--output", default="outputs/gs_price_2025_point_offset_calibrated_viz.png")
    args = parser.parse_args()

    rows = _load_rows(Path(args.prediction_csv))
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    selected = summary["selected"]

    canvas = Canvas(1900, 2450, PALETTE["bg"])
    canvas.text(42, 34, "SUPPLY DEMAND CANDIDATE + POINT OFFSET", PALETTE["ink"], 4)
    canvas.text(44, 82, f"96 POINT CURVE, GROUP SIZE {selected['group_size']}, SHRINK {selected['shrink']}", PALETTE["axis"], 2)
    cards = [
        ("MEAN DAILY ACC", f"{metrics['mean_daily_mean_accuracy']:.3f}"),
        ("DAILY >= 0.8", f"{metrics['share_days_daily_mean_accuracy_ge_0_8']:.3f}"),
        ("MEAN POINTWISE", f"{metrics['mean_relative_accuracy']:.3f}"),
        ("MEAN MAE", f"{metrics['mean_day_mae']:.1f}"),
    ]
    for idx, (label, value) in enumerate(cards):
        _draw_card(canvas, 42 + idx * 455, 126, 420, 118, label, value)
    canvas.rect(44, 270, 26, 8, PALETTE["truth"])
    canvas.text(82, 262, "true", PALETTE["axis"], 2)
    canvas.rect(170, 270, 26, 8, PALETTE["candidate"])
    canvas.text(208, 262, "candidate baseline", PALETTE["axis"], 2)
    canvas.rect(460, 270, 26, 8, PALETTE["calibrated"])
    canvas.text(498, 262, "point offset calibrated", PALETTE["axis"], 2)
    _draw_bars(canvas, 42, 320, 900, 300, rows)
    _draw_offsets(canvas, 970, 320, 888, 300, Path(args.offsets_csv))
    panel_w, panel_h = 890, 260
    start_y = 650
    for idx, row in enumerate(rows[:12]):
        col = idx % 2
        line = idx // 2
        _draw_curve(canvas, 42 + col * 930, start_y + line * 290, panel_w, panel_h, row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save_png(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
