from __future__ import annotations

import argparse
from pathlib import Path
import re


SVG_SIZE_RE = re.compile(r'viewBox="0 0 (\d+) (\d+)"')


def _load_svg(path: Path) -> tuple[int, int, str]:
    text = path.read_text(encoding="utf-8")
    m = SVG_SIZE_RE.search(text)
    if not m:
        raise ValueError(f"Could not parse viewBox from {path}")
    width = int(m.group(1))
    height = int(m.group(2))
    inner = text.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return width, height, inner


def main():
    parser = argparse.ArgumentParser(description="Combine multiple candidate SVGs into one large panel.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--labels", required=True,
                        help="Comma-separated candidate labels, e.g. C01,C03,C06,C07,C11")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("At least one label is required.")

    files = []
    for label in labels:
        matches = sorted(input_dir.glob(f"{label}_*.svg"))
        if not matches:
            raise FileNotFoundError(f"No SVG found for label {label} under {input_dir}")
        files.append(matches[0])

    loaded = [_load_svg(path) for path in files]
    cell_w = max(w for w, _, _ in loaded)
    cell_h = max(h for _, h, _ in loaded)
    cols = 2
    rows = (len(files) + cols - 1) // cols
    margin = 30
    title_h = 50
    out_w = margin * 2 + cols * cell_w
    out_h = title_h + margin + rows * cell_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{out_w}" height="{out_h}" viewBox="0 0 {out_w} {out_h}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="50%" y="30" font-size="22" fill="#111827" text-anchor="middle" font-family="sans-serif" font-weight="bold">Selected Full-vs-woDiffDoRA Report Candidates</text>',
    ]

    for idx, (_, _, inner) in enumerate(loaded):
        row = idx // cols
        col = idx % cols
        x = margin + col * cell_w
        y = title_h + row * cell_h
        parts.append(f'<g transform="translate({x},{y})">{inner}</g>')

    parts.append("</svg>")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
