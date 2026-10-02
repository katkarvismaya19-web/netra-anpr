"""
Step 4 - Generate synthetic plates for OCR training
===================================================

Renders labelled Indian number plates (see ``netra/ocr/synth.py`` for how)
and writes them to::

    data/ocr/synthetic/train/  000000.png ... + labels.csv
    data/ocr/synthetic/val/    000000.png ... + labels.csv

The validation set is rendered with a different random seed so the model is
always evaluated on plate numbers it has never seen.

Usage
-----
    python scripts/generate_plates.py                 # sizes from config.yaml
    python scripts/generate_plates.py --train 5000 --val 500   # quick test run
    python scripts/generate_plates.py --preview 12    # save a preview grid only
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netra.config import ensure_dir, load_config, resolve  # noqa: E402
from netra.ocr.synth import ensure_fonts, render_plate  # noqa: E402


def write_split(folder: Path, count: int, seed: int, fonts: list[Path]) -> None:
    """Render ``count`` plates into ``folder`` and write labels.csv."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "text"])
        for i in range(count):
            img, label = render_plate(rng, fonts)
            name = f"{i:06d}.png"
            cv2.imwrite(str(folder / name), img)
            writer.writerow([name, label])
            if (i + 1) % 5000 == 0 or i + 1 == count:
                print(f"  {folder.name}: {i + 1}/{count}")


def save_preview(path: Path, n: int, fonts: list[Path]) -> None:
    """Save a grid of example plates for the report / README."""
    rng = random.Random(7)
    tiles = []
    for _ in range(n):
        img, _ = render_plate(rng, fonts)
        tiles.append(cv2.resize(img, (300, 110) if img.shape[1] / img.shape[0] > 2.2 else (150, 110)))
    rows, row, width = [], [], 0
    for t in tiles:
        if width + t.shape[1] > 900 and row:
            rows.append(row)
            row, width = [], 0
        row.append(t)
        width += t.shape[1] + 10
    rows.append(row)
    canvas_rows = []
    for r in rows:
        line = np.full((130, 920, 3), 240, np.uint8)
        x = 10
        for t in r:
            line[10:120, x:x + t.shape[1]] = t
            x += t.shape[1] + 10
        canvas_rows.append(line)
    cv2.imwrite(str(path), np.vstack(canvas_rows))
    print(f"Preview saved to {path}")


def main() -> None:
    cfg = load_config()
    ocr = cfg["ocr"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=ocr["synthetic_train"])
    parser.add_argument("--val", type=int, default=ocr["synthetic_val"])
    parser.add_argument("--preview", type=int, default=0, help="Only save a preview grid of N plates")
    args = parser.parse_args()

    fonts = ensure_fonts(resolve(cfg["paths"]["fonts"]))
    print(f"Using {len(fonts)} fonts" + ("" if fonts else " (offline - Hershey fallback)"))

    if args.preview:
        save_preview(ensure_dir(cfg["paths"]["outputs"]) / "synthetic_preview.png", args.preview, fonts)
        return

    base = resolve(cfg["paths"]["ocr_data"]) / "synthetic"
    write_split(base / "train", args.train, ocr["seed"], fonts)
    write_split(base / "val", args.val, ocr["seed"] + 1000, fonts)
    print(f"Synthetic OCR data written to {base}")


if __name__ == "__main__":
    main()
