"""
Step 6 - Evaluate the trained models
====================================

Produces the numbers for the "Results" chapter of the report.

Detector  : mAP@0.5, mAP@0.5:0.95, precision and recall on the test split.
Recognizer: plate accuracy and character error rate on any labelled folder,
            optionally compared against the EasyOCR baseline.

Usage
-----
    python scripts/evaluate.py --detector
    python scripts/evaluate.py --ocr data/ocr/synthetic/val
    python scripts/evaluate.py --ocr data/ocr/real/val --compare-easyocr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netra.config import ensure_dir, load_config, resolve  # noqa: E402
from netra.metrics import character_error_rate, plate_accuracy  # noqa: E402
from netra.ocr.dataset import read_labels  # noqa: E402
from netra.ocr.recognizer import load_recognizer  # noqa: E402
from netra.postprocess import correct_plate  # noqa: E402


def evaluate_detector(cfg: dict) -> dict:
    from ultralytics import YOLO

    data_yaml = resolve(cfg["paths"]["yolo_data"]) / "data.yaml"
    model = YOLO(str(resolve(cfg["detector"]["weights"])))
    m = model.val(data=str(data_yaml), split="test", imgsz=cfg["detector"]["image_size"], verbose=False)
    return {"mAP50": m.box.map50, "mAP50_95": m.box.map, "precision": m.box.mp, "recall": m.box.mr}


def evaluate_ocr(cfg: dict, folder: Path, engine: str) -> dict:
    samples = read_labels(folder)
    if not samples:
        sys.exit(f"No labels.csv found in {folder}")
    recognizer = load_recognizer(engine, resolve(cfg["ocr"]["weights"]))
    images = [cv2.imread(str(p)) for p, _ in samples]
    targets = [t for _, t in samples]

    start = time.perf_counter()
    raw = []
    for i in range(0, len(images), 64):
        raw.extend(t for t, _ in recognizer.read(images[i:i + 64]))
    per_plate_ms = (time.perf_counter() - start) * 1000 / len(images)

    corrected = [correct_plate(t).text for t in raw]
    return {
        "engine": engine,
        "samples": len(samples),
        "plate_accuracy": plate_accuracy(raw, targets),
        "plate_accuracy_corrected": plate_accuracy(corrected, targets),
        "cer": character_error_rate(raw, targets),
        "cer_corrected": character_error_rate(corrected, targets),
        "ms_per_plate": per_plate_ms,
    }


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", action="store_true", help="Evaluate the YOLO detector")
    parser.add_argument("--ocr", type=Path, help="Folder with labels.csv to evaluate OCR on")
    parser.add_argument("--compare-easyocr", action="store_true", help="Also score the EasyOCR baseline")
    args = parser.parse_args()

    if not args.detector and not args.ocr:
        parser.error("Choose --detector and/or --ocr <folder>")

    report = {}
    if args.detector:
        report["detector"] = evaluate_detector(cfg)
    if args.ocr:
        folder = resolve(args.ocr)
        report["ocr"] = [evaluate_ocr(cfg, folder, "crnn")]
        if args.compare_easyocr:
            report["ocr"].append(evaluate_ocr(cfg, folder, "easyocr"))

    print(json.dumps(report, indent=2))
    out = ensure_dir(cfg["paths"]["outputs"]) / "evaluation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
