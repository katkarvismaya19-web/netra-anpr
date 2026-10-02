"""
Run the full pipeline from the command line
===========================================

Works on a single image, a folder of images, or a video file. Annotated
images / videos are written to ``outputs/predictions/`` and every reading is
appended to ``outputs/predictions/results.csv``.

Usage
-----
    python scripts/infer.py path/to/car.jpg
    python scripts/infer.py path/to/folder/
    python scripts/infer.py path/to/traffic.mp4 --stride 5
    python scripts/infer.py plate_crop.png --crop          # OCR only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netra.config import ensure_dir, load_config  # noqa: E402
from netra.pipeline import ANPRPipeline  # noqa: E402
from netra.visualize import draw_results  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def log_rows(writer, source: str, frame: int | None, output) -> None:
    for p in output.plates:
        writer.writerow([source, frame if frame is not None else "", p.reading.formatted,
                         p.reading.raw, p.reading.is_valid, p.reading.state or "",
                         f"{p.detection_confidence:.3f}", f"{p.ocr_confidence:.3f}"])
        print(f"{source}{'' if frame is None else f' @ frame {frame}'}: {p.reading.formatted} "
              f"({'valid' if p.reading.is_valid else 'check'}, {p.overall_confidence:.2f})")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--crop", action="store_true", help="Input images are already cropped plates")
    parser.add_argument("--engine", choices=["crnn", "easyocr"], default="crnn")
    parser.add_argument("--stride", type=int, default=cfg["app"]["video_frame_stride"])
    args = parser.parse_args()

    pipeline = ANPRPipeline(ocr_engine=args.engine, load_detector=not args.crop)
    out_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "predictions")

    with (out_dir / "results.csv").open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if fh.tell() == 0:
            writer.writerow(["source", "frame", "plate", "raw_ocr", "valid", "state",
                             "detection_conf", "ocr_conf"])

        if args.source.suffix.lower() in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(args.source))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_out = cv2.VideoWriter(str(out_dir / f"{args.source.stem}_annotated.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            frame_idx, last = 0, None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % args.stride == 0:
                    last = pipeline.run(frame)
                    log_rows(writer, args.source.name, frame_idx, last)
                video_out.write(draw_results(frame, last.plates) if last else frame)
                frame_idx += 1
            cap.release()
            video_out.release()
            return

        files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTS) \
            if args.source.is_dir() else [args.source]
        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                print(f"Skipping unreadable file {path}")
                continue
            output = pipeline.run_on_crop(image) if args.crop else pipeline.run(image)
            log_rows(writer, path.name, None, output)
            if not args.crop:
                cv2.imwrite(str(out_dir / f"{path.stem}_annotated.jpg"), draw_results(image, output.plates))

    print(f"\nResults saved in {out_dir}")


if __name__ == "__main__":
    main()
