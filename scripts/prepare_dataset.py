"""
Step 2 - Convert and merge the raw datasets into YOLO format
============================================================

Kaggle datasets come in different annotation formats. This script scans
everything under ``data/raw/`` and understands both:

* **Pascal VOC XML** - one ``.xml`` per image with ``<bndbox>`` pixel corners
  (used by andrewmvd/car-plate-detection).
* **YOLO TXT**       - one ``.txt`` per image with
  ``class x_center y_center width height`` normalised to 0-1.

All boxes are converted to YOLO format with a single class ``plate`` (id 0),
duplicate images are removed (by file hash), and the result is split into
train / val / test folders that Ultralytics can train on directly::

    data/yolo/
      images/{train,val,test}/
      labels/{train,val,test}/
      data.yaml

Usage
-----
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --export-crops   # also save plate crops
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netra.config import ensure_dir, load_config, resolve  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Annotation readers - both return a list of (xc, yc, w, h) normalised boxes
# --------------------------------------------------------------------------- #
def read_voc(xml_path: Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Parse a Pascal VOC XML file. Every object is treated as a plate."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is not None:  # trust the XML size if present
        img_w = int(float(size.findtext("width", img_w))) or img_w
        img_h = int(float(size.findtext("height", img_h))) or img_h
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h))
    return boxes


def read_yolo(txt_path: Path) -> list[tuple[float, float, float, float]]:
    """Parse a YOLO label file, ignoring the original class id."""
    boxes = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1:
            boxes.append((xc, yc, w, h))
    return boxes


def clip_box(box):
    """Keep a normalised box inside the image."""
    xc, yc, w, h = box
    x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
    x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def collect_samples(dataset_dir: Path) -> list[tuple[Path, list]]:
    """Find every annotated image inside one downloaded dataset folder."""
    xml_by_stem, txt_by_stem = defaultdict(list), defaultdict(list)
    images = []
    for p in dataset_dir.rglob("*"):
        suffix = p.suffix.lower()
        if suffix == ".xml":
            xml_by_stem[p.stem].append(p)
        elif suffix == ".txt" and p.name not in {"classes.txt", "readme.txt"}:
            txt_by_stem[p.stem].append(p)
        elif suffix in IMAGE_EXTS:
            images.append(p)

    samples = []
    for img_path in images:
        boxes = []
        if img_path.stem in xml_by_stem:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            boxes = read_voc(xml_by_stem[img_path.stem][0], img.shape[1], img.shape[0])
        elif img_path.stem in txt_by_stem:
            # Prefer the label file in the matching "labels" folder of the same split.
            candidates = txt_by_stem[img_path.stem]
            mirrored = Path(str(img_path.parent).replace("images", "labels")) / f"{img_path.stem}.txt"
            label = mirrored if mirrored in candidates else candidates[0]
            boxes = read_yolo(label)
        boxes = [clip_box(b) for b in boxes]
        boxes = [b for b in boxes if b[2] > 0.002 and b[3] > 0.002]
        if boxes:
            samples.append((img_path, boxes))
    return samples


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-crops", action="store_true",
                        help="Also save every plate crop to data/ocr/real_unlabeled/ for manual labelling")
    args = parser.parse_args()

    cfg = load_config()
    det = cfg["detector"]
    raw_dir = resolve(cfg["paths"]["raw_data"])
    out_dir = resolve(cfg["paths"]["yolo_data"])

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        sys.exit(f"No raw data in {raw_dir}. Run `python scripts/download_dataset.py` first.")

    # 1. Gather annotated images from every dataset folder.
    all_samples, seen = [], set()
    for dataset_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        samples = collect_samples(dataset_dir)
        kept = 0
        for img_path, boxes in samples:
            digest = file_hash(img_path)
            if digest in seen:          # same image shipped twice -> skip
                continue
            seen.add(digest)
            all_samples.append((img_path, boxes, dataset_dir.name))
            kept += 1
        print(f"{dataset_dir.name:<45} {kept:>5} annotated images")

    if not all_samples:
        sys.exit("No annotated images were found. Check the folder structure in data/raw/.")

    # 2. Shuffle and split.
    random.Random(det["seed"]).shuffle(all_samples)
    n = len(all_samples)
    n_test = int(n * det["test_split"])
    n_val = int(n * det["val_split"])
    splits = {
        "test": all_samples[:n_test],
        "val": all_samples[n_test:n_test + n_val],
        "train": all_samples[n_test + n_val:],
    }

    # 3. Write images + labels.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    crop_dir = ensure_dir(Path(cfg["paths"]["ocr_data"]) / "real_unlabeled") if args.export_crops else None
    n_boxes = 0
    for split, items in splits.items():
        img_out = ensure_dir(out_dir / "images" / split)
        lbl_out = ensure_dir(out_dir / "labels" / split)
        for i, (img_path, boxes, source) in enumerate(items):
            name = f"{source[:12]}_{i:05d}"
            dst = img_out / f"{name}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dst)
            lines = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for xc, yc, w, h in boxes]
            (lbl_out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_boxes += len(boxes)

            if crop_dir is not None:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                H, W = img.shape[:2]
                for j, (xc, yc, w, h) in enumerate(boxes):
                    x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
                    x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
                    crop = img[max(0, y1):y2, max(0, x1):x2]
                    if crop.size:
                        cv2.imwrite(str(crop_dir / f"{name}_{j}.png"), crop)

    # 4. Dataset description file for Ultralytics.
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n  0: plate\n",
        encoding="utf-8",
    )

    print(f"\nTotal: {n} images, {n_boxes} plates")
    print(f"train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}")
    print(f"YOLO dataset written to {out_dir}")
    if crop_dir is not None:
        print(f"Plate crops saved to {crop_dir} - see data/README.md to label them for OCR.")


if __name__ == "__main__":
    main()
