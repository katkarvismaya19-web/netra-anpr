"""
Step 3 - Train the number plate detector (YOLOv8)
=================================================

Fine-tunes a COCO-pretrained YOLOv8 model on the merged plate dataset.

What happens during training
----------------------------
* **Transfer learning** - the backbone already knows edges, shapes and
  textures from COCO, so it only needs to learn what a plate looks like.
* **Augmentation** (built into Ultralytics) - mosaic, HSV colour jitter,
  scaling, translation and horizontal flips create new variations of every
  image on the fly and reduce overfitting.
* **Loss** = box regression (CIoU + DFL) + classification (BCE).
* **Early stopping** - training stops when validation mAP has not improved
  for ``patience`` epochs.

The best checkpoint is copied to ``models/plate_detector.pt`` and training
curves (loss, precision, recall, mAP) are saved in ``outputs/detector/``.

Usage
-----
    python scripts/train_detector.py
    python scripts/train_detector.py --epochs 80 --model yolov8s.pt --batch 32

GPU strongly recommended (Google Colab free T4: ~15-25 min for 50 epochs).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netra.config import ensure_dir, load_config, resolve  # noqa: E402


def main() -> None:
    cfg = load_config()
    det = cfg["detector"]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=det["base_model"], help="Starting weights, e.g. yolov8n.pt / yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=det["epochs"])
    parser.add_argument("--batch", type=int, default=det["batch"])
    parser.add_argument("--imgsz", type=int, default=det["image_size"])
    parser.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' to force CPU")
    args = parser.parse_args()

    data_yaml = resolve(cfg["paths"]["yolo_data"]) / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"{data_yaml} not found. Run `python scripts/prepare_dataset.py` first.")

    from ultralytics import YOLO

    project_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "detector")
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=det["patience"],
        seed=det["seed"],
        device=args.device,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        # Augmentation tuned for plates: no vertical flips (text would be upside down),
        # horizontal flips kept low because mirrored text is unrealistic.
        fliplr=0.1,
        flipud=0.0,
        degrees=5.0,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
        verbose=True,
    )

    best = project_dir / "train" / "weights" / "best.pt"
    target = resolve(det["weights"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    print(f"\nBest detector weights copied to {target}")

    # Final evaluation on the held-out test split (falls back to val if test is empty).
    test_dir = data_yaml.parent / "images" / "test"
    split = "test" if test_dir.exists() and any(test_dir.iterdir()) else "val"
    metrics = YOLO(str(target)).val(data=str(data_yaml), split=split, imgsz=args.imgsz, verbose=False,
                                   project=str(project_dir), name="evaluation", exist_ok=True)
    print(f"\nEvaluation on the {split} split")
    print(f"mAP@0.5      : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")
    print(f"precision    : {metrics.box.mp:.3f}")
    print(f"recall       : {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
