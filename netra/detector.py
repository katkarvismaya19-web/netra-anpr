"""
netra.detector
==============

Stage 1 of the pipeline: locate number plates with YOLOv8.

YOLO ("You Only Look Once") is a single-shot object detector: one forward
pass of a CNN predicts bounding boxes and class scores for the whole image.
We start from weights pretrained on COCO (80 everyday object classes) and
fine-tune them on Kaggle number-plate images - this is *transfer learning*,
which lets a small dataset (a few thousand images) reach high accuracy.

Post-processing inside Ultralytics:
    * confidence threshold - drop weak predictions
    * Non-Max Suppression  - keep one box when several overlap (IoU test)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    """One plate found in an image (pixel coordinates)."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class PlateDetector:
    """Wraps an Ultralytics YOLO model trained for the single class 'plate'."""

    def __init__(self, weights: str | Path, conf: float = 0.35, iou: float = 0.45) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Detector weights not found at {weights}. Train them with "
                "`python scripts/train_detector.py`."
            )
        from ultralytics import YOLO  # heavy import kept local

        self.model = YOLO(str(weights))
        self.conf, self.iou = conf, iou
        self.weights = weights

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Return all plates in the image, highest confidence first."""
        result = self.model.predict(image_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections = []
        for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(Detection(x1, y1, x2, y2, float(score)))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)
