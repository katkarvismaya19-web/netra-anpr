"""
netra.pipeline
==============

Glues the two deep learning stages and the rule-based post-processor into
one call::

    image ──► PlateDetector (YOLOv8) ──► crop + pad ──► Recognizer (CRNN)
                                                             │
                     PlateResult list ◄── correct_plate() ◄──┘

The pipeline also times every stage so the dashboard can report latency,
which is an important metric for real-time traffic systems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from netra.config import load_config, resolve
from netra.detector import Detection, PlateDetector
from netra.ocr.recognizer import load_recognizer
from netra.postprocess import PlateReading, correct_plate
from netra.preprocess import expand_box


@dataclass
class PlateResult:
    """Everything known about one plate in one image."""

    box: tuple[int, int, int, int]
    detection_confidence: float
    ocr_confidence: float
    reading: PlateReading
    crop: np.ndarray = field(repr=False)

    @property
    def overall_confidence(self) -> float:
        """Joint confidence: both stages must be right for the result to be right."""
        return self.detection_confidence * self.ocr_confidence


@dataclass
class PipelineOutput:
    plates: list[PlateResult]
    timings_ms: dict[str, float]


class ANPRPipeline:
    """
    Detector + recognizer + post-processing.

    Args:
        detector_weights / ocr_weights: override paths from config.yaml.
        ocr_engine: ``"crnn"`` (trained here) or ``"easyocr"`` (baseline).
        conf / iou: detector thresholds.
        load_detector: set False to run OCR only on pre-cropped plates.
    """

    def __init__(
        self,
        detector_weights: str | None = None,
        ocr_weights: str | None = None,
        ocr_engine: str = "crnn",
        conf: float | None = None,
        iou: float | None = None,
        load_detector: bool = True,
    ) -> None:
        cfg = load_config()
        det_cfg, ocr_cfg = cfg["detector"], cfg["ocr"]

        self.detector = None
        if load_detector:
            self.detector = PlateDetector(
                resolve(detector_weights or det_cfg["weights"]),
                conf=conf if conf is not None else det_cfg["conf_threshold"],
                iou=iou if iou is not None else det_cfg["iou_threshold"],
            )
        self.recognizer = load_recognizer(ocr_engine, resolve(ocr_weights or ocr_cfg["weights"]))

    # ------------------------------------------------------------------ #
    def read_crops(self, crops: list[np.ndarray]) -> list[tuple[PlateReading, float]]:
        """OCR + post-processing for crops that are already isolated plates."""
        raw = self.recognizer.read(crops)
        return [(correct_plate(text), conf) for text, conf in raw]

    def run(self, image_bgr: np.ndarray) -> PipelineOutput:
        """Detect and read every plate in a full scene image."""
        if self.detector is None:
            raise RuntimeError("Pipeline was created without a detector.")
        timings: dict[str, float] = {}

        start = time.perf_counter()
        detections: list[Detection] = self.detector.detect(image_bgr)
        timings["Detection"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        boxes = [expand_box(d.box, image_bgr.shape) for d in detections]
        crops = [image_bgr[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]
        valid = [i for i, c in enumerate(crops) if c.size and min(c.shape[:2]) >= 8]
        readings = self.read_crops([crops[i] for i in valid])
        timings["Recognition"] = (time.perf_counter() - start) * 1000

        plates = [
            PlateResult(
                box=detections[i].box,
                detection_confidence=detections[i].confidence,
                ocr_confidence=ocr_conf,
                reading=reading,
                crop=crops[i],
            )
            for i, (reading, ocr_conf) in zip(valid, readings)
        ]
        timings["Total"] = timings["Detection"] + timings["Recognition"]
        return PipelineOutput(plates=plates, timings_ms=timings)

    def run_on_crop(self, crop_bgr: np.ndarray) -> PipelineOutput:
        """Treat the whole image as one plate (no detection step)."""
        start = time.perf_counter()
        reading, conf = self.read_crops([crop_bgr])[0]
        elapsed = (time.perf_counter() - start) * 1000
        h, w = crop_bgr.shape[:2]
        plate = PlateResult((0, 0, w, h), 1.0, conf, reading, crop_bgr)
        return PipelineOutput([plate], {"Detection": 0.0, "Recognition": elapsed, "Total": elapsed})
