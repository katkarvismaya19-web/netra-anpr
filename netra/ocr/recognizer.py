"""
netra.ocr.recognizer
====================

Inference wrapper that reads text from plate crops.

Two engines are supported:

* ``crnn``    - the model trained in this project (default, fully offline).
* ``easyocr`` - a pretrained general-purpose OCR used as a baseline for
                comparison in the report, or as a fallback before the CRNN
                has been trained. It is optional and only imported if used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from netra.ocr.charset import greedy_decode
from netra.ocr.model import CRNN, count_parameters
from netra.preprocess import prepare_for_ocr


def pick_device() -> torch.device:
    """Use the GPU when one is available, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CRNNRecognizer:
    """Loads a trained CRNN checkpoint and reads batches of plate crops."""

    name = "crnn"

    def __init__(self, weights: str | Path, device: torch.device | None = None) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"OCR weights not found at {weights}. Train them with "
                "`python scripts/train_ocr.py`."
            )
        self.device = device or pick_device()
        checkpoint = torch.load(weights, map_location=self.device)
        cfg = checkpoint.get("config", {})
        self.height = cfg.get("image_height", 32)
        self.width = cfg.get("image_width", 128)
        self.model = CRNN(hidden_size=cfg.get("hidden_size", 256)).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.metrics = checkpoint.get("metrics", {})
        self.parameters = count_parameters(self.model)

    @torch.no_grad()
    def read(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Return ``(text, confidence)`` for each BGR crop."""
        if not crops:
            return []
        batch = np.stack([prepare_for_ocr(c, self.height, self.width) for c in crops])
        x = torch.from_numpy(batch).unsqueeze(1).to(self.device)
        return greedy_decode(self.model(x).cpu())


class EasyOCRRecognizer:
    """Thin adapter so EasyOCR exposes the same ``read`` method."""

    name = "easyocr"

    def __init__(self) -> None:
        try:
            import easyocr  # imported lazily - large optional dependency
        except ImportError as exc:
            raise ImportError("EasyOCR is not installed. Run `pip install easyocr`.") from exc
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        self.metrics, self.parameters = {}, None

    def read(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        allow = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        results = []
        for crop in crops:
            parts = self.reader.readtext(crop, allowlist=allow, detail=1, paragraph=False)
            # Sort fragments top-to-bottom, then left-to-right (handles two-line plates).
            parts.sort(key=lambda p: (round(p[0][0][1] / max(crop.shape[0] / 2, 1)), p[0][0][0]))
            text = "".join(p[1] for p in parts)
            conf = float(np.mean([p[2] for p in parts])) if parts else 0.0
            results.append((text, conf))
        return results


def load_recognizer(engine: str, weights: str | Path):
    """Factory used by the pipeline and dashboard."""
    if engine == "easyocr":
        return EasyOCRRecognizer()
    return CRNNRecognizer(weights)
