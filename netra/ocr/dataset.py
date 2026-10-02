"""
netra.ocr.dataset
=================

PyTorch ``Dataset`` that feeds plate images and their text to the CRNN.

Expected folder layout (created by ``scripts/generate_plates.py``)::

    data/ocr/
      synthetic/train/labels.csv   + images
      synthetic/val/labels.csv     + images
      real/labels.csv              + images   (optional, your own crops)

``labels.csv`` has two columns: ``filename,text`` - e.g.
``000123.png,MH12AB1234``.

Augmentation (training only) simulates what real CCTV / phone images do to a
plate: perspective tilt, motion blur, sensor noise, low light, shadows,
JPEG compression and loose detector crops. This is what lets a model trained
on rendered plates generalise to photographs.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from netra.ocr.charset import encode
from netra.preprocess import prepare_for_ocr


# --------------------------------------------------------------------------- #
# Augmentations - each takes and returns a BGR uint8 image
# --------------------------------------------------------------------------- #
def random_perspective(img: np.ndarray, rng: random.Random, strength: float = 0.08) -> np.ndarray:
    """Tilt the plate as if photographed from an angle."""
    h, w = img.shape[:2]
    jitter = lambda s: rng.uniform(-strength, strength) * s  # noqa: E731
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [jitter(w), jitter(h)], [w + jitter(w), jitter(h)],
        [w + jitter(w), h + jitter(h)], [jitter(w), h + jitter(h)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_rotate(img: np.ndarray, rng: random.Random, max_deg: float = 5) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-max_deg, max_deg), 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gaussian (out of focus) or horizontal motion blur (moving vehicle)."""
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    k = rng.choice([5, 7, 9])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def random_lighting(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Brightness / contrast change plus an optional diagonal shadow."""
    alpha, beta = rng.uniform(0.55, 1.35), rng.uniform(-45, 35)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if rng.random() < 0.3:
        h, w = out.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        x = rng.randint(0, w)
        pts = np.array([[x, 0], [min(w, x + rng.randint(w // 4, w)), 0],
                        [rng.randint(0, w), h], [0, h]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        shade = rng.uniform(0.45, 0.8)
        out = np.where(mask[..., None] > 0, (out * shade).astype(np.uint8), out)
    return out


def random_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(3, 14)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 80)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def random_crop_margin(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mimic loose or tight detector boxes by padding / trimming the edges."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        pad = [rng.randint(0, int(h * 0.15)) for _ in range(2)] + [rng.randint(0, int(w * 0.05)) for _ in range(2)]
        colour = [rng.randint(30, 200)] * 3
        return cv2.copyMakeBorder(img, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=colour)
    t, b = rng.randint(0, int(h * 0.06)), rng.randint(0, int(h * 0.06))
    l, r = rng.randint(0, int(w * 0.02)), rng.randint(0, int(w * 0.02))
    return img[t:h - b or h, l:w - r or w]


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a random subset of augmentations (each with its own probability)."""
    if rng.random() < 0.6:
        img = random_perspective(img, rng)
    if rng.random() < 0.4:
        img = random_rotate(img, rng)
    if rng.random() < 0.5:
        img = random_crop_margin(img, rng)
    # Downscale then upscale: low-resolution CCTV look.
    if rng.random() < 0.5:
        h, w = img.shape[:2]
        f = rng.uniform(0.2, 0.6)
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        img = random_blur(img, rng)
    if rng.random() < 0.7:
        img = random_lighting(img, rng)
    if rng.random() < 0.4:
        img = random_noise(img, rng)
    if rng.random() < 0.4:
        img = random_jpeg(img, rng)
    return img


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def read_labels(folder: Path) -> list[tuple[Path, str]]:
    """Read ``labels.csv`` in ``folder`` and return (image_path, text) pairs."""
    csv_path = folder / "labels.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [(folder / row["filename"], row["text"].strip().upper()) for row in reader if row.get("text")]


class PlateTextDataset(Dataset):
    """(image tensor, encoded label, label length, raw text) samples for CTC training."""

    def __init__(self, folders: list[Path], height: int = 32, width: int = 128,
                 train: bool = True, seed: int = 0) -> None:
        self.samples: list[tuple[Path, str]] = []
        for folder in folders:
            self.samples.extend(read_labels(Path(folder)))
        if not self.samples:
            raise RuntimeError(
                f"No labelled images found in {folders}. "
                "Run `python scripts/generate_plates.py` first."
            )
        self.height, self.width, self.train = height, width, train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, text = self.samples[index]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image {path}")
        if self.train:
            img = augment(img, self.rng)
        x = prepare_for_ocr(img, self.height, self.width)
        target = encode(text)
        return (
            torch.from_numpy(x).unsqueeze(0),          # (1, H, W)
            torch.tensor(target, dtype=torch.long),
            len(target),
            text,
        )


def collate(batch):
    """
    Stack a batch for ``nn.CTCLoss``: images are stacked, labels are
    concatenated into one 1-D tensor alongside their individual lengths.
    """
    images, targets, lengths, texts = zip(*batch)
    return (
        torch.stack(images),
        torch.cat(targets),
        torch.tensor(lengths, dtype=torch.long),
        list(texts),
    )


def worker_init(worker_id: int) -> None:
    """Give each DataLoader worker its own random stream for augmentation."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = random.Random(info.seed % 2**31)
