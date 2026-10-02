"""
netra.preprocess
================

Image preparation that sits between the detector and the OCR model.

The same functions are used during OCR *training* and at *inference*, which
guarantees the network always sees images prepared in exactly the same way
(a common source of bugs when the two pipelines drift apart).

Steps applied to every plate crop:

1. **Grayscale**      - colour carries little information for reading text.
2. **CLAHE**          - Contrast Limited Adaptive Histogram Equalisation
                        evens out shadows, glare and night-time images.
3. **Two-line fix**   - square plates (bikes, autos, trucks) have two rows.
                        We cut them in half and place the rows side by side
                        so the single-line CRNN can read them.
4. **Resize + pad**   - scale to 32 px height while keeping the aspect
                        ratio, then pad to 128 px width.
"""

from __future__ import annotations

import cv2
import numpy as np

# A plate whose width / height is below this value is treated as two-line.
TWO_LINE_ASPECT_RATIO = 2.2


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) or already-gray images to single channel."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE so characters stand out under uneven lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))
    return clahe.apply(gray)


def is_two_line(image: np.ndarray) -> bool:
    """Square-ish crops are almost always two-row plates."""
    h, w = image.shape[:2]
    return h > 0 and (w / h) < TWO_LINE_ASPECT_RATIO


def merge_two_lines(gray: np.ndarray) -> np.ndarray:
    """
    Split a two-row plate horizontally and join the rows into one line.

    A small overlap (4 % of the height) is kept on each half so characters
    sitting slightly off-centre are not cut.
    """
    h = gray.shape[0]
    overlap = max(1, int(h * 0.04))
    top = gray[: h // 2 + overlap]
    bottom = gray[h // 2 - overlap:]
    height = min(top.shape[0], bottom.shape[0])
    return np.hstack([top[:height], bottom[:height]])


def resize_and_pad(gray: np.ndarray, height: int = 32, width: int = 128) -> np.ndarray:
    """
    Resize to ``height`` keeping the aspect ratio; pad (or squeeze) to ``width``.

    Padding uses the median border colour so it blends with the plate
    background instead of adding a hard black edge.
    """
    h, w = gray.shape[:2]
    new_w = max(1, min(width, int(round(w * height / max(h, 1)))))
    resized = cv2.resize(gray, (new_w, height), interpolation=cv2.INTER_AREA)
    if new_w == width:
        return resized
    fill = int(np.median(np.concatenate([resized[:, 0], resized[:, -1]])))
    canvas = np.full((height, width), fill, dtype=np.uint8)
    canvas[:, :new_w] = resized
    return canvas


def prepare_for_ocr(
    crop: np.ndarray,
    height: int = 32,
    width: int = 128,
    return_stages: bool = False,
):
    """
    Full preprocessing chain for one plate crop.

    Args:
        crop: BGR or grayscale image of a single plate.
        return_stages: if True, also return the intermediate images so the
            dashboard can show what the network actually "sees".

    Returns:
        ``float32`` array of shape (height, width) scaled to [0, 1]
        (and optionally a dict of intermediate uint8 images).
    """
    gray = to_gray(crop)
    enhanced = enhance_contrast(gray)
    line = merge_two_lines(enhanced) if is_two_line(enhanced) else enhanced
    final = resize_and_pad(line, height, width)
    tensor_ready = final.astype(np.float32) / 255.0

    if return_stages:
        stages = {"Grayscale": gray, "Contrast enhanced": enhanced, "Network input": final}
        return tensor_ready, stages
    return tensor_ready


def expand_box(box: tuple[int, int, int, int], shape: tuple[int, ...], pad: float = 0.06):
    """
    Grow a detector box by ``pad`` (fraction of its size) on every side.

    Detectors often place boxes tightly on the characters; a little margin
    stops the first and last characters from being clipped.
    """
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)
