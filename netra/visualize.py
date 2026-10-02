"""
netra.visualize
===============

Draws detection results onto images for the dashboard, CLI and reports.

Each plate gets a box plus a solid label tab showing the corrected number.
Valid readings use the accent blue; readings that do not match an Indian
format are drawn in amber so they stand out for manual review.
"""

from __future__ import annotations

import cv2
import numpy as np

VALID_COLOUR = (156, 78, 31)      # BGR of #1F4E9C - road-sign blue
REVIEW_COLOUR = (23, 163, 232)    # BGR of #E8A317 - road-marking amber
TEXT_COLOUR = (255, 255, 255)


def draw_results(image_bgr: np.ndarray, plates) -> np.ndarray:
    """
    Return a copy of ``image_bgr`` with every plate box and reading drawn.

    ``plates`` is the list of :class:`netra.pipeline.PlateResult` objects.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 900)
    thickness = max(2, int(round(2 * scale)))

    for plate in plates:
        x1, y1, x2, y2 = plate.box
        colour = VALID_COLOUR if plate.reading.is_valid else REVIEW_COLOUR
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

        label = plate.reading.formatted or "unreadable"
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), base = cv2.getTextSize(label, font, 0.7 * scale, 1)
        pad = int(6 * scale)
        top = y1 - th - 2 * pad
        if top < 0:  # no room above the box: draw the tab below it
            top = y2
        cv2.rectangle(out, (x1, top), (x1 + tw + 2 * pad, top + th + 2 * pad), colour, -1)
        cv2.putText(out, label, (x1 + pad, top + th + pad - 1), font, 0.7 * scale,
                    TEXT_COLOUR, max(1, int(scale)), cv2.LINE_AA)
    return out
