"""
netra.ocr.synth
===============

Renders realistic synthetic Indian number plates for training the CRNN.

Why synthetic data?
    Public Kaggle datasets give us plate *bounding boxes* (perfect for the
    detector) but almost never the *text* written on each plate. Training an
    OCR model needs tens of thousands of (image, text) pairs, so we generate
    them: random but format-correct registration numbers drawn in several
    condensed fonts, on the colour schemes used in India, with the blue
    "IND" strip of High Security Registration Plates (HSRP).

    Heavy augmentation in :mod:`netra.ocr.dataset` (blur, perspective, noise,
    shadows, JPEG artefacts) bridges the gap between rendered and real plates.

Fonts are open-licence (SIL OFL) Google Fonts downloaded on first use.
If there is no internet connection, OpenCV's built-in Hershey fonts are used.
"""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from netra.states import STATE_CODES

# Condensed, sign-like typefaces that resemble embossed plate lettering.
FONT_URLS: dict[str, str] = {
    "BarlowCondensed-SemiBold.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-SemiBold.ttf",
    "BarlowCondensed-Bold.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-Bold.ttf",
    "Oswald.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "RobotoCondensed.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/robotocondensed/RobotoCondensed%5Bwght%5D.ttf",
    "ArchivoNarrow.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/archivonarrow/ArchivoNarrow%5Bwght%5D.ttf",
    "PathwayGothicOne.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/pathwaygothicone/PathwayGothicOne-Regular.ttf",
}

# (background, text colour, probability) for Indian plate categories.
COLOUR_SCHEMES = [
    ((250, 250, 248), (20, 20, 20), 0.62),   # private vehicles: white / black
    ((32, 200, 245), (20, 20, 20), 0.22),    # commercial: yellow / black (BGR)
    ((70, 150, 40), (250, 250, 250), 0.08),  # electric private: green / white
    ((20, 20, 20), (40, 210, 245), 0.04),    # rental self-drive: black / yellow
    ((235, 235, 235), (40, 40, 40), 0.04),   # faded / dirty white plates
]

# States weighted roughly by vehicle population so common codes appear more.
HEAVY_STATES = ["MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "KL", "TS", "AP", "HR", "WB", "MP", "PB"]
SERIES_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # I and O are not issued in series


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
def ensure_fonts(font_dir: Path) -> list[Path]:
    """Download the plate fonts once and return the list of usable font files."""
    font_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FONT_URLS.items():
        target = font_dir / name
        if target.exists() and target.stat().st_size > 10_000:
            continue
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:  # noqa: BLE001 - offline is fine, we fall back
            print(f"[fonts] could not download {name}: {exc}")
    return sorted(p for p in font_dir.glob("*.ttf") if p.stat().st_size > 10_000)


# --------------------------------------------------------------------------- #
# Random registration numbers
# --------------------------------------------------------------------------- #
def random_plate_text(rng: random.Random) -> tuple[str, list[str]]:
    """
    Return a format-correct plate string and its logical groups.

    Example: ("MH12AB1234", ["MH", "12", "AB", "1234"])
    """
    if rng.random() < 0.05:  # Bharat series, e.g. 22 BH 4521 AK
        groups = [
            f"{rng.randint(21, 26):02d}", "BH",
            f"{rng.randint(0, 9999):04d}",
            "".join(rng.choices(SERIES_LETTERS, k=rng.choice([1, 2]))),
        ]
        return "".join(groups), groups

    state = rng.choice(HEAVY_STATES) if rng.random() < 0.7 else rng.choice(list(STATE_CODES))
    district = f"{rng.randint(1, 99):02d}" if rng.random() < 0.85 else str(rng.randint(1, 9))
    series = "".join(rng.choices(SERIES_LETTERS, k=rng.choices([0, 1, 2, 3], [0.05, 0.2, 0.65, 0.1])[0]))
    number = f"{rng.randint(1, 9999):04d}" if rng.random() < 0.9 else str(rng.randint(1, 999))
    groups = [g for g in (state, district, series, number) if g]
    return "".join(groups), groups


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _pick_scheme(rng: random.Random):
    weights = [s[2] for s in COLOUR_SCHEMES]
    bg, fg, _ = rng.choices(COLOUR_SCHEMES, weights)[0]
    return bg, fg


def _draw_text_line(
    canvas: Image.Image, text: str, box: tuple[int, int, int, int], font_path: Path | None,
    colour: tuple[int, int, int], rng: random.Random,
) -> None:
    """Draw ``text`` centred inside ``box`` using the largest size that fits."""
    x1, y1, x2, y2 = box
    draw = ImageDraw.Draw(canvas)
    box_w, box_h = x2 - x1, y2 - y1
    rgb = colour[::-1]  # PIL is RGB, our schemes are BGR

    if font_path is None:  # Hershey fallback via OpenCV
        arr = np.array(canvas)[:, :, ::-1].copy()
        face = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX])
        thick = rng.randint(2, 3)
        scale = box_h / 26
        (tw, th), _ = cv2.getTextSize(text, face, scale, thick)
        if tw > box_w:
            scale *= box_w / tw
            (tw, th), _ = cv2.getTextSize(text, face, scale, thick)
        org = (x1 + (box_w - tw) // 2, y1 + (box_h + th) // 2)
        cv2.putText(arr, text, org, face, scale, colour, thick, cv2.LINE_AA)
        canvas.paste(Image.fromarray(arr[:, :, ::-1]))
        return

    size = int(box_h * rng.uniform(1.0, 1.25))
    font = ImageFont.truetype(str(font_path), size)
    while size > 8:
        font = ImageFont.truetype(str(font_path), size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left <= box_w and bottom - top <= box_h:
            break
        size -= 2
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tx = x1 + (box_w - (right - left)) // 2 - left
    ty = y1 + (box_h - (bottom - top)) // 2 - top
    draw.text((tx, ty), text, font=font, fill=rgb)


def render_plate(
    rng: random.Random, fonts: list[Path], text_groups: list[str] | None = None,
) -> tuple[np.ndarray, str]:
    """
    Render one plate image (BGR uint8) and return it with its label.

    25 % of plates are drawn in the two-row layout used on two-wheelers and
    commercial vehicles, so the model also learns that style.
    """
    if text_groups is None:
        label, text_groups = random_plate_text(rng)
    else:
        label = "".join(text_groups)

    bg, fg = _pick_scheme(rng)
    font = rng.choice(fonts) if fonts and rng.random() > 0.08 else None
    two_line = rng.random() < 0.25
    hsrp = rng.random() < 0.6 and bg[0] > 200  # IND strip mostly on white plates

    if two_line:
        w, h = rng.randint(200, 240), rng.randint(130, 160)
    else:
        w, h = rng.randint(420, 520), rng.randint(95, 115)

    img = Image.new("RGB", (w, h), bg[::-1])
    draw = ImageDraw.Draw(img)

    # Raised border found on most plates.
    border = rng.randint(2, 5)
    draw.rectangle([border, border, w - border - 1, h - border - 1],
                   outline=fg[::-1], width=rng.randint(1, 3))

    left = border + 6
    if hsrp:  # Blue IND strip on the left edge of HSRP plates
        strip_w = int(h * (0.28 if not two_line else 0.2))
        draw.rectangle([border + 3, border + 3, border + 3 + strip_w, h - border - 4],
                       fill=(30, 70, 160))
        try:
            small = ImageFont.truetype(str(fonts[0]), max(8, strip_w // 3)) if fonts else None
        except OSError:
            small = None
        if small:
            draw.text((border + 5, h - border - strip_w // 2 - 4), "IND", font=small, fill=(255, 255, 255))
        left = border + strip_w + 8

    margin_y = int(h * 0.12)
    sep = " " if rng.random() < 0.8 else ""
    if two_line:
        top_text = sep.join(text_groups[:2])
        bottom_text = sep.join(text_groups[2:])
        mid = h // 2
        _draw_text_line(img, top_text, (left, margin_y, w - border - 6, mid - 2), font, fg, rng)
        _draw_text_line(img, bottom_text, (left, mid + 2, w - border - 6, h - margin_y), font, fg, rng)
    else:
        _draw_text_line(img, sep.join(text_groups), (left, margin_y, w - border - 6, h - margin_y),
                        font, fg, rng)

    return np.array(img)[:, :, ::-1].copy(), label
