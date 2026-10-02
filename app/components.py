"""
app.components
==============

Small HTML building blocks used by the Streamlit dashboard.

Streamlit's native widgets cover inputs well, but the result view (the
number plate readout, confidence bars, timing strip) is custom HTML styled
by ``theme.css`` so it looks like a finished product rather than a notebook.
All user-visible text passes through ``html.escape`` before rendering.
"""

from __future__ import annotations

from html import escape

import cv2
import numpy as np


def load_css(path) -> str:
    """Return the stylesheet wrapped in a <style> tag."""
    with open(path, encoding="utf-8") as fh:
        return f"<style>{fh.read()}</style>"


def masthead() -> str:
    """Brand mark and product name shown at the top left of the dashboard."""
    return """
    <header class="nt-masthead">
      <div class="nt-brand">
        <div class="nt-mark" aria-hidden="true"><span></span></div>
        <h1>Netra</h1>
      </div>
    </header>
    """


def account_block(name: str, username: str) -> str:
    """Logged-in user shown at the top right of the dashboard."""
    initials = "".join(part[0] for part in name.split()[:2]).upper() or username[:2].upper()
    return f"""
    <div class="nt-account">
      <div class="nt-account__avatar">{escape(initials)}</div>
      <div><strong>{escape(name)}</strong><span>Logged in as {escape(username)}</span></div>
    </div>
    """


def plate(reading, size: str = "lg") -> str:
    """
    Render a reading as a High Security Registration Plate: white face,
    black embossed characters, blue IND strip on the left.
    """
    text = escape(reading.formatted or reading.text or "—")
    review = "" if reading.is_valid else " nt-plate--review"
    return f"""
    <div class="nt-plate nt-plate--{size}{review}">
      <div class="nt-plate__strip"><span class="nt-plate__chakra"></span><b>IND</b></div>
      <div class="nt-plate__text">{text}</div>
    </div>
    """


def confidence_bar(label: str, value: float) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    tone = "high" if pct >= 80 else "mid" if pct >= 50 else "low"
    return f"""
    <div class="nt-bar">
      <div class="nt-bar__row"><span>{escape(label)}</span><strong>{pct:.1f}%</strong></div>
      <div class="nt-bar__track"><div class="nt-bar__fill nt-bar__fill--{tone}" style="width:{pct:.1f}%"></div></div>
    </div>
    """


def reading_details(result, show_detection: bool = True) -> str:
    """Plate readout plus state, validity, raw OCR text and confidences."""
    r = result.reading
    verdict = (
        '<span class="nt-pill nt-pill--ok">Matches Indian format</span>'
        if r.is_valid else
        '<span class="nt-pill nt-pill--warn">Check manually</span>'
    )
    corrections = (
        f"{r.corrections} character{'s' if r.corrections != 1 else ''} corrected"
        if r.corrections else "No corrections needed"
    )
    return f"""
    <section class="nt-reading">
      {plate(r)}
      <div class="nt-reading__meta">
        {verdict}
        <dl>
          <div><dt>Registered in</dt><dd>{escape(r.state or "Unknown state code")}</dd></div>
          <div><dt>Raw OCR output</dt><dd class="nt-raw">{escape(r.raw or "—")}</dd></div>
          <div><dt>Format check</dt><dd>{corrections}</dd></div>
        </dl>
      </div>
      {confidence_bar("Detection confidence", result.detection_confidence) if show_detection else ""}
      {confidence_bar("Reading confidence", result.ocr_confidence)}
    </section>
    """


def timings(timings_ms: dict[str, float]) -> str:
    cells = "".join(
        f'<div><span>{escape(k)}</span><strong>{v:,.0f}<small> ms</small></strong></div>'
        for k, v in timings_ms.items()
    )
    return f'<div class="nt-timings">{cells}</div>'


def empty_state(title: str, body: str) -> str:
    return f'<div class="nt-empty"><h3>{escape(title)}</h3><p>{body}</p></div>'


def setup_steps(detector_ready: bool, ocr_ready: bool) -> str:
    """Checklist shown until both models are trained. Steps really are ordered."""
    steps = [
        ("Download the Kaggle datasets", "python scripts/download_dataset.py", detector_ready),
        ("Convert them to YOLO format", "python scripts/prepare_dataset.py", detector_ready),
        ("Train the plate detector", "python scripts/train_detector.py", detector_ready),
        ("Render synthetic plates", "python scripts/generate_plates.py", ocr_ready),
        ("Train the character reader", "python scripts/train_ocr.py", ocr_ready),
    ]
    items = "".join(
        f'<li class="{"done" if done else ""}"><div><strong>{escape(t)}</strong>'
        f'<code>{escape(cmd)}</code></div></li>'
        for t, cmd, done in steps
    )
    return f"""
    <div class="nt-setup">
      <h3>Finish setting up the models</h3>
      <p>The dashboard needs trained weights in the <code>models/</code> folder.
      Run these from the project root, or use the Colab notebook in <code>notebooks/</code>.</p>
      <ol>{items}</ol>
    </div>
    """


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """OpenCV uses BGR; Streamlit expects RGB."""
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def decode_upload(data: bytes) -> np.ndarray | None:
    """Turn uploaded file bytes into a BGR image (None if unreadable)."""
    array = np.frombuffer(data, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)
