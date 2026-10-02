"""
Dashboard page  (URL: /dashboard)
=================================

The main working screen of Netra. Only reachable after signing in on the
login page; anyone who opens /dashboard directly is redirected to /login.

Tabs
----
* Image         - one photo: annotated result, plate readout, timings and the
                  intermediate images the OCR network receives.
* Video         - samples every Nth frame, groups repeated sightings of the
                  same plate and lists each unique vehicle once.
* Batch         - many photos at once with a downloadable results table.
* Detection log - every reading made in this session, exportable as CSV.
* Model         - architecture summary and the training metrics stored in
                  the checkpoints.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

# Make `netra` (project root) and the app helpers importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import auth  # noqa: E402
import components as ui  # noqa: E402
from netra.config import load_config, resolve  # noqa: E402
from netra.preprocess import prepare_for_ocr  # noqa: E402
from netra.visualize import draw_results  # noqa: E402

# --------------------------------------------------------------------------- #
# Access guard - visitors who are not signed in are sent to the login page.
# --------------------------------------------------------------------------- #
USER = auth.current_user()
if USER is None:
    st.switch_page(auth.LOGIN_PAGE)

CFG = load_config()
DETECTOR_WEIGHTS = resolve(CFG["detector"]["weights"])
OCR_WEIGHTS = resolve(CFG["ocr"]["weights"])

if "log" not in st.session_state:
    st.session_state.log = []          # list of dicts, one per plate reading
if "results" not in st.session_state:
    st.session_state.results = {}      # cache so reruns do not repeat inference


# --------------------------------------------------------------------------- #
# Sidebar - recognition settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## Recognition settings")
    engine_label = st.radio(
        "Character reader",
        ["CRNN trained in this project", "EasyOCR baseline"],
        help="EasyOCR is a general pretrained OCR used for comparison. "
             "It needs `pip install easyocr`.",
    )
    engine = "crnn" if engine_label.startswith("CRNN") else "easyocr"

    conf = st.slider("Detection confidence threshold", 0.05, 0.95,
                     float(CFG["detector"]["conf_threshold"]), 0.05,
                     help="Plates detected with lower confidence are ignored.")
    iou = st.slider("Overlap threshold (NMS)", 0.1, 0.9,
                    float(CFG["detector"]["iou_threshold"]), 0.05,
                    help="Boxes overlapping more than this are merged into one.")
    crop_mode = st.toggle("Images are already cropped plates",
                          help="Skip detection and read the whole image as one plate.")
    show_stages = st.toggle("Show what the network sees", value=True)

    st.markdown("## Video")
    stride = st.slider("Analyse every Nth frame", 1, 30, int(CFG["app"]["video_frame_stride"]))


# --------------------------------------------------------------------------- #
# Model loading (cached across reruns and users)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models")
def load_pipeline(ocr_engine: str, with_detector: bool):
    from netra.pipeline import ANPRPipeline
    return ANPRPipeline(ocr_engine=ocr_engine, load_detector=with_detector)


def easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


detector_ready = DETECTOR_WEIGHTS.exists()
ocr_ready = OCR_WEIGHTS.exists() if engine == "crnn" else easyocr_available()
can_run = ocr_ready and (crop_mode or detector_ready)

# --------------------------------------------------------------------------- #
# Header - brand on the left, logged-in user and "Log out" on the right
# --------------------------------------------------------------------------- #
brand_col, user_col = st.columns([3, 2], vertical_alignment="center")
with brand_col:
    st.markdown(ui.masthead(), unsafe_allow_html=True)
with user_col:
    who, action = st.columns([3, 1.3], vertical_alignment="center")
    who.markdown(ui.account_block(USER["name"], USER["username"]), unsafe_allow_html=True)
    if action.button("Log out", width="stretch"):
        auth.sign_out()
st.markdown('<div class="nt-rule"></div>', unsafe_allow_html=True)

pipeline, load_error = None, None
if can_run:
    try:
        pipeline = load_pipeline(engine, with_detector=detector_ready)
        if pipeline.detector is not None:     # thresholds change without reloading
            pipeline.detector.conf, pipeline.detector.iou = conf, iou
    except Exception as exc:  # noqa: BLE001 - shown to the user below
        load_error = str(exc)

if load_error:
    st.error(f"The models could not be loaded: {load_error}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def analyse(image_bgr, cache_key: str, source: str, frame: int | None = None):
    """Run the pipeline once per (file, settings) pair and log the readings."""
    key = (cache_key, engine, conf, iou, crop_mode)
    if key in st.session_state.results:
        return st.session_state.results[key]
    output = pipeline.run_on_crop(image_bgr) if crop_mode else pipeline.run(image_bgr)
    st.session_state.results[key] = output
    for p in output.plates:
        st.session_state.log.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "User": USER["username"],
            "Source": source,
            "Frame": frame,
            "Plate": p.reading.formatted,
            "State / UT": p.reading.state or "Unknown",
            "Valid format": p.reading.is_valid,
            "Detection conf.": round(p.detection_confidence, 3),
            "Reading conf.": round(p.ocr_confidence, 3),
            "Raw OCR": p.reading.raw,
        })
    return output


def blocked_message() -> None:
    """Shown in a tab when a file is uploaded but the models are not available."""
    if load_error:
        return
    if engine == "easyocr" and not ocr_ready:
        msg = "Install EasyOCR with <code>pip install easyocr</code>, or switch back to the CRNN reader."
    elif not ocr_ready:
        msg = "Add the trained character reader to the <code>models</code> folder to start reading plates."
    else:
        msg = ("Add the trained plate detector to the <code>models</code> folder, or turn on "
               "<em>Images are already cropped plates</em> in the sidebar to read plate close-ups.")
    st.markdown(ui.empty_state("Recognition is not ready", msg), unsafe_allow_html=True)


def results_table(rows: list[dict]) -> None:
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_image, tab_video, tab_batch, tab_log, tab_model = st.tabs(
    ["Image", "Video", "Batch", "Detection log", "Model"]
)

# ---------------------------------- Image ---------------------------------- #
with tab_image:
    upload = st.file_uploader("Upload a photo of a vehicle", type=["jpg", "jpeg", "png", "bmp", "webp"],
                              key="single")
    if upload is None:
        st.markdown(ui.empty_state(
            "Start with a photo",
            "Drop a picture of a car, bike or truck where the number plate is visible. "
            "Front or rear views both work; the plate should be at least about 80 pixels wide."),
            unsafe_allow_html=True)
    elif pipeline is None:
        blocked_message()
    else:
        image = ui.decode_upload(upload.getvalue())
        if image is None:
            st.error("This file could not be read as an image. Try a JPG or PNG.")
        else:
            with st.spinner("Reading plates"):
                output = analyse(image, upload.file_id, upload.name)

            left, right = st.columns([7, 5], gap="large")
            with left:
                shown = image if crop_mode else draw_results(image, output.plates)
                st.image(ui.to_rgb(shown), width="stretch")
                st.markdown(ui.timings(output.timings_ms), unsafe_allow_html=True)
            with right:
                if not output.plates:
                    st.markdown(ui.empty_state(
                        "No plate found",
                        "Lower the detection threshold in the sidebar, or try a closer, "
                        "sharper photo of the plate."), unsafe_allow_html=True)
                for plate in output.plates:
                    st.markdown(ui.reading_details(plate, show_detection=not crop_mode), unsafe_allow_html=True)

            if show_stages and output.plates:
                with st.expander("What the network sees", expanded=False):
                    st.markdown(
                        '<p class="nt-lede">Each detected plate is cropped with a small margin, '
                        'converted to grayscale, contrast-enhanced with CLAHE and resized to '
                        '32 × 128 pixels. Two-row plates are split and joined into one line '
                        'before reaching the CRNN.</p>', unsafe_allow_html=True)
                    for i, plate in enumerate(output.plates):
                        _, stages = prepare_for_ocr(plate.crop, return_stages=True)
                        cols = st.columns(len(stages) + 1)
                        cols[0].markdown('<div class="nt-stage-label">Detected crop</div>',
                                         unsafe_allow_html=True)
                        cols[0].image(ui.to_rgb(plate.crop), width="stretch")
                        for col, (name, img) in zip(cols[1:], stages.items()):
                            col.markdown(f'<div class="nt-stage-label">{name}</div>', unsafe_allow_html=True)
                            col.image(img, width="stretch")

# ---------------------------------- Video ---------------------------------- #
with tab_video:
    st.markdown('<p class="nt-lede">Upload traffic or dashcam footage. Frames are sampled at the '
                'interval set in the sidebar, and repeated sightings of the same plate are grouped '
                'so each vehicle appears once.</p>', unsafe_allow_html=True)
    video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video")
    if video is not None and pipeline is None:
        blocked_message()
    elif video is not None:
        if st.button("Analyse video", type="primary"):
            with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as tmp:
                tmp.write(video.getvalue())
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            progress = st.progress(0.0, text="Analysing frames")
            sightings: dict[str, dict] = {}
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    out = analyse(frame, f"{video.file_id}-{frame_idx}", video.name, frame_idx)
                    for p in out.plates:
                        if not p.reading.text or (not p.reading.is_valid and p.ocr_confidence < 0.6):
                            continue  # skip partial or garbled reads
                        entry = sightings.setdefault(p.reading.text, {
                            "first": frame_idx / fps, "count": 0, "best": None})
                        entry["count"] += 1
                        if entry["best"] is None or p.overall_confidence > entry["best"].overall_confidence:
                            entry["best"] = p
                    progress.progress(min(frame_idx / total, 1.0),
                                      text=f"Frame {frame_idx:,} of {total:,}")
                frame_idx += 1
            cap.release()
            Path(tmp_path).unlink(missing_ok=True)
            progress.empty()
            st.session_state.video_result = (video.file_id, sightings)

        stored = st.session_state.get("video_result")
        if stored and stored[0] == video.file_id:
            sightings = stored[1]
            if not sightings:
                st.markdown(ui.empty_state("No plates read in this video",
                            "Try analysing more frames (a smaller N) or lowering the detection threshold."),
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<h3 class="nt-section-title">{len(sightings)} vehicles identified</h3>',
                            unsafe_allow_html=True)
                rows = sorted(
                    ({"Plate": e["best"].reading.formatted,
                      "State / UT": e["best"].reading.state or "Unknown",
                      "Valid format": e["best"].reading.is_valid,
                      "First seen (s)": round(e["first"], 1),
                      "Sightings": e["count"],
                      "Best confidence": round(e["best"].overall_confidence, 3)}
                     for e in sightings.values()),
                    key=lambda r: r["First seen (s)"])
                results_table(rows)
                st.download_button("Download CSV", pd.DataFrame(rows).to_csv(index=False),
                                   file_name=f"{Path(video.name).stem}_plates.csv", mime="text/csv")
                cols = st.columns(3)
                for i, entry in enumerate(sorted(sightings.values(), key=lambda e: e["first"])[:12]):
                    with cols[i % 3]:
                        st.image(ui.to_rgb(entry["best"].crop), width="stretch")
                        st.markdown(ui.plate(entry["best"].reading, "sm"), unsafe_allow_html=True)
                        st.write("")

# ---------------------------------- Batch ---------------------------------- #
with tab_batch:
    files = st.file_uploader("Upload several photos", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True, key="batch")
    if files and pipeline is None:
        blocked_message()
    elif files:
        rows, annotated = [], []
        progress = st.progress(0.0, text="Analysing images")
        for i, f in enumerate(files, 1):
            img = ui.decode_upload(f.getvalue())
            if img is None:
                rows.append({"File": f.name, "Plate": "Unreadable file", "State / UT": "",
                             "Valid format": False, "Confidence": 0.0})
                continue
            out = analyse(img, f.file_id, f.name)
            if not out.plates:
                rows.append({"File": f.name, "Plate": "No plate found", "State / UT": "",
                             "Valid format": False, "Confidence": 0.0})
            for p in out.plates:
                rows.append({"File": f.name, "Plate": p.reading.formatted,
                             "State / UT": p.reading.state or "Unknown",
                             "Valid format": p.reading.is_valid,
                             "Confidence": round(p.overall_confidence, 3)})
            annotated.append((f.name, img if crop_mode else draw_results(img, out.plates)))
            progress.progress(i / len(files), text=f"Image {i} of {len(files)}")
        progress.empty()

        found = sum(1 for r in rows if r["Valid format"])
        st.markdown(f'<h3 class="nt-section-title">{found} valid plates in {len(files)} images</h3>',
                    unsafe_allow_html=True)
        results_table(rows)
        st.download_button("Download CSV", pd.DataFrame(rows).to_csv(index=False),
                           file_name="batch_results.csv", mime="text/csv")
        cols = st.columns(3)
        for i, (name, img) in enumerate(annotated):
            cols[i % 3].image(ui.to_rgb(img), caption=name, width="stretch")

# ------------------------------ Detection log ------------------------------ #
with tab_log:
    log = st.session_state.log
    if not log:
        st.markdown(ui.empty_state("Nothing recorded yet",
                    "Every plate read in the Image, Video and Batch tabs is listed here "
                    "for this session, ready to export."), unsafe_allow_html=True)
    else:
        df = pd.DataFrame(log)
        valid = int(df["Valid format"].sum())
        st.markdown(f'<h3 class="nt-section-title">{len(df)} readings, {valid} in a valid Indian format</h3>',
                    unsafe_allow_html=True)
        results_table(log)
        c1, c2, _ = st.columns([1, 1, 4])
        c1.download_button("Download CSV", df.to_csv(index=False), file_name="detection_log.csv",
                           mime="text/csv")
        if c2.button("Clear log"):
            st.session_state.log = []
            st.session_state.results = {}
            st.rerun()

# ---------------------------------- Model ---------------------------------- #
with tab_model:
    st.markdown("""
    <div class="nt-arch">
      <div><h4>Detection</h4><p>YOLOv8 fine-tuned from COCO weights on Kaggle plate images.
        Outputs a box and confidence for every plate.</p></div>
      <div><h4>Crop and enhance</h4><p>Box widened by 6 %, grayscale, CLAHE contrast,
        two-row plates joined into one line, resized to 32 × 128.</p></div>
      <div><h4>Recognition</h4><p>CRNN: six convolution layers, two bidirectional LSTM layers,
        trained with CTC loss on 40,000 rendered plates.</p></div>
      <div><h4>Format check</h4><p>Swaps look-alike characters to fit the Indian layout
        and looks up the state or union territory.</p></div>
    </div>
    """, unsafe_allow_html=True)

    results_csv = resolve(CFG["paths"]["outputs"]) / "detector" / "train" / "results.csv"
    sections = []
    if OCR_WEIGHTS.exists():
        sections.append("reader")
    if results_csv.exists():
        sections.append("detector")
    columns = st.columns(2, gap="large") if sections else []

    for column, section in zip(columns, sections):
        with column:
            if section == "reader":
                st.markdown('<h3 class="nt-section-title">Character reader</h3>', unsafe_allow_html=True)
                import torch
                ckpt = torch.load(OCR_WEIGHTS, map_location="cpu")
                m = ckpt.get("metrics", {})
                params = sum(v.numel() for k, v in ckpt["model"].items()
                             if v.dtype.is_floating_point and "running_" not in k)
                results_table([
                    {"Metric": "Best epoch", "Value": str(m.get("epoch", "—"))},
                    {"Metric": "Plate accuracy (validation)", "Value": f"{m.get('plate_acc', 0) * 100:.2f} %"},
                    {"Metric": "With format correction", "Value": f"{m.get('plate_acc_corrected', 0) * 100:.2f} %"},
                    {"Metric": "Character error rate", "Value": f"{m.get('cer', 0) * 100:.2f} %"},
                    {"Metric": "Parameters", "Value": f"{params:,}"},
                ])
                curves = resolve(CFG["paths"]["outputs"]) / "ocr" / "training_curves.png"
                if curves.exists():
                    st.image(str(curves), width="stretch")
            else:
                st.markdown('<h3 class="nt-section-title">Plate detector</h3>', unsafe_allow_html=True)
                hist = pd.read_csv(results_csv)
                hist.columns = [c.strip() for c in hist.columns]
                best = hist.loc[hist["metrics/mAP50(B)"].idxmax()]
                results_table([
                    {"Metric": "Best epoch", "Value": str(int(best["epoch"]))},
                    {"Metric": "mAP@0.5", "Value": f"{best['metrics/mAP50(B)']:.3f}"},
                    {"Metric": "mAP@0.5:0.95", "Value": f"{best['metrics/mAP50-95(B)']:.3f}"},
                    {"Metric": "Precision", "Value": f"{best['metrics/precision(B)']:.3f}"},
                    {"Metric": "Recall", "Value": f"{best['metrics/recall(B)']:.3f}"},
                ])
                plot = results_csv.parent / "results.png"
                if plot.exists():
                    st.image(str(plot), width="stretch")
