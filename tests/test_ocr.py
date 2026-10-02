"""
Shape and round-trip tests for the OCR components (no trained weights needed).
"""

import random

import numpy as np
import torch

from netra.metrics import character_error_rate, edit_distance
from netra.ocr.charset import NUM_CLASSES, encode, greedy_decode
from netra.ocr.model import CRNN
from netra.ocr.synth import random_plate_text, render_plate
from netra.postprocess import is_valid_plate
from netra.preprocess import is_two_line, prepare_for_ocr


def test_crnn_output_shape():
    model = CRNN().eval()
    out = model(torch.rand(4, 1, 32, 128))
    assert out.shape == (32, 4, NUM_CLASSES)        # (T, B, classes)
    assert torch.allclose(out.exp().sum(2), torch.ones(32, 4), atol=1e-4)


def test_greedy_decode_collapses_repeats_and_blanks():
    # Sequence "M M _ H _ 1 1" should decode to "MH1"
    ids = [encode("M")[0], encode("M")[0], 0, encode("H")[0], 0, encode("1")[0], encode("1")[0]]
    log_probs = torch.full((len(ids), 1, NUM_CLASSES), -20.0)
    for t, i in enumerate(ids):
        log_probs[t, 0, i] = 0.0
    assert greedy_decode(log_probs)[0][0] == "MH1"


def test_preprocess_output():
    crop = np.random.randint(0, 255, (60, 240, 3), np.uint8)
    x = prepare_for_ocr(crop)
    assert x.shape == (32, 128) and x.dtype == np.float32 and 0 <= x.min() <= x.max() <= 1


def test_two_line_detection():
    assert is_two_line(np.zeros((150, 220), np.uint8))
    assert not is_two_line(np.zeros((100, 480), np.uint8))


def test_synthetic_labels_are_valid():
    rng = random.Random(0)
    for _ in range(200):
        text, _ = random_plate_text(rng)
        assert is_valid_plate(text), text


def test_render_plate_without_fonts():
    img, label = render_plate(random.Random(1), fonts=[])
    assert img.ndim == 3 and img.dtype == np.uint8 and label


def test_metrics():
    assert edit_distance("MH12AB1234", "MH12AB1234") == 0
    assert edit_distance("MH12AB1234", "MH12A81234") == 1
    assert character_error_rate(["MH12"], ["MH13"]) == 0.25
