"""
netra.ocr.charset
=================

Defines the alphabet the CRNN can output and converts between text and the
integer label sequences used by CTC (Connectionist Temporal Classification).

CTC needs one extra symbol, the *blank*, which the network emits between
characters and to separate repeated letters (e.g. "11" -> "1 _ 1").
We reserve index 0 for the blank, so real characters start at index 1.
"""

from __future__ import annotations

import torch

ALPHABET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK: int = 0
NUM_CLASSES: int = len(ALPHABET) + 1          # +1 for the CTC blank

CHAR_TO_INDEX: dict[str, int] = {c: i + 1 for i, c in enumerate(ALPHABET)}
INDEX_TO_CHAR: dict[int, str] = {i + 1: c for i, c in enumerate(ALPHABET)}


def encode(text: str) -> list[int]:
    """'MH12' -> [23, 18, 2, 3]. Unknown characters are skipped."""
    return [CHAR_TO_INDEX[c] for c in text.upper() if c in CHAR_TO_INDEX]


def greedy_decode(log_probs: torch.Tensor) -> list[tuple[str, float]]:
    """
    Best-path CTC decoding.

    Args:
        log_probs: tensor of shape (T, B, C) - time steps, batch, classes.

    Returns:
        A list with one ``(text, confidence)`` tuple per batch item.
        Confidence is the mean probability of the characters that were kept.

    How it works: take the most likely class at every time step, collapse
    consecutive repeats, then remove blanks.
    """
    probs = log_probs.exp()
    best_prob, best_idx = probs.max(dim=2)              # (T, B)
    results = []
    for b in range(best_idx.shape[1]):
        chars, confs, prev = [], [], BLANK
        for t in range(best_idx.shape[0]):
            idx = int(best_idx[t, b])
            if idx != BLANK and idx != prev:
                chars.append(INDEX_TO_CHAR[idx])
                confs.append(float(best_prob[t, b]))
            prev = idx
        conf = sum(confs) / len(confs) if confs else 0.0
        results.append(("".join(chars), conf))
    return results
