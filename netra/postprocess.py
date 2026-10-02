"""
netra.postprocess
=================

Turns the raw text produced by the OCR model into a clean, validated Indian
registration number.

Why is this needed?
    Even a good OCR network confuses characters that look alike - ``O`` and
    ``0``, ``I`` and ``1``, ``B`` and ``8``, ``S`` and ``5``. Indian plates,
    however, follow a strict layout, so we always know whether a given
    position *should* hold a letter or a digit:

        Standard series :  MH 12 AB 1234
                           |  |  |  +-- 1-4 digits   (unique number)
                           |  |  +----- 0-3 letters  (series)
                           |  +-------- 1-2 digits   (RTO district code)
                           +----------- 2 letters    (state / UT code)

        Bharat series   :  22 BH 1234 AA
                           (year) BH (4 digits) (1-2 letters)

    The function :func:`correct_plate` tries every way the raw string could
    be split into this layout, swaps look-alike characters where a position
    demands the other type, and keeps the split that needs the fewest and
    most plausible changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from netra.states import STATE_CODES, state_name

# Characters a digit is commonly misread as (and vice versa).
LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "J": "1", "T": "1",
    "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "E": "8",
}
DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S",
    "6": "G", "7": "T", "8": "B", "3": "B", "9": "P",
}
# Extra letter guesses tried only for the state code, where the list of valid
# codes lets us pick the right one (e.g. "0L" -> "DL", "1K" -> "UK").
STATE_ALTERNATIVES: dict[str, str] = {
    "0": "ODQ", "1": "ITJL", "2": "Z", "4": "A", "5": "S",
    "6": "G", "7": "T", "8": "B", "3": "B", "9": "P",
    "O": "D", "D": "O", "I": "T", "Q": "O",
}

STANDARD_PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$")
BHARAT_PATTERN = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")


@dataclass
class PlateReading:
    """Result of post-processing one OCR string."""

    raw: str                 # Text exactly as the OCR model produced it
    text: str                # Corrected plate without spaces, e.g. MH12AB1234
    formatted: str           # Human-friendly form, e.g. MH 12 AB 1234
    is_valid: bool           # True when text matches an official layout
    state: str | None        # Registered state / UT, if the code is known
    corrections: int         # How many characters were swapped


def clean(text: str) -> str:
    """Upper-case the text and drop everything that is not A-Z or 0-9."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _coerce(ch: str, want_digit: bool) -> tuple[str, int] | None:
    """
    Force one character to be a digit or a letter.

    Returns ``(new_char, cost)`` where cost is 0 if nothing changed and 1 if a
    look-alike swap was needed, or ``None`` if no sensible swap exists.
    """
    if want_digit:
        if ch.isdigit():
            return ch, 0
        return (LETTER_TO_DIGIT[ch], 1) if ch in LETTER_TO_DIGIT else None
    if ch.isalpha():
        return ch, 0
    return (DIGIT_TO_LETTER[ch], 1) if ch in DIGIT_TO_LETTER else None


def _fit(text: str, layout: list[tuple[int, bool]]) -> tuple[str, int] | None:
    """
    Fit ``text`` onto a layout such as ``[(2, False), (2, True), ...]``
    (segment length, is-digit). Returns the coerced string and total cost.
    """
    out, cost, i = [], 0, 0
    for length, want_digit in layout:
        for _ in range(length):
            fixed = _coerce(text[i], want_digit)
            if fixed is None:
                return None
            out.append(fixed[0])
            cost += fixed[1]
            i += 1
    return "".join(out), cost


def _best_state(code: str) -> tuple[str, int] | None:
    """
    Choose the most likely valid state code for a two-character string.

    Returns ``(code, cost)`` where cost counts swapped characters, or None.
    """
    first = [(code[0], 0)] + [(c, 1) for c in STATE_ALTERNATIVES.get(code[0], "")]
    second = [(code[1], 0)] + [(c, 1) for c in STATE_ALTERNATIVES.get(code[1], "")]
    options = [(a + b, ca + cb) for a, ca in first for b, cb in second
               if (a + b).isalpha() and a + b in STATE_CODES]
    return min(options, key=lambda o: o[1]) if options else None


def _candidates(text: str):
    """
    Yield ``(corrected_text, score, layout_kind, segment_lengths)`` for every
    official layout the text can be bent into. Lower score = better guess.
    """
    n = len(text)

    # Standard series: 2 letters | 1-2 digits | 0-3 letters | 1-4 digits
    if n < 4:
        return
    state = _best_state(text[:2])            # valid code via look-alike swaps
    if state is None:                         # unknown code: plain letter coercion
        plain = _fit(text[:2], [(2, False)])
        state = (plain[0], plain[1] + 1.2) if plain else None  # +12 score penalty below
    if state is None:
        return
    state_code, state_cost = state
    for d in (2, 1):
        for s in (0, 1, 2, 3):
            num = n - 2 - d - s
            if not 1 <= num <= 4:
                continue
            fitted = _fit(text[2:], [(d, True), (s, False), (num, True)])
            if fitted is None:
                continue
            rest, cost = fitted
            fixed = state_code + rest
            score = (cost + state_cost) * 10
            score += 0 if num == 4 else (4 - num) * 3   # modern plates use 4 digits
            score += 0 if d == 2 else 1                 # most RTO codes are 2 digits
            score += 12 * sum(c in "IO" for c in rest[d:d + s])  # I/O are never issued in series
            yield fixed, score, "standard", (2, d, s, num)

    # Bharat (BH) series: 2 digits | BH | 4 digits | 1-2 letters
    for tail in (2, 1):
        if n != 2 + 2 + 4 + tail:
            continue
        layout = [(2, True), (2, False), (4, True), (tail, False)]
        fitted = _fit(text, layout)
        if fitted is None or fitted[0][2:4] != "BH":
            continue
        yield fitted[0], fitted[1] * 10, "bharat", (2, 2, 4, tail)


def _format(text: str, kind: str, parts: tuple[int, ...]) -> str:
    """Insert spaces between the logical groups of the plate."""
    groups, i = [], 0
    for length in parts:
        if length:
            groups.append(text[i:i + length])
        i += length
    return " ".join(groups)


def correct_plate(raw_text: str) -> PlateReading:
    """
    Clean, correct and validate one OCR string.

    If the text cannot be matched to any official layout it is returned
    unchanged with ``is_valid=False`` - the dashboard still shows it so the
    user can judge the reading themselves.
    """
    text = clean(raw_text)
    best = None

    # OCR sometimes picks up stray marks or the "IND" strip at the edges,
    # so we also try trimmed versions (with a small penalty per dropped char).
    for start in range(len(text)):
        for end in range(len(text), start + 5, -1):
            sub = text[start:end]
            dropped = len(text) - len(sub)
            for fixed, score, kind, parts in _candidates(sub):
                total = score + dropped * 6
                if best is None or total < best[1]:
                    best = (fixed, total, kind, parts)

    if best is None:
        return PlateReading(raw_text, text, text, False, state_name(text), 0)

    fixed, _, kind, parts = best
    corrections = sum(a != b for a, b in zip(fixed, text)) if len(fixed) == len(text) else None
    if corrections is None:
        corrections = abs(len(text) - len(fixed))

    state = "Bharat series (all-India)" if kind == "bharat" else state_name(fixed)
    return PlateReading(
        raw=raw_text,
        text=fixed,
        formatted=_format(fixed, kind, parts),
        is_valid=is_valid_plate(fixed),
        state=state,
        corrections=corrections,
    )


def is_valid_plate(text: str) -> bool:
    """True when ``text`` (no spaces) matches a standard or BH layout."""
    text = clean(text)
    if BHARAT_PATTERN.match(text):
        return True
    return bool(STANDARD_PATTERN.match(text)) and text[:2] in STATE_CODES
