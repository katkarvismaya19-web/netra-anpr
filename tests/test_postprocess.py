"""
Unit tests for the Indian plate format corrector.

Run with:  pytest -q
"""

import pytest

from netra.postprocess import clean, correct_plate, is_valid_plate


@pytest.mark.parametrize(
    "raw, expected, formatted",
    [
        ("MH12AB1234", "MH12AB1234", "MH 12 AB 1234"),    # already correct
        ("mh 12 ab 1234", "MH12AB1234", "MH 12 AB 1234"),  # lower case + spaces
        ("MH12AB12B4", "MH12AB1284", "MH 12 AB 1284"),    # B misread in digit slot
        ("0L3CAB1234", "DL3CAB1234", "DL 3 CAB 1234"),    # 0 misread in letter slot
        ("KA01M1234", "KA01M1234", "KA 01 M 1234"),       # single-letter series
        ("TN1OAB5678", "TN10AB5678", "TN 10 AB 5678"),    # O instead of 0 in district
        ("22BH1234AA", "22BH1234AA", "22 BH 1234 AA"),    # Bharat series
        ("INDMH12AB1234", "MH12AB1234", "MH 12 AB 1234"), # stray IND strip text
    ],
)
def test_correct_plate(raw, expected, formatted):
    reading = correct_plate(raw)
    assert reading.text == expected
    assert reading.formatted == formatted
    assert reading.is_valid


def test_state_lookup():
    assert correct_plate("MH12AB1234").state == "Maharashtra"
    assert correct_plate("22BH1234AA").state.startswith("Bharat")


def test_invalid_text_is_flagged():
    reading = correct_plate("HELLO")
    assert not reading.is_valid


def test_clean_and_validate():
    assert clean(" ka-05 mn.9999 ") == "KA05MN9999"
    assert is_valid_plate("KA05MN9999")
    assert not is_valid_plate("XX05MN9999")  # unknown state code
