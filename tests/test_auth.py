"""Password hashing used by the dashboard sign-in page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from auth import hash_password, verify_password  # noqa: E402


def test_hash_round_trip():
    stored = hash_password("correct horse")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong", stored)


def test_salts_are_random():
    assert hash_password("same") != hash_password("same")


def test_malformed_hash_is_rejected():
    assert not verify_password("anything", "not-a-hash")
