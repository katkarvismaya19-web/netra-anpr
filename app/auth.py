"""
app.auth
========

Password hashing, account lookup and session helpers shared by the login
page (``views/login.py``) and the dashboard page (``views/dashboard.py``).

How accounts are stored
-----------------------
Passwords are never saved in plain text. Each one is hashed with
PBKDF2-HMAC-SHA256 (200,000 iterations, random 16-byte salt) and stored as::

    pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>

Accounts live in ``.streamlit/secrets.toml`` (ignored by git)::

    [auth.users.admin]
    name = "Traffic Control Room"
    password = "pbkdf2_sha256$200000$..."

Create or update an account with::

    python scripts/create_user.py admin

If no accounts are configured, a single demo account is available so the
project works straight after cloning: username ``admin``, password
``netra@2026``. Create your own account before deploying anywhere public.

Brute-force protection: after 5 wrong passwords the form is locked for
60 seconds (per browser session).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import streamlit as st

ITERATIONS = 200_000
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60

# Page files, relative to the entry point app/main.py.
LOGIN_PAGE = "views/login.py"
DASHBOARD_PAGE = "views/dashboard.py"

# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
def hash_password(password: str, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    """Return a salted PBKDF2 hash string for ``password``."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check ``password`` against a stored hash in constant time."""
    try:
        algorithm, iterations, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt_hex), int(iterations))
    return hmac.compare_digest(digest.hex(), hash_hex)


# --------------------------------------------------------------------------- #
# Account lookup
# --------------------------------------------------------------------------- #
def _demo_users() -> dict:
    """Build the demo account hash once (deterministic salt, fixed password)."""
    salt = b"netra-demo-salt"
    return {"admin": {"name": "Demo administrator",
                      "password": hash_password("netra@2026", salt=salt)}}


def load_users() -> tuple[dict, bool]:
    """
    Return ``(users, is_demo)``. Users come from secrets.toml when present,
    otherwise the demo account is used.
    """
    try:
        users = st.secrets["auth"]["users"]
        if users:
            return {k: dict(v) for k, v in users.items()}, False
    except Exception:  # noqa: BLE001 - no secrets file or no [auth] section
        pass
    return _demo_users(), True


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def current_user() -> dict | None:
    """The signed-in user for this browser session, or None."""
    return st.session_state.get("auth_user")


def sign_in(username: str, password: str) -> bool:
    """Check credentials and store the user in the session if they are correct."""
    users, _ = load_users()
    username = username.strip().lower()
    record = users.get(username)
    if record and verify_password(password, record.get("password", "")):
        st.session_state.auth_user = {"username": username, "name": record.get("name", username)}
        return True
    return False


def sign_out() -> None:
    """Forget the user and everything they analysed, then go back to /login."""
    for key in ("auth_user", "log", "results", "video_result"):
        st.session_state.pop(key, None)
    st.switch_page(LOGIN_PAGE)
