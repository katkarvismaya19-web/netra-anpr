"""
Login page  (URL: /login)
=========================

The first page every visitor sees. On a correct username and password the
user is stored in the session and sent to /dashboard. Signed-in users who
open /login are forwarded to the dashboard straight away.

Security
--------
* Passwords are checked against salted PBKDF2-SHA256 hashes (see auth.py).
* After 5 wrong attempts the form is locked for 60 seconds.
* The error message never says whether the username or the password was
  wrong, so it cannot be used to discover valid usernames.
"""

from __future__ import annotations

import sys
import time
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

import auth  # noqa: E402

# Already signed in? Skip the form.
if auth.current_user() is not None:
    st.switch_page(auth.DASHBOARD_PAGE)

_, is_demo = auth.load_users()
st.session_state.setdefault("auth_attempts", 0)
st.session_state.setdefault("auth_locked_until", 0.0)

# The login page has no sidebar and a narrower, centred layout.
st.markdown(
    '<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"]'
    '{display:none !important;} .block-container{max-width:1120px; padding-top:8vh;}</style>',
    unsafe_allow_html=True,
)

intro, form_col = st.columns([1.15, 1], gap="large")

with intro:
    st.markdown(f"""
    <div class="nt-login__intro">
      <div class="nt-brand">
        <div class="nt-mark" aria-hidden="true"><span></span></div>
        <div><h1>Netra</h1><p>Number plate recognition for Indian vehicles</p></div>
      </div>
      <div class="nt-plate nt-plate--lg nt-login__plate">
        <div class="nt-plate__strip"><span class="nt-plate__chakra"></span><b>IND</b></div>
        <div class="nt-plate__text">{escape("MH 01 AB 1234")}</div>
      </div>
      <p class="nt-login__lede">Detects number plates in photos and traffic video,
      reads the registration number and identifies the state it was issued in.</p>
      <dl class="nt-login__facts">
        <div><dt>Detector</dt><dd>YOLOv8, fine-tuned on Kaggle plate images</dd></div>
        <div><dt>Reader</dt><dd>CRNN with CTC decoding</dd></div>
        <div><dt>Formats</dt><dd>Standard, two-row and Bharat series plates</dd></div>
      </dl>
    </div>
    """, unsafe_allow_html=True)

with form_col:
    st.markdown('<div class="nt-login__card-head"><h2>Log in</h2>'
                '<p>Use the account given to you by the system administrator.</p></div>',
                unsafe_allow_html=True)

    locked_for = st.session_state.auth_locked_until - time.time()
    with st.form("log_in", border=True):
        username = st.text_input("Username", autocomplete="username", disabled=locked_for > 0)
        password = st.text_input("Password", type="password", autocomplete="current-password",
                                 disabled=locked_for > 0)
        submitted = st.form_submit_button("Log in", type="primary", width="stretch",
                                          disabled=locked_for > 0)

    if locked_for > 0:
        st.error(f"Too many incorrect attempts. Try again in {int(locked_for) + 1} seconds.")
    elif submitted:
        if auth.sign_in(username, password):
            st.session_state.auth_attempts = 0
            st.switch_page(auth.DASHBOARD_PAGE)
        st.session_state.auth_attempts += 1
        left = auth.MAX_ATTEMPTS - st.session_state.auth_attempts
        if left <= 0:
            st.session_state.auth_locked_until = time.time() + auth.LOCKOUT_SECONDS
            st.session_state.auth_attempts = 0
            st.error(f"Too many incorrect attempts. The form is locked for {auth.LOCKOUT_SECONDS} seconds.")
        else:
            st.error(f"Username or password is incorrect. {left} attempt{'s' if left != 1 else ''} left.")

    if is_demo:
        st.markdown('<p class="nt-login__note">Demo mode: no accounts are configured yet. '
                    'Run <code>python scripts/create_user.py &lt;username&gt;</code> to add one.</p>',
                    unsafe_allow_html=True)
