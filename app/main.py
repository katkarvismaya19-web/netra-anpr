"""
Netra web app - entry point
===========================

Run from the project root:

    streamlit run app/main.py

The app has two separate pages, each with its own URL:

    /login       views/login.py       log-in form (default page)
    /dashboard   views/dashboard.py   recognition dashboard (signed-in users only)

This file only sets up what both pages share - page settings and the
stylesheet - and hands control to the page in the address bar. Access
control lives in the pages themselves: the dashboard redirects anyone who
is not signed in back to /login.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import components as ui  # noqa: E402

st.set_page_config(
    page_title="Netra | Number plate recognition",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.load_css(APP_DIR / "theme.css"), unsafe_allow_html=True)

login_page = st.Page("views/login.py", title="Log in", url_path="login", default=True)
dashboard_page = st.Page("views/dashboard.py", title="Dashboard", url_path="dashboard")

# position="hidden": no page menu - users move between pages by signing in and out.
st.navigation([login_page, dashboard_page], position="hidden").run()
