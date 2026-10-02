"""
netra.config
============

Loads ``config.yaml`` once and exposes it as a plain dictionary.

Why a helper instead of reading YAML everywhere?
    * All paths in the YAML are written relative to the project root.
      ``resolve()`` turns them into absolute paths, so scripts work no matter
      which folder you launch them from (terminal, Colab, Streamlit).
    * The file is cached, so repeated calls are free.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = the folder that contains config.yaml (one level above /netra).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_FILE: Path = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Read config.yaml and return it as a nested dictionary."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {CONFIG_FILE}. "
            "Make sure you are running from inside the project folder."
        )
    with CONFIG_FILE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative_path: str | Path) -> Path:
    """Convert a path from config.yaml into an absolute path inside the project."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(relative_path: str | Path) -> Path:
    """Resolve a directory path and create it if it does not exist yet."""
    path = resolve(relative_path)
    path.mkdir(parents=True, exist_ok=True)
    return path
