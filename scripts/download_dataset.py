"""
Step 1 - Download the number plate datasets from Kaggle
=======================================================

Downloads every dataset listed under ``datasets:`` in config.yaml into
``data/raw/<dataset-name>/``.

Authentication (one-time setup)
-------------------------------
1. Log in to kaggle.com -> Settings -> API -> "Create New Token".
   This downloads ``kaggle.json``.
2. Put it in ``~/.kaggle/kaggle.json`` (Windows: ``C:\\Users\\<you>\\.kaggle\\``)
   or set the environment variables ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.

Usage
-----
    python scripts/download_dataset.py
    python scripts/download_dataset.py --only andrewmvd/car-plate-detection

If an automatic download fails, download the zip from the Kaggle website and
extract it into ``data/raw/<dataset-name>/`` - the next script will find it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netra.config import ensure_dir, load_config  # noqa: E402


def download_with_kagglehub(slug: str, target: Path) -> bool:
    """Preferred method: the official ``kagglehub`` package (caches downloads)."""
    try:
        import kagglehub
    except ImportError:
        return False
    try:
        cached = Path(kagglehub.dataset_download(slug))
    except Exception as exc:  # noqa: BLE001
        print(f"  kagglehub failed: {exc}")
        return False
    shutil.copytree(cached, target, dirs_exist_ok=True)
    return True


def download_with_cli(slug: str, target: Path) -> bool:
    """Fallback: the classic ``kaggle`` command-line tool."""
    if shutil.which("kaggle") is None:
        return False
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"]
    return subprocess.run(cmd, check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="Download a single dataset slug instead of all of them")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = ensure_dir(cfg["paths"]["raw_data"])
    slugs = [args.only] if args.only else [d["slug"] for d in cfg["datasets"]]

    failed = []
    for slug in slugs:
        target = raw_dir / slug.split("/")[-1]
        if target.exists() and any(target.rglob("*.*")):
            print(f"[skip] {slug} already present in {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"[download] {slug} -> {target}")
        if download_with_kagglehub(slug, target) or download_with_cli(slug, target):
            n_images = sum(1 for p in target.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            print(f"  done - {n_images} images")
        else:
            failed.append(slug)

    if failed:
        print("\nSome downloads failed. Install `kagglehub` (pip install kagglehub), check your")
        print("Kaggle API token, or download manually from:")
        for slug in failed:
            print(f"  https://www.kaggle.com/datasets/{slug}")
        print(f"and extract into {raw_dir}/<dataset-name>/")
        sys.exit(1)


if __name__ == "__main__":
    main()
