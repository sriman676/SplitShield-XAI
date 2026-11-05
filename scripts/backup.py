"""Daily backup to another drive (owner decision 2026-10-14: D:\\SplitShield-backups).

Copies everything that cannot be re-downloaded - models (artifacts/), results/,
runs/ (every experiment's metrics and outputs), logs/, configs, code, docs - into
``<DEST>/<YYYY-MM-DD>/``; keeps the newest ``KEEP`` days. Not copied: data/ (the
raw datasets, 23 GB, restorable from the zips) and .venv/ (rebuilt by setup).
Read-only on the project: it never modifies or deletes project files. Only old
backup folders created by this script (named YYYY-MM-DD under DEST) are removed.

    python -m scripts.backup [--dest D:\\SplitShield-backups]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(r"D:\SplitShield-backups")
KEEP = 7
DIRS = ("artifacts", "results", "runs", "logs", "config", "src", "scripts", "models",
        "planning", "tests", "utils", ".streamlit")  # fmt: skip
TOP = (
    "*.py",
    "*.md",
    "*.cmd",
    "*.ps1",
    "*.toml",
    "*.yaml",
    "*.txt",
    ".env",
    ".gitignore",
)
SKIP = shutil.ignore_patterns("__pycache__", "*.tmp", "*.lock")
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def backup(dest: Path = DEST, root: Path = ROOT, keep: int = KEEP) -> Path:
    target = dest / time.strftime("%Y-%m-%d")
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # an interrupted earlier attempt of this script
    partial.mkdir(parents=True)
    for d in DIRS:
        if (root / d).is_dir():
            shutil.copytree(root / d, partial / d, ignore=SKIP)
    for pat in TOP:
        for f in root.glob(pat):
            if f.is_file():
                shutil.copy2(f, partial / f.name)
    if target.exists():  # second backup the same day replaces the first
        shutil.rmtree(target)
    partial.rename(target)  # complete backups only ever carry a date name
    olds = sorted(
        (p for p in dest.iterdir() if p.is_dir() and DAY.match(p.name)), reverse=True
    )
    for old in olds[keep:]:
        shutil.rmtree(old)
    return target


def destination() -> Path:
    """Where backups go: the dashboard setting (Getting started), else ``DEST``."""
    try:
        from src.soc import sinks

        dest = sinks.load_settings().get("backup_dest")
    except Exception:  # noqa: BLE001 - settings unreadable: fall back
        dest = None
    return Path(dest) if dest else DEST


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=None)
    a = ap.parse_args()
    t = backup(a.dest or destination())
    size = sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
    print(f"[backup] {t} ({size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
