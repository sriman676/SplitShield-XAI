"""Shrink the project to a pendrive copy (owner 2026-10-15).

Keeps what is needed to RUN the SOC and read the results: code, trained models
(artifacts/), results and reports, Sigma rules, the ATT&CK knowledge file, the
evolution benchmark, installers and docs, plus the small per-run metric JSONs of
the research runs. Removes datasets, the Python environment (INSTALL.cmd rebuilds
it), heavy run files, caches, logs and agent worktrees.

    python -m scripts.make_portable            # DRY RUN: lists what would go and stay
    python -m scripts.make_portable --delete   # actually deletes (asks to type YES)

Never runs on its own; deletion needs the owner's explicit confirmation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# artifacts/mlareas/insider/data stays: the insider model's updates need it (23 MB)
REMOVE_DIRS = [
    "data", "datasets", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", "logs",
    "artifacts/logs", ".claude/worktrees", "artifacts/controller/snapshots",
]  # fmt: skip
KEEP_IN_RUNS = ("metrics", "config.yaml", "manifest.json", "config.json")
# small folders inside a removed folder that must survive: the test fixtures (28 KB)
# - the update agent and the controller run the full test suite as their gate, and
# 12 tests read data/fixtures (found by a trial portable copy, 2026-10-15)
KEEP_INSIDE = ("data/fixtures",)


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def plan() -> tuple[list[Path], int, int]:
    remove: list[Path] = []
    for d in REMOVE_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        keep = [ROOT / k for k in KEEP_INSIDE if (ROOT / k).is_relative_to(p)]
        if keep:  # remove the folder's other children, not the kept ones
            remove += [c for c in p.iterdir() if not any(k.is_relative_to(c) for k in keep)]
        else:
            remove.append(p)
    remove += [p for p in ROOT.rglob("__pycache__") if ".venv" not in p.parts]
    runs = ROOT / "runs"
    if runs.is_dir():  # keep each run's metric JSONs + config, drop models/arrays
        for f in runs.rglob("*"):
            if f.is_file() and not any(
                k in f.relative_to(runs).parts or f.name == k for k in KEEP_IN_RUNS
            ):
                remove.append(f)
    total = _size(ROOT)
    gone = sum(_size(p) for p in remove)
    return remove, total, gone


RERUN_COPY = Path(r"D:\SplitShield-rerun")  # isolated full re-run (2026-10-16)


def rerun_busy(copy: Path = RERUN_COPY) -> bool:
    """True while the re-run copy still reads this project's datasets."""
    return copy.is_dir() and not (copy / "RERUN_DONE.json").is_file()


def remove_rerun_copy(copy: Path = RERUN_COPY, keep_in: Path = ROOT / "docs") -> None:
    """Keep its report, unlink its dataset junctions (never follow them), delete it."""
    for name in ("REPRODUCIBILITY.md", "RERUN_DONE.json"):
        if (copy / name).is_file():
            shutil.copy2(copy / name, keep_in / f"rerun_{name}")
    for link in (copy / "data" / "raw", copy / "data" / "blind", copy / "data" / "attack_data_dev"):
        if link.is_junction() or link.is_symlink():
            link.rmdir() if link.is_junction() else link.unlink()
    shutil.rmtree(copy, ignore_errors=True)  # removes, never follows, other junctions


def stop_live_watchers() -> None:
    """End the live-log watcher task (it runs pythonw from .venv; a running program's
    files cannot be deleted on Windows). Its logon trigger stays: after the next
    INSTALL.cmd it starts again by itself."""
    import subprocess

    subprocess.run(
        ["schtasks", "/End", "/TN", "SplitShield-XAI live logs"],
        capture_output=True,
        check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true")
    ap.add_argument(
        "--with-rerun-copy",
        action="store_true",
        help=f"also delete the re-run copy {RERUN_COPY} (its report is kept in docs/)",
    )
    a = ap.parse_args()
    if a.delete and rerun_busy():
        raise SystemExit(f"the full re-run in {RERUN_COPY} is still running: wait")
    remove, total, gone = plan()
    print(
        f"project now: {total / 2**30:.1f} GB -> after cleanup: {(total - gone) / 2**30:.2f} GB"
    )
    top: dict[str, int] = {}
    for p in remove:
        key = p.relative_to(ROOT).parts[0]
        top[key] = top.get(key, 0) + _size(p)
    for k, v in sorted(top.items(), key=lambda kv: -kv[1]):
        print(f"  remove {k:<28} {v / 2**20:>10,.0f} MB")
    if a.with_rerun_copy and RERUN_COPY.is_dir():
        print(f"  remove {str(RERUN_COPY):<28} (re-run copy; report kept in docs/)")
    if not a.delete:
        print("dry run only - nothing deleted (use --delete after the owner approves)")
        return
    if input("Type YES to delete the folders listed above: ").strip() != "YES":
        print("cancelled")
        return
    stop_live_watchers()  # they run from .venv: stop them before it is deleted
    if a.with_rerun_copy and RERUN_COPY.is_dir():
        remove_rerun_copy()
    for p in sorted(remove, key=lambda x: len(x.parts)):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    print(f"done: {_size(ROOT) / 2**30:.2f} GB left")


if __name__ == "__main__":
    main()
