"""Make an isolated copy of the project for a full re-run (2026-10-17).

    python -m scripts.make_rerun_copy --dest D:/SplitShield-rerun
    python -m scripts.make_rerun_copy --dest D:/SplitShield-rerun2

* Copies the committed project (the working tree must be clean), without the Python
  environment, git history, caches, logs, runs, results, raw archives or the live SOC
  state; ``data/processed`` is copied so caches can be rebuilt without touching the
  original.
* ``data/raw`` and ``data/blind`` become JUNCTIONS to the real datasets (read-only by
  convention; never delete through them - scripts.make_portable.remove_rerun_copy
  unlinks them first).
* Writes ``COPY_INFO.json``: git commit, time, and the SHA-256 of every research file
  (ai_engineer.py, run_pipeline.py, deploy.py, src/{data,training,models,eval,attack,
  defense,explain}, config/). ``verify`` later proves they never changed.

    python -m scripts.make_rerun_copy --verify D:/SplitShield-rerun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".git", ".claude", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
        "datasets", "runs", "results", "logs"}  # fmt: skip
SKIP_REL = {"data/raw", "data/blind", "data/attack_data_dev", "artifacts/soc",
            "artifacts/controller/snapshots", "artifacts/logs",
            # run state of the original: a copy must start from nothing (2026-10-17)
            "artifacts/ai_engineer_state.json", "artifacts/full_rerun_state.json",
            "artifacts/queued_phases.json", "artifacts/watchdog_state.json",
            "RERUN_DONE.json", "RESEARCH_DONE.json"}  # fmt: skip
JUNCTIONS = ("data/raw", "data/blind", "data/attack_data_dev")  # dev data: non-blind Splunk follow-up
SOC_CONFIG = {"models.yaml", "places.yaml"}  # SOC settings, not research configuration
RESEARCH = ["ai_engineer.py", "run_pipeline.py", "run_matrix.py", "deploy.py", "src/data", "src/training",
            "src/models", "src/eval", "src/attack", "src/defense", "src/explain", "config"]  # fmt: skip


def research_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in RESEARCH:
        p = root / rel
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
        files = [f for f in files if f.name not in SOC_CONFIG]
        for f in files:
            out[f.relative_to(root).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _ignore(src: str, names: list[str]) -> set[str]:
    rel = Path(src).resolve().relative_to(ROOT).as_posix()
    rel = "" if rel == "." else rel
    out = {n for n in names if n in SKIP}
    out |= {n for n in names if (f"{rel}/{n}" if rel else n) in SKIP_REL}
    return out


def make(dest: Path) -> dict[str, Any]:
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        raise SystemExit("commit first: the copy must be exactly a commit")
    if dest.exists():
        raise SystemExit(f"{dest} exists - remove it with scripts.make_portable.remove_rerun_copy first")
    shutil.copytree(ROOT, dest, ignore=_ignore)
    for rel in JUNCTIONS:
        target, link = ROOT / rel, dest / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info = {"commit": commit, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "original": str(ROOT),
            "research_sha256": research_hashes(dest)}  # fmt: skip
    (dest / "COPY_INFO.json").write_text(json.dumps(info, indent=1), "utf-8")
    if research_hashes(dest) != research_hashes(ROOT):
        raise SystemExit("copy differs from the original research files")
    return info


def verify(copy: Path) -> dict[str, Any]:
    info = json.loads((copy / "COPY_INFO.json").read_text("utf-8"))
    now = research_hashes(copy)
    changed = sorted(k for k in info["research_sha256"] if now.get(k) != info["research_sha256"][k])
    added = sorted(set(now) - set(info["research_sha256"]))
    return {"commit": info["commit"], "files": len(now), "changed": changed, "added": added,
            "unchanged": not changed and not added}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--verify", type=Path)
    a = ap.parse_args()
    if a.verify:
        print(json.dumps(verify(a.verify), indent=1))
        return 0
    info = make(a.dest)
    print(f"copied commit {info['commit'][:10]} to {a.dest}; {len(info['research_sha256'])} research files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
