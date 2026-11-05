"""Fast mode (owner 2026-10-17): the research pipeline with a larger training batch,
for faster RE-TRAINING. Opt-in, off by default; the registered results always come
from the normal mode.

    python -m scripts.fast_mode --batch 512 -- run_pipeline.py --experiment T1_temporal --seed 0
    python -m scripts.fast_mode --batch 512 -- ai_engineer.py --fresh --skip-intake --parallel 3

No research file is edited: scripts/fastmode_site/sitecustomize.py overrides the batch
size when Python starts (also in every child process the pipeline launches). Run it in
a COPY of the project (scripts/make_rerun_copy.py), never in the original.

A batch size is accepted for fast mode only if, against the normal-mode runs of the
same commit, every hypothesis verdict is unchanged and no run's held-out precision,
recall or macro-F1 drops (scripts/fast_mode.py --compare).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SITE = HERE / "fastmode_site"


def env_for(batch: int) -> dict[str, str]:
    env = dict(os.environ)
    env["SPLITSHIELD_FAST_BATCH"] = str(int(batch))
    env["PYTHONPATH"] = os.pathsep.join([str(SITE), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    return env


def _metrics(run_dir: Path) -> dict[str, float]:
    """Held-out precision / recall / macro-F1 numbers of one run (metrics/*.json)."""
    out: dict[str, float] = {}

    def walk(o: Any, p: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}.{k}")
        elif isinstance(o, int | float) and not isinstance(o, bool):
            key = p.lower()
            # held-out numbers: stage files store them as <arm>_macro_f1.value (no "test" in the
            # name); skip validation curves, training numbers and the reference "control" values
            skip = ("val", "train", "control", "per_turn")
            if any(m in key for m in ("macro_f1", "precision", "recall")) and not any(x in key for x in skip):
                out[p] = float(o)

    for f in sorted((run_dir / "metrics").glob("*.json")) if (run_dir / "metrics").is_dir() else []:
        try:
            walk(json.loads(f.read_text("utf-8")), f.stem)
        except ValueError:
            continue
    return out


def _runs(root: Path) -> dict[tuple[str, str], Path]:
    """(experiment, seed) -> newest run folder."""
    found: dict[tuple[str, str], Path] = {}
    for d in sorted((root / "runs").glob("*__seed*__*")):
        exp, seed = d.name.split("__")[0], d.name.split("__")[1]
        found[(exp, seed)] = d
    return found


def compare(normal: Path, fast: Path, tolerance: float = 0.0) -> dict[str, Any]:
    """Per run: every held-out P / R / macro-F1 of fast mode >= normal mode - tolerance."""
    a, b = _runs(normal), _runs(fast)
    rows, worse = [], []
    for key in sorted(set(a) & set(b)):
        ma, mb = _metrics(a[key]), _metrics(b[key])
        drops = {k: (ma[k], mb[k]) for k in ma if k in mb and mb[k] < ma[k] - tolerance}
        rows.append({"run": "/".join(key), "metrics": len(set(ma) & set(mb)), "drops": len(drops)})
        if drops:
            worse.append({"run": "/".join(key), "drops": dict(list(drops.items())[:10])})
    va = json.loads((normal / "results" / "final_verdicts.json").read_text("utf-8")) \
        if (normal / "results" / "final_verdicts.json").is_file() else {}  # fmt: skip
    vb = json.loads((fast / "results" / "final_verdicts.json").read_text("utf-8")) \
        if (fast / "results" / "final_verdicts.json").is_file() else {}  # fmt: skip
    return {"runs_compared": len(rows), "runs_worse": len(worse), "worse": worse,
            "verdicts_normal": va, "verdicts_fast": vb,
            "verdicts_same": (va == vb) if va and vb else None, "rows": rows}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("NORMAL_COPY", "FAST_COPY"))
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- script.py args...")
    a = ap.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=1))
        return 0
    cmd = [c for c in a.cmd if c != "--"]
    if not cmd:
        ap.error("give the command after --")
    if (Path.cwd() / ".git").exists():  # the original has git history; copies do not
        raise SystemExit("run fast mode in a copy of the project, not in the original")
    return subprocess.run([sys.executable, *cmd], env=env_for(a.batch)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
