"""Re-run stage 1 (RF + GBM + MLP baselines) for runs made before the GBM fix.

The LightGBM baseline was trained without class weights and diverged on the rarest
classes (macro-F1 0.16-0.45 vs RF 0.95). Owner decision (2026-10-13): re-run the
full stage 1 of every affected run as a NEW run with the same experiment and seed;
the old ``stage1_baseline.json`` is renamed ``*.superseded`` (no longer aggregated)
with a note naming the replacement run.

    python -m scripts.rerun_baselines            # all affected full runs
    python -m scripts.rerun_baselines --dry-run  # list them only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def affected(runs_dir: Path = RUNS) -> list[tuple[Path, str, int]]:
    """(run dir, experiment, seed) of non-smoke runs whose stage-1 GBM is unfixed."""
    out = []
    for d in sorted(runs_dir.glob("*")):
        s1 = d / "metrics" / "stage1_baseline.json"
        man = d / "manifest.json"
        if not (s1.is_file() and man.is_file()):
            continue
        m = json.loads(man.read_text("utf-8"))
        metrics = json.loads(s1.read_text("utf-8")).get("metrics", {})
        exp = m.get("experiment_id", "")
        if exp == "smoke" or "skipped" in metrics or "gbm_class_weight" in metrics:
            continue
        if "gbm_macro_f1" not in metrics:  # no GBM was trained: nothing is wrong
            continue
        out.append((d, exp, int(m.get("seed", 0))))
    return out


def _new_run_id(output: str) -> str | None:
    for line in output.splitlines():
        if "run_id=" in line:
            return line.split("run_id=")[1].split()[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    todo = affected()
    print(f"[baselines] {len(todo)} run(s) need stage 1 re-run")
    failed = 0
    for d, exp, seed in todo:
        print(f"[baselines] {d.name} -> rerun {exp} seed {seed}")
        if args.dry_run:
            continue
        cmd = [sys.executable, "run_pipeline.py", "--experiment", exp,
               "--seed", str(seed), "--stage", "1"]  # fmt: skip
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        new = _new_run_id(res.stdout)
        if res.returncode or not new:
            failed += 1
            print(
                f"[baselines] FAILED {d.name}:\n{res.stdout[-800:]}{res.stderr[-800:]}"
            )
            continue
        s1 = d / "metrics" / "stage1_baseline.json"
        s1.rename(s1.with_name(s1.name + ".superseded"))
        (d / "metrics" / "SUPERSEDED.txt").write_text(
            f"stage1_baseline superseded by run {new}: the GBM baseline here was "
            "trained without class weights and diverged (fixed 2026-10-13).\n",
            "utf-8",
        )
        print(f"[baselines] {d.name}: stage 1 superseded by {new}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
