"""Keep one result per (run, stage) before aggregation: newest attempt wins.

A retried phase (or a stage-1 re-run, scripts/rerun_baselines.py) writes a new run
directory with the same experiment, seed and config hash. The older attempt's
metric files would otherwise be aggregated too, counting that seed twice. Run ids
are ``<exp>__seed<N>__<YYYYMMDDTHHMMZ>__<hash>``; runs equal after removing the
timestamp are the same run, and for each metric file only the newest copy is kept
(older ones renamed ``*.superseded`` with a note). E3 matrix cells carry no
timestamp, so each is unique and never touched. Idempotent.

    python -m scripts.dedupe_runs
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TS = re.compile(r"__\d{8}T\d{4}Z")


def dedupe(runs_dir: Path = ROOT / "runs") -> list[str]:
    """Supersede older copies of each metric file; return what was superseded."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in runs_dir.glob("*"):
        if d.is_dir() and _TS.search(d.name):
            groups[_TS.sub("", d.name)].append(d)
    done: list[str] = []
    for dirs in groups.values():
        if len(dirs) < 2:
            continue
        dirs.sort(key=lambda p: _TS.search(p.name).group(0))  # type: ignore[union-attr]
        seen: dict[str, Path] = {}
        for d in reversed(dirs):  # newest first
            for mf in sorted((d / "metrics").glob("*.json")):
                if mf.stem.endswith("_failed"):
                    continue  # failure stubs are never aggregated as results
                if mf.name in seen:
                    mf.rename(mf.with_name(mf.name + ".superseded"))
                    with open(
                        d / "metrics" / "SUPERSEDED.txt", "a", encoding="utf-8"
                    ) as fh:
                        fh.write(f"{mf.name} superseded by run {seen[mf.name].name}\n")
                    done.append(f"{d.name}/{mf.name}")
                else:
                    seen[mf.name] = d
    return done


def main() -> int:
    out = dedupe()
    print(f"[dedupe] superseded {len(out)} older metric file(s)")
    for x in out:
        print(f"  {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
