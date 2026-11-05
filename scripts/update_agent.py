"""Internet update agent - command line (src/soc/updater.py).

    python -m scripts.update_agent               # daily run (does nothing when off)
    python -m scripts.update_agent --enable      # switch the agent on (default: off)
    python -m scripts.update_agent --disable
    python -m scripts.update_agent --force       # run now, even if off / ran today
    python -m scripts.update_agent --status
    python -m scripts.update_agent --rollback sigma|attack|urlhaus|feodo|...
    python -m scripts.update_agent --dry-run [--out DIR]
        # stage + validate + gate, apply into a scratch copy only

The scheduled task (scripts/install_update_task.ps1) runs the first form daily.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--enable", action="store_true")
    g.add_argument("--disable", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--rollback", choices=None, metavar="SOURCE")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, help="dry-run output folder (default: temp)")
    ap.add_argument(
        "--root", type=Path, help="project root to read/test (default: this)"
    )
    ap.add_argument("--no-tests", action="store_true", help="skip the pytest gate")
    ap.add_argument("--sources", help="comma list, default all")
    a = ap.parse_args(argv)

    from src.soc import intel, updater
    from utils.logging_config import configure_logging

    configure_logging()
    root = (a.root or ROOT).resolve()
    lay = updater.Layout(root=root, out=root)
    if a.enable or a.disable:
        updater.set_enabled(bool(a.enable), lay.state)
        print(f"update agent {'ENABLED' if a.enable else 'DISABLED'}")
        return 0
    if a.status:
        st = updater.load_state(lay.state)
        print(json.dumps({k: st.get(k) for k in ("enabled", "last_run", "versions")}
                         | {"intel": intel.status(lay.intel_dir())}, indent=2))  # fmt: skip
        return 0
    if a.rollback:
        print(json.dumps(updater.rollback(a.rollback, lay), indent=2))
        return 0
    if a.dry_run:
        out = (a.out or Path(tempfile.mkdtemp(prefix="ud_"))).resolve()
        if out == root:
            print("--out must differ from the project root")
            return 2
        lay = updater.Layout(root=root, out=out)
    sources = tuple(a.sources.split(",")) if a.sources else updater.SOURCES
    rep = updater.run_once(
        force=a.force or a.dry_run,
        dry_run=a.dry_run,
        lay=lay,
        sources=sources,
        run_tests_gate=not a.no_tests,
    )
    if a.dry_run:
        rep["dry_run_output"] = str(lay.out)
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
