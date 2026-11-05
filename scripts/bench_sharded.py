"""Log analysis speed, old path vs sharded store path (owner 2026-10-24).

Old: workers parse, one process stores. Sharded: every worker parses AND stores its
share (src/soc/parallel.py parse_and_record). Same benchmark lines as the
Performance page calibration, a temporary store, end-to-end logs/s.

    python -m scripts.bench_sharded [--lines 240000]
Writes results/extra/bench_sharded.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from utils.paths import RESULTS_DIR


def main() -> None:
    from src.soc import engine, eventindex, parallel, sandbox
    from src.soc.codeopt import _bench_lines
    from src.soc.perf import cores

    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=240_000)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="bench_shard_"))
    sandbox.isolate(tmp)
    base = _bench_lines()
    lines = (base * (a.lines // len(base) + 1))[: a.lines]
    engine.record(parallel.parse_lines(lines[:2000], workers=1), send=False)  # rules / models loaded once
    out: dict = {"lines": len(lines), "cores": cores(), "old": {}, "sharded": {}}

    def clear() -> None:
        eventindex.close()
        for n in ("EVENTS", "ALERTS"):
            p = getattr(engine, n)
            if p.exists():
                p.unlink()

    for w in sorted({1, 2, 4, 6, 8, max(1, cores() - 1)}):
        clear()
        if w > 1:
            parallel.parse_lines(lines[: parallel.SLICE * w], workers=w)  # pool warm-up
        t = time.perf_counter()
        engine.record(parallel.parse_lines(lines, workers=w), send=False)
        out["old"][str(w)] = round(len(lines) / (time.perf_counter() - t))
        if w > 1:
            clear()
            parallel.parse_and_record(lines[: parallel.SLICE * w], w, send=False)  # warm the stores' code
            clear()
            t = time.perf_counter()
            parallel.parse_and_record(lines, w, send=False)
            out["sharded"][str(w)] = round(len(lines) / (time.perf_counter() - t))
        print(w, out["old"][str(w)], out["sharded"].get(str(w)), flush=True)
    parallel.shutdown()
    eventindex.close()
    shutil.rmtree(tmp, ignore_errors=True)
    out["when"] = time.strftime("%Y-%m-%d %H:%M")
    (RESULTS_DIR / "extra").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "extra" / "bench_sharded.json").write_text(json.dumps(out, indent=1), "utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
