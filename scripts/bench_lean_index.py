"""Index size: full-text for every event vs the last N days (owner 2026-10-24, lean index).

200,000 parsed benchmark log events spread over 30 days in a temporary store; the
index is built twice (full-text days 0 = all, and 7) and both files are measured.

    python -m scripts.bench_lean_index
Writes results/extra/bench_lean_index.json.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from utils.paths import RESULTS_DIR


def main() -> None:
    from src.soc import engine, eventindex, parallel, perf, sandbox, search
    from src.soc.codeopt import _bench_lines

    out: dict = {"events": 200_000, "days_spread": 30}
    base = _bench_lines()
    for days in (0, 7):
        tmp = Path(tempfile.mkdtemp(prefix="lean_"))
        sandbox.isolate(tmp)
        lines = (base * (200_000 // len(base) + 1))[:200_000]
        evs = parallel.parse_lines(lines, workers=1)
        now = time.time()
        for k, e in enumerate(evs):
            e["ts"] = now - 30 * 86_400 + k * (30 * 86_400 / len(evs))
        engine._append(engine.EVENTS, evs)
        perf.save_settings("auto", fulltext_days=days)
        eventindex.close()
        t = time.perf_counter()
        eventindex.sync()
        build = time.perf_counter() - t
        t = time.perf_counter()
        n_old = search.run_store('"failed password"', now=now)["count"]
        q_s = time.perf_counter() - t
        eventindex.close()
        size = sum(p.stat().st_size for p in tmp.rglob("*.db*"))
        out[f"fulltext_days_{days}"] = {"index_mb": round(size / 2**20, 1), "build_s": round(build, 1),
                                        "text_search_all_time_s": round(q_s, 2), "matches": n_old,
                                        "store_mb": round(engine.EVENTS.stat().st_size / 2**20, 1)}  # fmt: skip
        print(days, out[f"fulltext_days_{days}"], flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    a, b = out["fulltext_days_0"]["index_mb"], out["fulltext_days_7"]["index_mb"]
    out["saving"] = f"{1 - b / a:.0%}"
    (RESULTS_DIR / "extra").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "extra" / "bench_lean_index.json").write_text(json.dumps(out, indent=1), "utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
