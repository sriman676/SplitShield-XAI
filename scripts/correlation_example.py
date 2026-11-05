"""A worked cross-log correlation example for the report (owner 2026-10-22): the demo
attack data (flows, Windows, Sysmon, syslog, web, cloud ...) analysed in a sandbox;
the incident that combines the most different sensors is written out with its
timeline. Output: results/extra/correlation_example.json
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from utils.paths import RESULTS_DIR


def main() -> int:
    from src.soc import engine, samples, sandbox

    root = Path(tempfile.mkdtemp(prefix="splitshield-corr-"))
    sandbox.isolate(root)
    det = engine.load_detector()
    for p in samples.make_samples(root / "samples"):
        engine.ingest_text(p.read_text("utf-8"), p.name, det, False)
    incs = engine.incidents()
    if not incs:
        raise SystemExit("no incidents from the demo data")
    best = max(incs, key=lambda i: (len(i.get("sources", [])), i["severity"]))
    tl = [{"time": time.strftime("%H:%M:%S", time.localtime(e["ts"])), "source": e["source"],
           "what": str(e.get("detail") or e.get("kind"))[:120],
           "techniques": ", ".join(e.get("techniques") or [])} for e in best["timeline"]]  # fmt: skip
    path = " -> ".join(f"{t['tactic']}: {t['name']} ({t['id']})" for t in best["techniques"])
    summary = (f"{len(incs)} incidents were built from {engine.event_count():,} demo events. The incident about "
               f"{best['src_ip']} combines {len(best['sources'])} sensors ({', '.join(best['sources'])}) "
               f"into one attack path: {path}. Severity {best['severity']}/10 ({best['level']})"
               + (f", raised by {best.get('board_boost', 0)} because other model families saw the same entity"
                  if best.get("board_boost") else "") + ".")  # fmt: skip
    out = {"summary": summary, "incident": best["id"], "sources": best["sources"], "timeline": tl,
           "linked_findings": best.get("linked_findings", [])}  # fmt: skip
    (RESULTS_DIR / "extra" / "correlation_example.json").write_text(json.dumps(out, indent=1, default=str), "utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
