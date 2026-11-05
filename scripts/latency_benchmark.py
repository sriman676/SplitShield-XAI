"""Alert delay, end to end, per path (owner 2026-10-22: "what is the delay").

In a sandbox (src/soc/sandbox.py), with a mock SIEM syslog receiver (TCP):

1. log line -> alert -> SIEM       an attack line handed to the SOC (as the live
                                   watcher / an upload does) until the SIEM has it
2. agent -> receiver -> SIEM       a line written to a log file on an "agent",
                                   sent over HTTPS (pinned TLS) by the real agent
3. live flow -> alert -> SIEM      the batched flow scorer (batch <= 256 / <= 1 s)
4. incident -> SIEM                building incidents + forwarding a new one
5. dataset upload                  analysis speed of a large file (logs/s)

Each is measured N times; the report gives median and 95th percentile. Waiting
times by design are added separately (this PC's live logs are read every 2 s,
agents send every 2 s, incidents are pushed every 1 s), so:
    worst-case delay = measured processing + polling interval.

    python -m scripts.latency_benchmark [--n 30]
Results: results/extra/latency.json and .md
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from utils.paths import RESULTS_DIR

OUT = RESULTS_DIR / "extra"
ATTACK = "{ts} web9 sshd[{i}]: Failed password for root from 203.0.113.{o} port {p} ssh2"
POLL = {"this computer's logs (Windows wevtutil / Linux journal / macOS log)": 2.0,
        "agents on other machines": 2.0, "incidents to the SIEM": 1.0, "live flows (batch window)": 0.5}  # fmt: skip
# (2026-10-22: were 10 / 10 / 5 / 1 s; owner "reduce latency for live ones")


class SiemRx:
    """TCP syslog receiver that timestamps every line it gets."""

    def __init__(self) -> None:
        self.srv = socket.create_server(("127.0.0.1", 0))
        self.port = self.srv.getsockname()[1]
        self.got: list[tuple[float, str]] = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn: socket.socket) -> None:
        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.got.append((time.perf_counter(), line.decode("utf-8", "replace")))

    def wait_for(self, needle: str, t0: float, timeout: float = 30.0) -> float | None:
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            for ts, line in list(self.got):
                if ts >= t0 and needle in line:
                    return ts - t0
            time.sleep(0.001)
        return None


def _stats(xs: list[float]) -> dict[str, float]:
    xs = sorted(x * 1000 for x in xs)
    if not xs:
        return {}
    return {"n": len(xs), "median_ms": round(statistics.median(xs), 1),
            "p95_ms": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1), "max_ms": round(xs[-1], 1)}  # fmt: skip


def _syslog_ts() -> str:
    t = time.localtime()
    return f"{time.strftime('%b', t)} {t.tm_mday:2d} {time.strftime('%H:%M:%S', t)}"


def run(n: int) -> dict[str, Any]:
    from src.soc import engine, sandbox, sinks

    root = Path(tempfile.mkdtemp(prefix="splitshield-latency-"))
    sandbox.isolate(root)
    rx = SiemRx()
    s = sinks.load_settings()
    s["syslog"] |= {"enabled": True, "host": "127.0.0.1", "port": rx.port, "protocol": "tcp", "format": "cef"}
    sinks.save_settings(s)
    det = engine.load_detector()
    out: dict[str, Any] = {"when": time.strftime("%Y-%m-%d %H:%M"), "n": n}

    # warm-up: models, Sigma rules, index
    engine.ingest_text(ATTACK.format(ts=_syslog_ts(), i=0, o=1, p=1), "auth.log", det, True)

    # 1. log line -> alert -> SIEM
    lat = []
    for i in range(n):
        o = 10 + i
        t0 = time.perf_counter()
        engine.ingest_text(ATTACK.format(ts=_syslog_ts(), i=i, o=o, p=40000 + i), "auth.log", det, True)
        d = rx.wait_for(f"203.0.113.{o}", t0)
        if d is not None:
            lat.append(d)
    out["log_line_to_siem"] = _stats(lat)

    # 2. agent -> HTTPS receiver -> SIEM (the real agent and receiver)
    out["agent_to_siem"] = _agent_path(n, rx, root, det)

    # 3. live flow -> alert -> SIEM (batched scorer)
    out["flow_to_alert"] = _flow_path(det)

    # 4. incidents built + forwarded
    lat = []
    for _ in range(max(3, n // 5)):
        t0 = time.perf_counter()
        engine.incidents()
        engine.push_incidents(min_severity=0)
        lat.append(time.perf_counter() - t0)
    out["incidents_build_and_push"] = _stats(lat)

    # 5. dataset upload throughput
    lines = [ATTACK.format(ts=_syslog_ts(), i=i, o=(i % 200) + 1, p=1000 + i % 60000) for i in range(100_000)]
    t0 = time.perf_counter()
    engine.ingest_text("\n".join(lines), "big_auth.log", det, False)
    secs = time.perf_counter() - t0
    out["dataset_upload"] = {"lines": len(lines), "seconds": round(secs, 1), "logs_per_s": int(len(lines) / secs)}
    out["polling_s"] = POLL
    return out


def _agent_path(n: int, rx: SiemRx, root: Path, det: Any) -> dict[str, Any]:
    import importlib.util

    from src.soc import collector
    from utils.paths import PROJECT_ROOT

    srv = collector.make_server(port=0, bind="127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    spec = importlib.util.spec_from_file_location("ss_agent", PROJECT_ROOT / "scripts" / "agents" / "splitshield_agent.py")
    assert spec and spec.loader
    mod: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    home = root / "agent"
    home.mkdir()
    log = root / "agent_auth.log"
    log.write_text("", "utf-8")
    (home / "agent.json").write_text(json.dumps({
        "server": f"https://127.0.0.1:{srv.server_address[1]}", "key": collector.new_agent("bench", "bench1"),
        "fingerprint": collector.fingerprint(), "files": [str(log)], "journald": False,
        "macos_unified": False, "windows_logs": [], "backfill": False}), "utf-8")  # fmt: skip
    agent = mod.Agent(home)
    agent.load()
    agent.run_once()  # bookmark at the end
    lat = []
    for i in range(n):
        o = 100 + (i % 150)
        t0 = time.perf_counter()
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(ATTACK.format(ts=_syslog_ts(), i=i, o=o, p=50000 + i) + "\n")
        agent.run_once()  # what the agent does every 2 s
        d = rx.wait_for(f"203.0.113.{o}", t0)
        if d is not None:
            lat.append(d)
    srv.shutdown()
    return _stats(lat)


def _flow_path(det: Any) -> dict[str, Any]:
    if det is None:
        return {"skipped": "no deployed detector"}
    import pandas as pd

    from src.soc import engine
    from utils.paths import FIXTURES_DIR

    df = pd.read_csv(FIXTURES_DIR / "fixture.csv").dropna().head(256)
    lat_one, lat_batch = [], []
    for k in range(20):
        t0 = time.perf_counter()
        engine.record(engine.detect_flows(det, df.iloc[[k]], "bench"), send=False)
        lat_one.append(time.perf_counter() - t0)
    for _ in range(5):
        t0 = time.perf_counter()
        engine.record(engine.detect_flows(det, df, "bench"), send=False)
        lat_batch.append(time.perf_counter() - t0)
    return {"one_flow": _stats(lat_one), "batch_of_256": _stats(lat_batch),
            "note": "live flows wait at most 0.5 s for their batch"}  # fmt: skip


def write(res: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(res, indent=1), "utf-8")
    def row(name: str, st: dict[str, Any], wait: float) -> str:
        if not st or "median_ms" not in st:
            return f"| {name} | - | - | - |"
        worst = st["p95_ms"] / 1000 + wait
        return f"| {name} | {st['median_ms']} ms | {st['p95_ms']} ms | {wait:g} s | ~{worst:.1f} s |"

    md = [f"# Alert delay ({res['when']}, this PC)", "",
          "| Path | Processing (median) | Processing (95th pct) | Waiting by design | Worst case |",
          "|---|---|---|---|---|",
          row("Log line → alert → SIEM", res["log_line_to_siem"], 2),
          row("Agent → HTTPS receiver → SIEM", res["agent_to_siem"], 2),
          row("Live flow → alert (batch of 256)", res["flow_to_alert"].get("batch_of_256", {}), 0.5),
          row("Incident built and forwarded", res["incidents_build_and_push"], 1), "",
          f"Dataset upload: {res['dataset_upload']['lines']:,} lines in {res['dataset_upload']['seconds']} s "
          f"= {res['dataset_upload']['logs_per_s']:,} logs/s. Every line of this file is an attack (each becomes an "
          "alert, the worst case); ordinary mixed logs run at about 60,000 logs/s (Performance calibration). "
          "Alerts appear while the file is analysed.", ""]  # fmt: skip
    (OUT / "latency.md").write_text("\n".join(md), "utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    res = run(a.n)
    write(res)
    print((OUT / "latency.md").read_text("utf-8").encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
