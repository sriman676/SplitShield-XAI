"""End-to-end load (soak) test (owner 2026-10-23): many agents -> the real HTTPS receiver
-> parsing, rules, models -> event store -> incidents -> guaranteed delivery -> a mock
SIEM (syslog over TCP, CEF), for a sustained time, with a SIEM outage and a receiver
restart in the middle.

Every log line carries a unique marker (its source port), so the checks are exact:

* nothing lost:      every line written by an agent's log is stored exactly once;
* exactly once:      every alert in the outbox reaches the SIEM, none twice;
* outages survived:  the SIEM down for ``--siem-down`` s and the receiver down for
                     ``--rx-down`` s mid-run; spooled / queued data arrives afterwards;
* delay:             line written -> alert at the SIEM, median / 95th percentile, outside
                     the outage windows;
* memory:            working set sampled every 10 s; growth after warm-up must be bounded.

Runs in a sandbox (src/soc/sandbox.py); the SOC's own store is never touched.

    python -m scripts.soak_test [--minutes 30] [--agents 10] [--rate 20]
Results: results/extra/soak_test.json and .md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import socket
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT, RESULTS_DIR

OUT = RESULTS_DIR / "extra"
ATTACK_SHARE = 0.1
_MARK = "port "


class Siem:
    """TCP syslog receiver that can be stopped and started again on the same port."""

    def __init__(self) -> None:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.got: dict[str, list[float]] = {}  # marker -> receive times
        self.srv: socket.socket | None = None
        self.conns: list[socket.socket] = []
        self.lock = threading.Lock()

    def start(self) -> None:
        self.srv = socket.create_server(("127.0.0.1", self.port), reuse_port=False)
        threading.Thread(target=self._accept, args=(self.srv,), daemon=True).start()

    def stop(self) -> None:
        if self.srv:
            self.srv.close()
            self.srv = None
        for c in self.conns:
            try:
                c.close()
            except OSError:
                pass
        self.conns = []

    def _accept(self, srv: socket.socket) -> None:
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            self.conns.append(c)
            threading.Thread(target=self._read, args=(c,), daemon=True).start()

    def _read(self, c: socket.socket) -> None:
        buf = b""
        while True:
            try:
                chunk = c.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", "replace")
                i = text.find("SOAK")
                if i >= 0:
                    mk = text[i:].split()[0].strip(".,;|")
                    with self.lock:
                        self.got.setdefault(mk, []).append(time.time())


def _line(agent: int, n: int, attack: bool) -> tuple[str, str]:
    mk = f"SOAK{agent:02d}x{n:07d}"
    t = time.localtime()
    ts = f"{time.strftime('%b', t)} {t.tm_mday:2d} {time.strftime('%H:%M:%S', t)}"
    if attack:
        txt = f"{ts} web{agent} sshd[{n}]: Failed password for root from 203.0.113.{agent + 1} port {n} ssh2 {mk}"
    else:
        txt = f"{ts} web{agent} sshd[{n}]: Accepted publickey for deploy from 10.1.{agent}.5 port {n} ssh2 {mk}"
    return mk, txt


def run(minutes: float, n_agents: int, rate: float, siem_down: float, rx_down: float) -> dict[str, Any]:
    from src.soc import collector, delivery, engine, sandbox, sinks, winmem

    root = Path(tempfile.mkdtemp(prefix="splitshield-soak-"))
    sandbox.isolate(root)
    siem = Siem()
    siem.start()
    s = sinks.load_settings()
    s["syslog"] |= {"enabled": True, "host": "127.0.0.1", "port": siem.port, "protocol": "tcp", "format": "cef"}
    sinks.save_settings(s)
    srv = collector.make_server(port=0, bind="127.0.0.1")
    rx_port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    spec = importlib.util.spec_from_file_location("ss_agent", PROJECT_ROOT / "scripts" / "agents" / "splitshield_agent.py")
    assert spec and spec.loader
    mod: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    agents, logs = [], []
    for i in range(n_agents):
        home = root / f"agent{i}"
        home.mkdir()
        log = root / f"agent{i}.log"
        log.write_text("", "utf-8")
        (home / "agent.json").write_text(json.dumps({
            "server": f"https://127.0.0.1:{rx_port}", "key": collector.new_agent("soak", f"soak{i}"),
            "fingerprint": collector.fingerprint(), "files": [str(log)], "journald": False,
            "macos_unified": False, "windows_logs": [], "backfill": True}), "utf-8")  # fmt: skip
        a = mod.Agent(home)
        a.load()
        agents.append(a)
        logs.append(log)

    written: dict[str, tuple[float, bool]] = {}  # marker -> (write time, attack)
    wlock = threading.Lock()
    stop = threading.Event()
    rng = random.Random(7)
    t_end = time.time() + minutes * 60
    outages: list[tuple[str, float, float]] = []

    def writer(i: int) -> None:
        n = 0
        while not stop.is_set():
            t0 = time.time()
            with open(logs[i], "a", encoding="utf-8") as fh:
                for _ in range(max(1, int(rate))):
                    attack = rng.random() < ATTACK_SHARE
                    mk, txt = _line(i, n, attack)
                    fh.write(txt + "\n")
                    with wlock:
                        written[mk] = (time.time(), attack)
                    n += 1
            time.sleep(max(0.0, 1.0 - (time.time() - t0)))

    def shipper(i: int) -> None:
        while not stop.is_set():
            try:
                agents[i].run_once()
            except Exception:  # noqa: BLE001 - an agent keeps running (as the real one does)
                pass
            time.sleep(1.0)

    threads = [threading.Thread(target=f, args=(i,), daemon=True) for i in range(n_agents) for f in (writer, shipper)]
    for t in threads:
        t.start()
    mem: list[tuple[float, float]] = []
    t_start = time.time()
    siem_at, rx_at = t_start + (t_end - t_start) / 3, t_start + (t_end - t_start) / 2
    siem_done = rx_done = False
    while time.time() < t_end:
        now = time.time()
        if not siem_done and now >= siem_at:
            siem.stop()
            time.sleep(siem_down)
            siem.start()
            outages.append(("siem", now, time.time()))
            siem_done = True
        elif not rx_done and now >= rx_at:
            srv.shutdown()
            srv.server_close()
            time.sleep(rx_down)
            srv = collector.make_server(port=rx_port, bind="127.0.0.1")
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            outages.append(("receiver", now, time.time()))
            rx_done = True
        delivery.flush()  # what the autopilot / monitor does every few seconds
        mem.append((time.time() - t_start, winmem.current_mb() or 0.0))
        time.sleep(10)
    stop.set()
    time.sleep(2)
    for a in agents:  # the last lines and anything still spooled
        for _ in range(3):
            a.run_once()
    for _ in range(60):
        delivery.flush(force=True)
        if not delivery.status().get("syslog", {}).get("queued"):
            break
        time.sleep(1)
    time.sleep(3)
    srv.shutdown()
    return _check(written, siem, outages, mem, engine, delivery, minutes, n_agents, rate)


def _check(written: dict[str, tuple[float, bool]], siem: Siem, outages: list[tuple[str, float, float]],
           mem: list[tuple[float, float]], engine: Any, delivery: Any, minutes: float, n_agents: int,
           rate: float) -> dict[str, Any]:  # fmt: skip
    stored: dict[str, int] = {}
    for e in engine.read_jsonl(engine.EVENTS):
        d = str(e.get("detail", "")) + " " + str(e.get("raw", ""))
        i = d.find("SOAK")
        if i >= 0:
            mk = d[i:].split()[0].strip(".,;|")
            stored[mk] = stored.get(mk, 0) + 1
    alerts = [a for a in engine.read_jsonl(engine.ALERTS)]
    alert_marks = set()
    for a in alerts:
        d = str(a.get("detail", "")) + " " + str(a.get("reason", ""))
        i = d.find("SOAK")
        if i >= 0:
            alert_marks.add(d[i:].split()[0].strip(".,;|"))
    lost = [m for m in written if m not in stored]
    dup_store = [m for m, c in stored.items() if c > 1]
    siem_missing = [m for m in alert_marks if m not in siem.got]
    siem_dup = [m for m, ts in siem.got.items() if len(ts) > 1]

    def in_outage(t: float) -> bool:
        return any(a - 5 <= t <= b + 60 for _, a, b in outages)

    lat = sorted((siem.got[m][0] - written[m][0]) * 1000 for m in alert_marks
                 if m in siem.got and m in written and not in_outage(written[m][0]))  # fmt: skip
    warm = [mb for t, mb in mem if t >= 120]
    growth = (warm[-1] - warm[0]) if len(warm) > 1 else 0.0
    st = delivery.status().get("syslog", {})
    res = {
        "when": time.strftime("%Y-%m-%d %H:%M"), "minutes": minutes, "agents": n_agents,
        "lines_per_s": n_agents * rate, "lines_written": len(written),
        "attack_lines": sum(1 for _, a in written.values() if a),
        "stored_once": len(written) - len(lost) - len(dup_store), "lost": len(lost), "stored_twice": len(dup_store),
        "alerts": len(alert_marks), "siem_received": len(alert_marks) - len(siem_missing),
        "siem_missing": len(siem_missing), "siem_duplicates": len(siem_dup),
        "queued_at_end": st.get("queued", 0), "dead_lettered": st.get("dead", 0),
        "outages": [{"what": w, "seconds": round(b - a)} for w, a, b in outages],
        "delay_ms": {"median": round(statistics.median(lat), 1) if lat else None,
                     "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None, "n": len(lat)},
        "memory_mb": {"start": round(mem[0][1], 1) if mem else None, "peak": round(max(m for _, m in mem), 1) if mem else None,
                      "growth_after_warmup": round(growth, 1)},
        "examples_lost": lost[:5], "examples_missing_at_siem": siem_missing[:5],
    }  # fmt: skip
    res["checks"] = {
        "nothing lost": res["lost"] == 0, "nothing stored twice": res["stored_twice"] == 0,
        "every alert reached the SIEM": res["siem_missing"] == 0, "no alert twice at the SIEM": res["siem_duplicates"] == 0,
        "queue drained": res["queued_at_end"] == 0 and res["dead_lettered"] == 0,
        "memory bounded (< 300 MB growth after warm-up)": growth < 300,
    }  # fmt: skip
    res["ok"] = all(res["checks"].values())
    return res


def write(res: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "soak_test.json").write_text(json.dumps(res, indent=1), "utf-8")
    md = [f"# End-to-end soak test ({res['when']}): {'PASS' if res['ok'] else 'FAIL'}", "",
          f"{res['agents']} agents, {res['lines_per_s']:.0f} lines/s for {res['minutes']} min "
          f"({res['lines_written']:,} lines, {res['attack_lines']:,} attacks); outages: "
          + ", ".join(f"{o['what']} {o['seconds']} s" for o in res["outages"]), "",
          "| Check | Result |", "|---|---|"] + [f"| {k} | {'✅' if v else '❌'} |" for k, v in res["checks"].items()] + [
          "", f"Stored once {res['stored_once']:,} / {res['lines_written']:,}; alerts {res['alerts']:,}, at the SIEM "
          f"{res['siem_received']:,} (duplicates {res['siem_duplicates']}); delay line -> SIEM median "
          f"{res['delay_ms']['median']} ms, 95th pct {res['delay_ms']['p95']} ms (outside outages); memory start "
          f"{res['memory_mb']['start']} MB, peak {res['memory_mb']['peak']} MB, growth after warm-up "
          f"{res['memory_mb']['growth_after_warmup']} MB."]  # fmt: skip
    (OUT / "soak_test.md").write_text("\n".join(md), "utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--rate", type=float, default=20, help="lines per second per agent")
    ap.add_argument("--siem-down", type=float, default=120)
    ap.add_argument("--rx-down", type=float, default=60)
    a = ap.parse_args()
    res = run(a.minutes, a.agents, a.rate, a.siem_down, a.rx_down)
    write(res)
    print((OUT / "soak_test.md").read_text("utf-8").encode("ascii", "replace").decode())
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
