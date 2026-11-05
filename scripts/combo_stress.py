"""Every-combination stress test (owner 2026-10-16/17): each combination of the SOC's
options runs the whole SOC pipeline end to end in its own sandbox
(src/soc/sandbox.py), and a list of invariants must hold.

Dimensions, run EXHAUSTIVELY (every combination of every value):
  evolution   evolving | frozen
  perf        auto | memory (2 GB) | speed (target 50k logs/s)
  place       home_pc | small_office | server | lab
  format      cef | leef | ecs | ocsf          (syslog body; webhook ecs / ocsf / json)
  transport   udp | tcp | tls | webhook        (to mock SIEMs started per worker)
  siem        up | down-then-up                (guaranteed delivery)
  retention   off | archive | delete
  report      docx+md | html
  sensitivity validated | middle | strictest   (every live model's alert threshold)
  role        admin | analyst | viewer         (who operates; recorded in the audit log)
= 2 x 3 x 4 x 4 x 4 x 2 x 3 x 2 x 3 x 3 = 41,472 combinations,
plus per-model freezes: all, and each component (9 models + rules) alone.
Pages and buttons per role are covered by tests/test_dashboard*.py.

Pipeline per combination: settings -> ingest the demo attack data (flows, Windows,
syslog, web, cloud...) through engine.ingest_text with SIEM forwarding -> incidents ->
incident case (status + note; refused for viewers) -> assistant questions -> ICS v3
window + phishing check -> SIEM delivery (mock receivers; for 'down-then-up' the
receiver starts late and the queue must drain; pseudonymised names on every other
combination) -> report in the chosen formats -> retention 40 days later.
(2026-10-23: cases, assistant, ICS, phishing and pseudonymisation added so every
pipeline runs in every combination; agents / receiver are covered by
tests/test_agent.py, tests/test_collector*.py and scripts/latency_benchmark.py.)

Invariants: no exception; events stored; incidents exist and are severity-ordered;
every alert sent to the SIEM arrived exactly once and parses in its format (none lost
when the SIEM was down); the report files exist and open; retention conserves every
line (kept + archived = before, delete mode archives nothing); frozen evolution blocks
model updates and evolving allows them; parsing is identical in every perf mode;
the audit chain stays intact.

    python -m scripts.combo_stress --workers 2              # full run (~hours)
    python -m scripts.combo_stress --limit 20 --workers 1   # smoke
Results: results/extra/combo_stress.json and .md (failures with combination + error).
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import socket
import ssl
import tempfile
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from utils.paths import RESULTS_DIR

DIMS: dict[str, list[str]] = {
    "evolution": ["evolving", "frozen"],
    "perf": ["auto", "memory", "speed"],
    "place": ["home_pc", "small_office", "server", "lab"],
    "format": ["cef", "leef", "ecs", "ocsf"],
    "transport": ["udp", "tcp", "tls", "webhook"],
    "siem": ["up", "down-then-up"],
    "retention": ["off", "archive", "delete"],
    "report": ["docx+md", "html"],
    "sensitivity": ["validated", "middle", "strictest"],
    "role": ["admin", "analyst", "viewer"],
}
EXTRA: dict[str, list[str]] = {}
FREEZES = ["none", "all", "flow", "netflow", "zeek", "phishing", "malware", "ics", "insider", "intent",
           "shieldlm", "rules"]
OUT = RESULTS_DIR / "extra"


def combinations() -> list[dict[str, str]]:
    keys = list(DIMS)
    rows = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*DIMS.values())]
    for r in rows:
        r["freeze"] = "none"
    rows += [dict(rows[0], freeze=f, evolution="evolving") for f in FREEZES[1:]]
    return rows


# --- mock SIEM receivers (one set per worker process) ----------------------------------


def _make_cert(root: Path) -> tuple[Path, Path]:
    from src.soc import collector

    here = root / "siem_cert"
    here.mkdir(parents=True, exist_ok=True)
    old = collector.home
    collector.home = lambda: here
    try:
        return collector.ensure_cert()
    finally:
        collector.home = old


class Receivers:
    """Fresh mock SIEM receivers for ONE combination (stopped afterwards)."""

    def __init__(self, cert: Path, key: Path) -> None:
        self.got: dict[str, list[bytes]] = {"udp": [], "tcp": [], "tls": [], "webhook": []}
        self.ports: dict[str, int] = {}
        self.cert, self.key = cert, key
        self.servers: dict[str, Any] = {}

    def stop_all(self) -> None:
        for which, srv in self.servers.items():
            try:
                if which == "webhook":
                    srv.shutdown()
                    srv.server_close()
                else:
                    srv.close()
            except OSError:
                pass
        self.servers.clear()

    def start(self, which: str) -> int:
        if which in self.servers:
            return self.ports[which]
        if which == "udp":
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.bind(("127.0.0.1", self.ports.get(which, 0)))
            self.servers[which] = sk

            def loop() -> None:
                while True:
                    try:
                        self.got["udp"].append(sk.recv(65535))
                    except OSError:
                        return

        elif which in ("tcp", "tls"):
            srv = socket.create_server(("127.0.0.1", self.ports.get(which, 0)))
            self.servers[which] = srv
            ctx = None
            if which == "tls":
                ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ctx.load_cert_chain(str(self.cert), str(self.key))

            def loop() -> None:
                while True:
                    try:
                        conn, _ = srv.accept()
                    except OSError:
                        return
                    threading.Thread(target=self._stream, args=(conn, which, ctx), daemon=True).start()

        else:
            got = self.got["webhook"]

            class H(BaseHTTPRequestHandler):
                def do_POST(self) -> None:  # noqa: N802
                    got.append(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    self.send_response(200)
                    self.end_headers()

                def log_message(self, *_a: Any) -> None:
                    pass

            srv2 = HTTPServer(("127.0.0.1", self.ports.get(which, 0)), H)
            self.servers[which] = srv2

            def loop() -> None:
                srv2.serve_forever()

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        srv_obj = self.servers[which]
        port = srv_obj.getsockname()[1] if hasattr(srv_obj, "getsockname") else srv_obj.server_address[1]
        self.ports[which] = port
        return port

    def _stream(self, conn: socket.socket, which: str, ctx: ssl.SSLContext | None) -> None:
        s: Any = ctx.wrap_socket(conn, server_side=True) if ctx else conn
        buf = b""
        try:
            while chunk := s.recv(65536):
                buf += chunk
                while True:
                    if which == "tls":
                        head, sp, rest = buf.partition(b" ")
                        if not sp or not head.isdigit() or len(rest) < int(head):
                            break
                        self.got["tls"].append(rest[: int(head)])
                        buf = rest[int(head):]
                    else:
                        line, nl, rest = buf.partition(b"\n")
                        if not nl:
                            break
                        self.got["tcp"].append(line)
                        buf = rest
        except OSError:
            pass

    def port_for(self, which: str) -> int:
        """A port for a receiver that is NOT started yet (SIEM down)."""
        if which not in self.ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM if which != "udp" else socket.SOCK_DGRAM) as s:
                s.bind(("127.0.0.1", 0))
                self.ports[which] = s.getsockname()[1]
        return self.ports[which]



_W: dict[str, Any] = {}


def _worker_init(base: str) -> None:
    import logging

    logging.disable(logging.WARNING)
    root = Path(base) / f"w{id(_W)}-{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    _W["root"] = root
    _W["cert"] = _make_cert(root)
    from src.soc import engine, samples

    _W["det"] = engine.load_detector()
    _W["samples"] = [(p.name, p.read_text("utf-8")) for p in samples.make_samples(root / "samples")]
    from src.mlareas import common
    from src.nlp.assistant import ShieldAssistant

    _W["assistant"] = ShieldAssistant()
    hai = common.raw_data_dir("hai") / "hai-22.04_test3.csv"
    _W["ics_window"] = __import__("pandas").read_csv(hai, nrows=400) if hai.is_file() else None


def run_one(i: int, combo: dict[str, str]) -> dict[str, Any]:
    t0 = time.time()
    try:
        checks = _run(i, combo)
        return {"i": i, "combo": combo, "ok": True, "s": round(time.time() - t0, 2), "checks": checks}
    except Exception as exc:  # noqa: BLE001 - every failure is recorded with its combination
        return {"i": i, "combo": combo, "ok": False, "s": round(time.time() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2500:]}  # fmt: skip


def _run(i: int, c: dict[str, str]) -> dict[str, Any]:
    from src.soc import sandbox

    root: Path = _W["root"] / f"c{i}"
    shutil.rmtree(root, ignore_errors=True)
    sandbox.isolate(root)
    rx = Receivers(*_W["cert"])
    try:
        return _pipeline(i, c, root, rx)
    finally:
        rx.stop_all()
        from src.soc import sinks as _s

        for sk in _s._CONN.values():
            try:
                sk.close()
            except OSError:
                pass
        _s._CONN.clear()


def _pipeline(i: int, c: dict[str, str], root: Path, rx: Receivers) -> dict[str, Any]:
    from src.soc import (
        audit,
        custom,
        delivery,
        engine,
        eventindex,
        evolution,
        perf,
        places,
        reports,
        retention,
        sinks,
        users,
    )

    checks: dict[str, Any] = {}

    # --- settings ----------------------------------------------------------------------
    users.create_first_admin("admin", "combo-password-1")
    who = "admin"
    if c["role"] != "admin":
        users.add_user("admin", f"op-{c['role']}", "combo-password-1", c["role"])
        who = f"op-{c['role']}"
    frozen = [c["freeze"]] if c["freeze"] not in ("none", "all") else []
    evolution.save("frozen" if c["freeze"] == "all" else c["evolution"], frozen, who=who)
    perf.save_settings(c["perf"], memory_gb=2.0 if c["perf"] == "memory" else None,
                       target_lps=50_000 if c["perf"] == "speed" else None)  # fmt: skip
    places.set_place(c["place"], mode="manual")
    for model in custom.LIVE_MODELS:
        base = custom.validated(model)
        if base is None or c["sensitivity"] == "validated":
            continue
        v = 0.999 if c["sensitivity"] == "strictest" else round((base + 0.999) / 2, 4)
        custom.set_threshold(model, v, base)
    s = sinks.load_settings()
    s["jsonl"]["enabled"] = True
    s["syslog"]["enabled"] = s["webhook"]["enabled"] = False
    t = c["transport"]
    up = c["siem"] == "up"
    port = rx.start(t) if up else rx.port_for(t)
    if t == "webhook":
        fmt = {"cef": "json", "leef": "splunk_hec"}.get(c["format"], c["format"])
        s["webhook"] |= {"enabled": True, "url": f"http://127.0.0.1:{port}/in", "format": fmt}
    else:
        s["syslog"] |= {"enabled": True, "host": "localhost" if t == "tls" else "127.0.0.1", "port": port,
                        "protocol": t, "format": c["format"], "ca_file": str(rx.cert)}  # fmt: skip
    pseudo = i % 2 == 1  # every other combination (all other options vary independently)
    s["pseudonymize"] = {"enabled": pseudo, "fields": ["user", "host"]}
    sinks.save_settings(s)
    sinks._CONN.clear()

    # --- ingest ------------------------------------------------------------------------
    det = _W["det"]
    for name, text in _W["samples"]:
        engine.ingest_text(text, name, det, True)
    n_events = eventindex.count()
    checks["events"] = n_events
    assert n_events > 0, "no events stored"

    # parsing must not depend on the performance mode: the same events as the reference
    got_kinds = sorted(str(e.get("kind")) for e in engine.read_jsonl(engine.EVENTS))
    if "ref_kinds" not in _W:
        _W["ref_kinds"] = got_kinds
    assert got_kinds == _W["ref_kinds"], "parsed events differ from the reference run"

    # --- incidents ---------------------------------------------------------------------
    incs = engine.incidents()
    checks["incidents"] = len(incs)
    assert incs, "no incidents from the demo attack"
    assert all(engine_order_ok(a, b) for a, b in zip(incs, incs[1:], strict=False)), "incidents out of order"
    engine.push_incidents(min_severity=0)

    # --- incident case: who may change it --------------------------------------------
    from src.soc import cases

    iid = incs[0]["id"]
    if c["role"] == "viewer":
        try:
            cases.set_status(who, iid, "investigating")
            raise AssertionError("a viewer changed a case")
        except PermissionError:
            pass
    else:
        cases.set_status(who, iid, "investigating")
        cases.add_note(who, iid, f"combo {i}")
        assert cases.get(iid)["status"] == "investigating", "case status not kept"

    # --- assistant: answers come from this store -------------------------------------
    a = _W["assistant"]
    assert a.answer("who is attacking us?")["intent"] == "top", "assistant: top intent"
    assert a.answer("what is a brute force attack?")["intent"] == "explain", "assistant: explain"
    assert a.answer("tell me a joke")["intent"] == "off_topic", "assistant: off-topic"
    ip = next((x["src_ip"] for x in incs if correlate_ip(x["src_ip"])), None)
    if ip:
        r = a.answer(f"what did {ip} do?")
        assert r["intent"] == "ip" and ip in r["text"], "assistant: IP answer not from the store"

    # --- area models stay usable (ICS v3 window, phishing) ---------------------------
    if _W.get("ics_window") is not None:
        from src.mlareas import ics

        out = ics.score_window(_W["ics_window"])
        assert {"alarm", "why", "model"} <= set(out) and out["why"], "ICS window"
    from src.mlareas import phishing

    ph = phishing.check_url("http://paypa1-secure-login.example.com/verify")
    assert {"verdict", "why", "phishing_probability"} <= set(ph), "phishing check"

    # --- SIEM delivery -----------------------------------------------------------------
    sent_out = [x for x in engine.read_jsonl(sinks.OUTBOX)]
    named = [x for x in sent_out if x.get("user")]
    if pseudo and named:
        assert all(str(x["user"]).startswith("u-") for x in named), "user names not pseudonymised"
    checks["pseudonymised"] = pseudo
    if not up:
        st = delivery.status()
        key = "webhook" if t == "webhook" else "syslog"
        queued = st.get(key, {}).get("queued", 0)
        checks["queued_while_down"] = queued
        if t != "udp":  # UDP cannot know the SIEM is down: nothing is queued
            assert queued > 0, "nothing queued while the SIEM was down"
        rx.start(t)
        con = delivery._db()
        con.execute("UPDATE stats SET open_until=0")
        con.close()
        for _ in range(20):
            delivery.flush(force=True)
            if not delivery.status().get(key, {}).get("queued"):
                break
        assert not delivery.status().get(key, {}).get("queued"), "queue did not drain"
    time.sleep(0.3)  # in-flight datagrams / frames
    got = rx.got[t]
    checks["siem_received"] = len(got)
    if t != "udp" or up:
        assert len(got) == len(sent_out), f"SIEM got {len(got)} of {len(sent_out)} alerts"
    for body in got[:50]:
        _check_format(body.decode("utf-8", "replace"), c, t)

    # --- report ------------------------------------------------------------------------
    fmts = ["docx", "md"] if c["report"] == "docx+md" else ["html"]
    rep = reports.generate("today", template={"formats": fmts}, who=who)
    for f in fmts:
        p = rep["files"][f]
        assert p.is_file() and p.stat().st_size > 200, f"report {f} missing"
    if "docx" in fmts:
        from docx import Document

        assert Document(str(rep["files"]["docx"])).paragraphs

    # --- evolution gate ----------------------------------------------------------------
    # the gate controller.guarded / cycle / autopilot ask before any update (the update
    # path itself is unit-tested in tests/test_evolution.py; it would touch real models)
    for comp in evolution.components():
        blocked = not evolution.allowed(comp)
        should = c["freeze"] in ("all", comp) or (c["evolution"] == "frozen" and c["freeze"] == "none")
        assert blocked == should, f"evolution gate wrong for {comp} (blocked={blocked})"

    # --- retention ---------------------------------------------------------------------
    if c["retention"] != "off":
        retention.save({"enabled": True, "days": 30, "mode": c["retention"]}, who=who)
        before = sum(1 for _ in open(engine.EVENTS, "rb"))
        r = retention.apply(now=time.time() + 40 * 86400)
        ev = r["events"]
        assert ev["kept"] + ev["moved"] == before, "retention lost lines"
        archived = list(retention.archive_dir().glob("events-*.jsonl.gz")) if retention.archive_dir().is_dir() else []
        assert (c["retention"] == "archive") == bool(archived), "archive / delete mode wrong"
        checks["retention_moved"] = ev["moved"]
        assert eventindex.count() == ev["kept"]

    assert audit.verify()["ok"], "audit chain broken"
    eventindex.close()
    return checks


def correlate_ip(entity: str) -> bool:
    """A real source IP (not user:/host:/unknown entities)."""
    import ipaddress

    try:
        ipaddress.ip_address(entity)
        return True
    except ValueError:
        return False


def engine_order_ok(a: dict[str, Any], b: dict[str, Any]) -> bool:
    from src.soc import correlate

    return correlate.incident_order(a) <= correlate.incident_order(b)


def _check_format(text: str, c: dict[str, str], t: str) -> None:
    if t == "webhook":
        doc = json.loads(text)
        if c["format"] == "ecs":
            assert "@timestamp" in doc and doc["event"]["kind"] == "alert"
        elif c["format"] == "ocsf":
            assert doc["class_uid"] == 2004
        elif c["format"] == "leef":  # splunk_hec envelope
            assert "event" in doc and "sourcetype" in doc
        return
    body = text.split("SplitShield: ", 1)[1] if "SplitShield: " in text else text.split(" - - - ", 1)[1]
    if c["format"] == "cef":
        assert body.startswith("CEF:0|SplitShield|")
    elif c["format"] == "leef":
        assert body.startswith("LEEF:2.0|SplitShield|")
    else:
        doc = json.loads(body)
        assert ("@timestamp" in doc) if c["format"] == "ecs" else doc["class_uid"] == 2004


def write_report(rows: list[dict[str, Any]], started: float) -> dict[str, Any]:
    bad = [r for r in rows if not r["ok"]]
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "combinations": len(rows), "passed": len(rows) - len(bad),
           "failed": len(bad), "hours": round((time.time() - started) / 3600, 2), "dimensions": DIMS,
           "covered": EXTRA, "freezes": FREEZES, "failures": bad[:500],
           "seconds_median": sorted(r["s"] for r in rows)[len(rows) // 2] if rows else None}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "combo_stress.json").write_text(json.dumps(out, indent=1, default=str), "utf-8")
    md = [f"# Every-combination stress test ({out['when']})", "",
          f"**{out['passed']:,} of {out['combinations']:,} combinations passed**, {out['failed']} failed, "
          f"{out['hours']} h on this PC.", "", "Dimensions (exhaustive): "
          + "; ".join(f"{k}: {', '.join(v)}" for k, v in DIMS.items()) + ".", "",
          "Covered with every other value: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in EXTRA.items())
          + f". Per-model freezes: {', '.join(FREEZES)}.", ""]  # fmt: skip
    if bad:
        md += ["## Failures", "", "| # | Combination | Error |", "|---|---|---|"]
        md += [f"| {r['i']} | {', '.join(f'{k}={v}' for k, v in r['combo'].items())} | {r['error'][:200]} |"
               for r in bad[:200]]  # fmt: skip
    (OUT / "combo_stress.md").write_text("\n".join(md) + "\n", "utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N combinations (smoke)")
    ap.add_argument("--stride", type=int, default=1, help="every Nth combination (spread smoke)")
    a = ap.parse_args()
    combos = combinations()[:: a.stride]
    if a.limit:
        combos = combos[: a.limit]
    base = Path(tempfile.mkdtemp(prefix="splitshield-combo-"))
    started = time.time()
    rows: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(a.workers, initializer=_worker_init, initargs=(str(base),)) as pool:
            futs = [pool.submit(run_one, i, c) for i, c in enumerate(combos)]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if k % 25 == 0 or k == len(futs):
                    print(f"{k}/{len(futs)} done, {sum(not r['ok'] for r in rows)} failed", flush=True)
                    write_report(rows, started)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    out = write_report(rows, started)
    print(json.dumps({k: out[k] for k in ("combinations", "passed", "failed", "hours")}))
    return 0 if not out["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
