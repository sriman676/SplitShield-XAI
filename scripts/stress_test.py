"""SOC stress tests (owner 2026-10-16): hostile input, concurrency, faults + recovery,
volume. Everything runs on temporary stores; the SOC's own store is never touched.

    python -m scripts.stress_test                    # hostile, concurrency, faults
    python -m scripts.stress_test --volume 1000000   # + a 1 M-event volume test
    python -m scripts.stress_test --only hostile

(Every dashboard page and button: tests/test_dashboard_buttons.py; speed at
200 k / 1 M / 10 M events: scripts/speed_benchmark.py.)
Results: results/extra/stress_test.json + .md. Exit code 1 if any check failed.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from utils.paths import RESULTS_DIR

OUT = RESULTS_DIR / "extra" / "stress_test"
results: list[dict[str, Any]] = []


def check(
    section: str, name: str, ok: bool, detail: str = "", seconds: float | None = None
) -> None:
    results.append({"section": section, "check": name, "ok": bool(ok), "detail": detail,
                    "seconds": None if seconds is None else round(seconds, 3)})  # fmt: skip
    print(
        f"[{'PASS' if ok else 'FAIL'}] {section}: {name}"
        + (f" - {detail}" if detail else ""),
        flush=True,
    )


def _temp_store() -> Path:
    """Point the engine at a fresh temporary store (+ its index) and return the folder."""
    from src.soc import engine, eventindex

    d = Path(tempfile.mkdtemp(prefix="stress_"))
    for name in ("EVENTS", "ALERTS", "PUSHED", "STATUS", "FLOW_SAMPLE"):
        setattr(engine, name, d / getattr(engine, name).name)
    eventindex.close()
    return d


def _event(i: int, now: float, **kw: Any) -> dict[str, Any]:
    ev = {"ts": now - i, "source": "winsec", "kind": "failed_logon", "src_ip": f"10.0.{i % 200}.{i % 250}",
          "user": f"u{i % 50}", "host": "WS01", "detail": f"failed logon {i}", "techniques": ["T1110"]}  # fmt: skip
    return ev | kw


def peak_mb() -> float:
    """Peak working set of this process (Windows), MB."""
    from src.soc.winmem import peak_mb as _p

    return _p() or 0.0


# --- hostile input --------------------------------------------------------------------

UNESCAPED_LINK = re.compile(r"(?<!\\)\]\((?:https?|javascript|data):", re.I)


def hostile() -> None:
    from src.nlp.assistant import ShieldAssistant
    from src.soc import custom, engine, eventindex, search
    from src.soc.parsers import parse_text

    s = "hostile"
    _temp_store()
    now = time.time()
    evil = [
        _event(
            1,
            now,
            detail="![x](http://evil.test/pixel.png) seen",
            user="[Microsoft](http://evil.test)",
        ),
        _event(
            2,
            now,
            detail="<script>alert(1)</script> [ok](javascript:alert(1))",
            host="<img src=x>",
        ),
        _event(3, now, src_ip="10.9.9.9", detail="normal failed logon"),
    ]
    engine.record(evil, send=False)
    res = search.run_store("show failed logins")
    text = search.answer_text(res)
    check(s, "log text cannot become a link or remote image (search answer)",
          not UNESCAPED_LINK.search(text), f"{res['count']} events")  # fmt: skip
    bot = ShieldAssistant()
    for q in (
        "show failed logins",
        "who is attacking us?",
        "show critical incidents",
        "is brute force happening?",
    ):
        ans = bot.answer(q)["text"]
        check(s, f"assistant answer safe: {q!r}", not UNESCAPED_LINK.search(ans))
    # search syntax injection: quotes, FTS operators, SQL
    n0 = eventindex.count()
    for q in ('find "', 'find "a" OR "b"', "find *", "find NEAR(a b)", "find a) OR (1=1",
              "find '; DROP TABLE ev; --", 'find """"', "find \\"):  # fmt: skip
        t = time.perf_counter()
        try:
            r = search.run_store(q)
            ok = isinstance(r["count"], int)
            detail = f"{r['count']} hits"
        except Exception as exc:  # noqa: BLE001 - the check is that nothing escapes
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        check(s, f"search survives {q!r}", ok, detail, time.perf_counter() - t)
    check(s, "index intact after injection attempts", eventindex.count() == n0)
    t = time.perf_counter()
    r = search.run_store("find " + "x" * 10_000)
    check(
        s,
        "10,000-character search",
        time.perf_counter() - t < 5,
        f"{r['count']} hits",
        time.perf_counter() - t,
    )
    # parser: huge line, binary, NUL bytes, deep JSON
    for name, blob in (("5 MB single line", "A" * 5_000_000),
                       ("binary bytes", bytes(random.Random(0).randrange(256) for _ in range(200_000)).decode("latin-1")),
                       ("NUL bytes", "a\x00b\x00" * 50_000),
                       ("deeply nested JSON", "[" * 5000 + "]" * 5000),
                       ("JSON bomb-like array", json.dumps([{"a": "b" * 100}] * 20_000))):  # fmt: skip
        t = time.perf_counter()
        try:
            evs = parse_text(blob, "hostile.log")
            ok, detail = True, f"{len(evs)} events"
        except RecursionError as exc:
            ok, detail = False, f"RecursionError: {exc}"
        except Exception as exc:  # noqa: BLE001 - must not crash the ingest path
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t
        check(s, f"parser survives {name}", ok and dt < 60, detail, dt)
    # zip: path traversal names and a bomb-sized header
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "../../escape.log",
            "Oct 16 10:00:00 h sshd[1]: Failed password for root from 1.2.3.4 port 1 ssh2\n",
        )
        z.writestr("C:/Windows/evil.log", "x\n")
    before = set(Path.cwd().parent.glob("escape.log"))
    try:
        engine.ingest_zip(buf.getvalue(), "evil.zip", send=False)
        ok = True
    except ValueError:
        ok = True  # refused cleanly is fine too
    check(
        s,
        "zip with ../ names writes nothing outside",
        ok and set(Path.cwd().parent.glob("escape.log")) == before,
    )
    # regex rules: catastrophic patterns refused, long lines bounded
    for pat in ("(a+)+$", r"(\w*)*x", "((ab)*)+", "(a|aa)+b"):
        try:
            custom._simple_pred({"field": "any", "op": "regex", "value": pat})
            refused = False
        except ValueError:
            refused = True
        check(s, f"custom regex {pat!r} refused", refused)
    check(
        s,
        "slow regex caught by the timed probe",
        not custom.regex_runs_in_time("(a|aa)+b"),
    )
    pred = custom._simple_pred(
        {"field": "detail", "op": "regex", "value": r"\w+@evil\.test"}
    )
    t = time.perf_counter()
    pred({"detail": "a" * 100_000})
    check(s, "allowed regex rule on a 100 k-char line is time-bounded", time.perf_counter() - t < 5,
          seconds=time.perf_counter() - t)  # fmt: skip
    eventindex.close()


# --- concurrency ------------------------------------------------------------------------


def _writer(store: str, k: int, n: int) -> None:
    from src.soc import engine

    engine.EVENTS = Path(store)
    now = time.time()
    for b in range(0, n, 500):
        engine._append(
            engine.EVENTS,
            [_event(k * 1_000_000 + i, now) for i in range(b, min(n, b + 500))],
        )


def _reader(store: str, secs: float, q: Any) -> None:
    from src.soc import engine, eventindex, search

    engine.EVENTS = Path(store)
    errors, ops = [], 0
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            search.run_store("show failed logins for u7")
            eventindex.incidents(900.0, scale=1.0)
            eventindex.count()
            ops += 1
        except Exception as exc:  # noqa: BLE001 - counted, reported
            errors.append(f"{type(exc).__name__}: {exc}"[:200])
    eventindex.close()
    q.put({"ops": ops, "errors": errors[:5], "n_errors": len(errors)})


def concurrency(writers: int = 3, per: int = 20_000, readers: int = 2) -> None:
    from src.soc import engine, eventindex

    s = "concurrency"
    d = _temp_store()
    store = str(engine.EVENTS)
    Path(store).write_text("", "utf-8")
    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    ws = [ctx.Process(target=_writer, args=(store, k, per)) for k in range(writers)]
    rs = [ctx.Process(target=_reader, args=(store, 45.0, q)) for _ in range(readers)]
    t = time.perf_counter()
    for p in ws + rs:
        p.start()
    for p in ws:
        p.join()
    for p in rs:
        p.join()
    dt = time.perf_counter() - t
    rep = [q.get() for _ in rs]
    errs = sum(r["n_errors"] for r in rep)
    check(s, f"{readers} readers searching / correlating while {writers} writers append",
          errs == 0, f"{sum(r['ops'] for r in rep)} reader rounds, errors: {[e for r in rep for e in r['errors']][:3]}", dt)  # fmt: skip
    lines = Path(store).read_text("utf-8").splitlines()
    bad = sum(1 for ln in lines if not ln.startswith("{") or not ln.endswith("}"))
    check(s, "no interleaved / torn lines in the store", bad == 0 and len(lines) == writers * per,
          f"{len(lines):,} lines, {bad} torn")  # fmt: skip
    eventindex.close()
    check(s, "index count equals the store after the race", eventindex.count() == writers * per,
          f"{eventindex.count():,}")  # fmt: skip
    eventindex.close()
    del d


# --- faults + recovery -------------------------------------------------------------------


def _build(store: str) -> None:
    from src.soc import engine, eventindex

    engine.EVENTS = Path(store)
    eventindex.count()


def faults(n: int = 300_000) -> None:
    from src.soc import engine, eventindex, search

    s = "faults"
    _temp_store()
    now = time.time()
    with open(engine.EVENTS, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps(_event(i, now)) + "\n")
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_build, args=(str(engine.EVENTS),))
    p.start()
    time.sleep(8)
    p.kill()  # power cut in the middle of the first index build
    p.join()
    eventindex.close()
    t = time.perf_counter()
    got = eventindex.count()
    check(s, "index build killed half-way resumes to the exact count", got == n,
          f"{got:,} of {n:,}", time.perf_counter() - t)  # fmt: skip
    a = search.run(q := "show failed logins for u7", engine.read_jsonl(engine.EVENTS))
    b = search.run_store(q)
    check(s, "answers after the crash equal a full scan", a["count"] == b["count"] or b["capped"],
          f"{a['count']} vs {b['count']}")  # fmt: skip
    eventindex.close()
    db = eventindex.db_path()
    for suf in ("-wal", "-shm"):
        Path(f"{db}{suf}").unlink(missing_ok=True)
    db.write_bytes(os.urandom(4096))  # disk damage
    check(s, "damaged index file is rebuilt", eventindex.count() == n)
    engine.EVENTS.unlink()
    check(
        s,
        "store deleted -> empty answers, no error",
        eventindex.count() == 0 and eventindex.incidents(900.0, scale=1.0) == [],
    )
    eventindex.close()


# --- volume --------------------------------------------------------------------------------


def volume(n: int) -> None:
    from scripts.speed_benchmark import base_events, measure, write_store

    s = "volume"
    d = Path(tempfile.mkdtemp(prefix="stress_vol_", dir=os.environ.get("STRESS_WORK")))
    store = d / "events.jsonl"
    t = time.perf_counter()
    write_store(store, base_events(), n, time.time())
    check(s, f"{n:,}-event store written", store.is_file(), f"{store.stat().st_size / 2**20:,.0f} MB",
          time.perf_counter() - t)  # fmt: skip
    row = measure(store, "auto", base_events()[:1000])
    worst_search = max(row["search_ms"].values())
    worst_chat = max(row["chat_ms"].values())
    check(
        s,
        f"{n:,} events: index built",
        row["events"] >= n,
        f"{row['index_build_s']} s, db {row['db_mb']} MB",
    )
    check(
        s,
        f"{n:,} events: slowest search {worst_search:.0f} ms",
        worst_search < 2000,
        json.dumps(row["search_ms"]),
    )
    check(
        s,
        f"{n:,} events: slowest chat answer {worst_chat:.0f} ms",
        worst_chat < 3000,
        json.dumps(row["chat_ms"]),
    )
    check(s, f"{n:,} events: peak memory {peak_mb():,.0f} MB", peak_mb() < 6000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="hostile,concurrency,faults")
    ap.add_argument(
        "--volume", type=int, default=0, help="also run a volume test of N events"
    )
    a = ap.parse_args()
    os.environ["SPLITSHIELD_IN_TRIAL"] = "1"  # never start nightly work from here
    parts: dict[str, Any] = {
        "hostile": hostile,
        "concurrency": concurrency,
        "faults": faults,
    }
    for name, fn in parts.items():
        if not a.only or name in a.only.split(","):
            fn()
    if a.volume:
        volume(a.volume)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    failed = [r for r in results if not r["ok"]]
    doc = {"when": time.strftime("%Y-%m-%d %H:%M"), "passed": len(results) - len(failed),
           "failed": len(failed), "checks": results}  # fmt: skip
    OUT.with_suffix(".json").write_text(json.dumps(doc, indent=1), "utf-8")
    md = [f"# Stress tests ({doc['when']}): {doc['passed']} passed, {doc['failed']} failed", "",
          "| section | check | result | detail | seconds |", "|---|---|---|---|---|"]  # fmt: skip
    def cell(v: object) -> str:  # a '|' inside a check (e.g. a regex) must not split the row
        return str(v).replace("|", "\\|")

    md += [f"| {r['section']} | {cell(r['check'])} | {'✅' if r['ok'] else '❌'} | {cell(r['detail'][:120])} | {r['seconds'] or ''} |"
           for r in results]  # fmt: skip
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n", "utf-8")
    print(f"{doc['passed']} passed, {doc['failed']} failed -> {OUT}.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
