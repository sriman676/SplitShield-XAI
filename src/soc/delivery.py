"""Guaranteed delivery to the SIEM (owner 2026-10-16, enterprise).

Every alert for a network sink (syslog, webhook) is either delivered or kept:

* An alert that cannot be delivered goes into a durable queue (SQLite,
  ``delivery.db`` next to the SOC settings) and is retried, oldest first, so the SIEM
  receives alerts in order once it is back.
* A circuit breaker: after a failure the sink is not tried again for 30 s, doubling
  up to 1 h. While it is open, new alerts go straight into the queue, so a SIEM
  outage never slows log analysis (before, every alert waited on a 5 s timeout).
* Alerts older than ``max_age_days`` (default 7) that still fail, or the oldest beyond
  ``max_queue`` (default 1,000,000), move to ``dead_letter.jsonl``. Nothing is thrown
  away; the Connect SOC page can put them back in the queue.
* The local JSON-lines outbox is written directly (it is the durable record itself).

``flush()`` is called by the monitor every 5 s, by every new alert while the breaker
is closed, and by the "Retry now" button. ``status()`` feeds the delivery card.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

NETWORK_SINKS = ("syslog", "webhook")
BACKOFF_START_S = 30.0
BACKOFF_MAX_S = 3600.0
FLUSH_BATCH = 500


def _paths() -> tuple[Path, Path]:
    from src.soc import sinks

    base = Path(sinks.SETTINGS).parent
    return base / "delivery.db", base / "dead_letter.jsonl"


def _db() -> sqlite3.Connection:
    db, _ = _paths()
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db), timeout=30, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, sink TEXT, "
                "created REAL, attempts INTEGER, last_error TEXT, payload TEXT)")  # fmt: skip
    con.execute("CREATE INDEX IF NOT EXISTS q_sink ON queue(sink, id)")
    con.execute("CREATE TABLE IF NOT EXISTS stats (sink TEXT PRIMARY KEY, delivered INTEGER "
                "DEFAULT 0, retried INTEGER DEFAULT 0, dead INTEGER DEFAULT 0, last_ok REAL, "
                "last_error TEXT, last_error_ts REAL, fails INTEGER DEFAULT 0, "
                "open_until REAL DEFAULT 0)")  # fmt: skip
    return con


def _lock() -> Any:
    from src.soc.engine import _file_lock

    return _file_lock(_paths()[0].with_suffix(".lock"))


def _limits(s: dict[str, Any]) -> tuple[float, int]:
    d = s.get("delivery", {})
    return float(d.get("max_age_days", 7)), int(d.get("max_queue", 1_000_000))


def _stat(con: sqlite3.Connection, sink: str) -> dict[str, Any]:
    con.execute("INSERT OR IGNORE INTO stats(sink) VALUES (?)", (sink,))
    row = con.execute("SELECT delivered, retried, dead, last_ok, last_error, last_error_ts, "
                      "fails, open_until FROM stats WHERE sink=?", (sink,)).fetchone()  # fmt: skip
    keys = ("delivered", "retried", "dead", "last_ok", "last_error", "last_error_ts", "fails",
            "open_until")  # fmt: skip
    return dict(zip(keys, row, strict=True))


def _ok(con: sqlite3.Connection, sink: str, retried: bool = False) -> None:
    con.execute("UPDATE stats SET delivered=delivered+1, retried=retried+?, last_ok=?, "
                "fails=0, open_until=0 WHERE sink=?", (int(retried), time.time(), sink))  # fmt: skip


def _fail(con: sqlite3.Connection, sink: str, err: str) -> None:
    st = _stat(con, sink)
    wait = min(BACKOFF_START_S * 2 ** int(st["fails"] or 0), BACKOFF_MAX_S)
    now = time.time()
    con.execute("UPDATE stats SET fails=fails+1, last_error=?, last_error_ts=?, open_until=? "
                "WHERE sink=?", (err[:300], now, now + wait, sink))  # fmt: skip


def _queued(con: sqlite3.Connection, sink: str) -> int:
    return int(con.execute("SELECT COUNT(*) FROM queue WHERE sink=?", (sink,)).fetchone()[0])


def _enqueue(con: sqlite3.Connection, sink: str, alert: dict[str, Any], err: str,
             s: dict[str, Any]) -> None:  # fmt: skip
    con.execute("INSERT INTO queue(sink, created, attempts, last_error, payload) VALUES (?,?,?,?,?)",
                (sink, time.time(), 1, err[:300], json.dumps(alert, default=str)))  # fmt: skip
    _, max_queue = _limits(s)
    over = _queued(con, sink) - max_queue
    if over > 0:
        _to_dead(con, sink, f"queue full (over {max_queue:,})", limit=over)


def _to_dead(con: sqlite3.Connection, sink: str, why: str, limit: int | None = None,
             older_than: float | None = None) -> int:  # fmt: skip
    q = "SELECT id, created, attempts, last_error, payload FROM queue WHERE sink=?"
    args: list[Any] = [sink]
    if older_than is not None:
        q += " AND created < ?"
        args.append(older_than)
    q += " ORDER BY id" + (f" LIMIT {int(limit)}" if limit else "")
    rows = con.execute(q, args).fetchall()
    if not rows:
        return 0
    _, dead = _paths()
    with open(dead, "a", encoding="utf-8") as fh:
        for _id, created, attempts, err, payload in rows:
            fh.write(json.dumps({"sink": sink, "why": why, "queued": created, "attempts": attempts,
                                 "last_error": err, "alert": json.loads(payload)}) + "\n")  # fmt: skip
    con.executemany("DELETE FROM queue WHERE id=?", [(r[0],) for r in rows])
    con.execute("UPDATE stats SET dead=dead+? WHERE sink=?", (len(rows), sink))
    return len(rows)


Sender = Callable[[dict[str, Any], dict[str, Any]], str]


def deliver(alert: dict[str, Any], s: dict[str, Any], senders: dict[str, Sender]) -> dict[str, str]:
    """Send one alert to each enabled network sink, or queue it. Never raises."""
    res: dict[str, str] = {}
    try:
        con = _db()
    except sqlite3.Error as exc:  # no queue available: plain best-effort send
        return {k: f(alert, s) for k, f in senders.items()} | {"queue": f"error: {exc}"}
    try:
        for sink, send in senders.items():
            st = _stat(con, sink)
            if float(st["open_until"] or 0) > time.time():
                _enqueue(con, sink, alert, "SIEM unreachable - waiting to retry", s)
                res[sink] = f"queued (retry after {time.strftime('%H:%M:%S', time.localtime(st['open_until']))})"
                continue
            if _queued(con, sink):
                with _lock():
                    _flush_sink(con, sink, s, send)
                if _queued(con, sink):  # still backed up: keep the order
                    _enqueue(con, sink, alert, "behind older queued alerts", s)
                    res[sink] = "queued (older alerts first)"
                    continue
            out = send(alert, s)
            if out == "ok":
                _ok(con, sink)
            else:
                _fail(con, sink, out)
                _enqueue(con, sink, alert, out, s)
                out += " (queued for retry)"
            res[sink] = out
    finally:
        con.close()
    return res


def _flush_sink(con: sqlite3.Connection, sink: str, s: dict[str, Any], send: Sender,
                budget_s: float = 3.0) -> int:  # fmt: skip
    max_age, _ = _limits(s)
    _to_dead(con, sink, f"older than {max_age:g} days", older_than=time.time() - max_age * 86400)
    sent, t0 = 0, time.time()
    rows = con.execute("SELECT id, payload FROM queue WHERE sink=? ORDER BY id LIMIT ?",
                       (sink, FLUSH_BATCH)).fetchall()  # fmt: skip
    for qid, payload in rows:
        out = send(json.loads(payload), s)
        if out != "ok":
            con.execute("UPDATE queue SET attempts=attempts+1, last_error=? WHERE id=?",
                        (out[:300], qid))  # fmt: skip
            _fail(con, sink, out)
            break
        con.execute("DELETE FROM queue WHERE id=?", (qid,))
        _ok(con, sink, retried=True)
        sent += 1
        if time.time() - t0 > budget_s:
            break
    return sent


def flush(s: dict[str, Any] | None = None, force: bool = False) -> dict[str, int]:
    """Retry queued alerts for every sink whose breaker is closed (or all, ``force``)."""
    from src.soc import sinks

    s = s or sinks.load_settings()
    senders = sinks.network_senders(s)
    out: dict[str, int] = {}
    if not _paths()[0].is_file():
        return out
    con = _db()
    try:
        for sink in NETWORK_SINKS:
            if not _queued(con, sink):
                continue
            if sink not in senders:  # sink switched off: keep the alerts for later
                out[sink] = 0
                continue
            if force or float(_stat(con, sink)["open_until"] or 0) <= time.time():
                with _lock():  # one process retries a sink at a time (no duplicates)
                    out[sink] = _flush_sink(con, sink, s, senders[sink])
    finally:
        con.close()
    return out


def status() -> dict[str, dict[str, Any]]:
    """Per sink: delivered, queued, oldest queued, dead-lettered, last ok / error."""
    out: dict[str, dict[str, Any]] = {}
    if not _paths()[0].is_file():
        return out
    con = _db()
    try:
        for sink in NETWORK_SINKS:
            st = _stat(con, sink)
            n, oldest = con.execute("SELECT COUNT(*), MIN(created) FROM queue WHERE sink=?",
                                    (sink,)).fetchone()  # fmt: skip
            out[sink] = st | {"queued": int(n), "oldest": oldest,
                              "breaker_open": float(st["open_until"] or 0) > time.time()}  # fmt: skip
    finally:
        con.close()
    return out


def dead_letters(last: int = 200) -> list[dict[str, Any]]:
    _, dead = _paths()
    try:
        lines = dead.read_text("utf-8").splitlines()[-last:]
    except OSError:
        return []
    return [json.loads(ln) for ln in lines if ln.strip()]


def requeue_dead() -> int:
    """Put every dead-lettered alert back in the queue (the file is kept as .done)."""
    _, dead = _paths()
    if not dead.is_file():
        return 0
    rows = [json.loads(ln) for ln in dead.read_text("utf-8").splitlines() if ln.strip()]
    con = _db()
    try:
        con.execute("BEGIN")
        for r in rows:
            con.execute("INSERT INTO queue(sink, created, attempts, last_error, payload) "
                        "VALUES (?,?,?,?,?)", (r["sink"], time.time(), 0, "requeued",
                                               json.dumps(r["alert"], default=str)))  # fmt: skip
        con.execute("UPDATE stats SET open_until=0")
        con.execute("COMMIT")
    finally:
        con.close()
    done = dead.with_name(f"dead_letter.requeued-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    dead.replace(done)  # kept, never deleted
    return len(rows)
