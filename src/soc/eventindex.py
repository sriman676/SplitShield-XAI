"""On-disk index of the event store: searches, answers and incidents in milliseconds,
up to about 10 million stored events (owner 2026-10-16).

``artifacts/soc/events.jsonl`` stays the store of record (append-only: SIEM export,
backups and every other reader use it). ``events.db`` next to it is derived and
disposable: before every query it reads only the bytes appended since the last
time, and it rebuilds itself when it is missing, damaged, or the store was cleared
or replaced. Several processes (dashboard, monitor, autopilot) may use it at once
(SQLite WAL; one of them catches up, the others wait a few ms).

What it answers - always the same result as scanning the whole list:
  * ``search``    the plain-English filters of src/soc/search.py as SQL plus a trigram
                  full-text index (exact substring matching, any case). Newest first;
                  above ``CAP`` matches the count shows as "10,000+" (owner decision)
                  unless ``exact=True``.
  * ``evidence``  the events src/nlp/status.py needs ("is brute force happening?").
  * ``incidents`` correlation per entity, kept in the index; new events only make the
                  entities they mention be recomputed. Risk-board links come from the
                  index too.
  * ``stats``     how many events, which sources, what time span.

Memory for the index cache follows the performance plan (src/soc/perf.py:
automatic, a memory budget in GB, or a logs-per-second target).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

Event = dict[str, Any]
SCHEMA = 10  # 6: entity-less events per source + file; 7 (2026-10-23): inc.sev for bounded views;
# 8 (2026-10-24): findings also under "obs:<machine that logged them>" (risk-board victim follow);
# 9 (2026-10-24): loopback / unspecified IPs are no longer board entities (no links through them);
# 10 (2026-10-24): full-text rows can be pruned (lean index: Performance -> full-text days)
CAP = 10_000  # counts above this are shown as "10,000+" (owner 2026-10-16)
CHUNK = 32 * 2**20  # bytes of new store read per step while catching up
_lock = threading.RLock()
_conns: dict[str, sqlite3.Connection] = {}

_DDL = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS ev(
  id INTEGER PRIMARY KEY, ts REAL, entity TEXT, src_ip TEXT, dst_ip TEXT,
  user TEXT, host TEXT, sha TEXT, kind TEXT, source TEXT, alert INTEGER,
  label TEXT, dport TEXT, js TEXT);
CREATE INDEX IF NOT EXISTS ev_ts ON ev(ts);
CREATE INDEX IF NOT EXISTS ev_src ON ev(src_ip);
CREATE INDEX IF NOT EXISTS ev_dst ON ev(dst_ip);
CREATE INDEX IF NOT EXISTS ev_user ON ev(user);
CREATE INDEX IF NOT EXISTS ev_host ON ev(host);
CREATE INDEX IF NOT EXISTS ev_sha ON ev(sha);
CREATE INDEX IF NOT EXISTS ev_kind ON ev(kind);
CREATE INDEX IF NOT EXISTS ev_source ON ev(source);
CREATE INDEX IF NOT EXISTS ev_entity ON ev(entity, ts, id);
CREATE INDEX IF NOT EXISTS ev_alert ON ev(alert, id);
CREATE INDEX IF NOT EXISTS ev_label ON ev(label);
CREATE TABLE IF NOT EXISTS tech(
  id INTEGER, typ TEXT, raw TEXT, up TEXT, base TEXT, pos INTEGER);
CREATE INDEX IF NOT EXISTS tech_id ON tech(id);
CREATE INDEX IF NOT EXISTS tech_raw ON tech(raw, id);
CREATE INDEX IF NOT EXISTS tech_up ON tech(up, id);
CREATE INDEX IF NOT EXISTS tech_base ON tech(base, id);
CREATE VIRTUAL TABLE IF NOT EXISTS hay USING fts5(h, tokenize='trigram', content='', contentless_delete=1);
CREATE TABLE IF NOT EXISTS link(
  a TEXT, b TEXT, first_id INTEGER, PRIMARY KEY(a, b)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS finding(key TEXT, ts REAL, family TEXT, techs TEXT, what TEXT);
CREATE INDEX IF NOT EXISTS finding_key ON finding(key, ts);
CREATE TABLE IF NOT EXISTS scache(k TEXT PRIMARY KEY, upto INTEGER, state TEXT);
CREATE TABLE IF NOT EXISTS inc(win REAL, entity TEXT, start REAL, end_ REAL, data TEXT, sev REAL);
CREATE INDEX IF NOT EXISTS inc_entity ON inc(win, entity, start);
CREATE INDEX IF NOT EXISTS inc_sev ON inc(win, sev DESC, end_ DESC);
"""


# --- settings ------------------------------------------------------------------


def _store() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS)


def db_path() -> Path:
    return _store().with_name("events.db")


# --- connection + catching up ------------------------------------------------------


def close() -> None:
    with _lock:
        for ev in _checkpointing.values():  # background checkpoint threads stop
            ev.set()
        _checkpointing.clear()
        for c in _conns.values():
            try:
                c.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            c.close()
        _conns.clear()
        _istate.clear()


def _connect() -> sqlite3.Connection:
    path = db_path()
    key = str(path)
    con = _conns.get(key)
    if con is not None:
        return con
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(
        path, timeout=30, check_same_thread=False, isolation_level=None
    )
    from src.soc import perf

    prof = perf.plan()
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute(f"PRAGMA cache_size=-{int(prof['index_cache_mb']) * 1024}")
        con.execute(f"PRAGMA mmap_size={int(prof['index_mmap_mb']) * 2**20}")
        con.execute("PRAGMA temp_store=MEMORY")
        # checkpoints run in the background (see _checkpointer), not inside a query's
        # commit; the large automatic limit is only a safety net for short processes
        con.execute("PRAGMA wal_autocheckpoint=16384")
        con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        if _meta(con, "schema") != str(SCHEMA):  # new or older layout: start over
            names = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'hay_%'"
                )
            ]
            for name in names:
                con.execute(f'DROP TABLE IF EXISTS "{name}"')
            con.executescript(_DDL)
            _reset(con)
        else:
            con.executescript(_DDL)
    except sqlite3.DatabaseError:  # damaged file: it is only an index - rebuild it
        con.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        con = sqlite3.connect(
            path, timeout=30, check_same_thread=False, isolation_level=None
        )
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(_DDL)
        _reset(con)
    _conns[key] = con
    _start_checkpointer(path)
    return con


_checkpointing: dict[str, threading.Event] = {}


def _start_checkpointer(path: Path, every_s: float = 15.0) -> None:
    """Background thread: moves the write-ahead log into the index file every few
    seconds (PASSIVE: never blocks a reader or writer), so no question pays for it.
    It opens the file only for the moment of each checkpoint."""
    key = str(path)
    if key in _checkpointing:
        return
    stop = _checkpointing[key] = threading.Event()

    def loop() -> None:
        while not stop.wait(every_s):
            try:
                c = sqlite3.connect(path, timeout=1)
                try:
                    c.execute("PRAGMA wal_checkpoint(PASSIVE)")
                finally:
                    c.close()
            except sqlite3.Error:
                pass  # busy or gone: try again next time

    threading.Thread(target=loop, name="index-checkpoint", daemon=True).start()


def _meta(con: sqlite3.Connection, k: str, default: str = "") -> str:
    row = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return str(row[0]) if row else default


def _set_meta(con: sqlite3.Connection, k: str, v: Any) -> None:
    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (k, str(v)))


def _reset(con: sqlite3.Connection) -> None:
    for t in ("ev", "tech", "link", "finding", "inc", "scache", "meta"):
        con.execute(f"DELETE FROM {t}")  # noqa: S608 - fixed table names
    con.execute("INSERT INTO hay(hay) VALUES ('delete-all')")  # contentless table
    _set_meta(con, "schema", SCHEMA)
    _set_meta(con, "off", 0)
    _istate.clear()


def invalidate() -> None:
    """The store was rewritten in place (rule re-check): rebuild the index from scratch
    on the next use."""
    with _lock:
        con = _connect()
        con.execute("BEGIN IMMEDIATE")
        _reset(con)
        con.execute("COMMIT")
    _istate.clear()


def _head(path: Path) -> str:
    """Identity of the store file: a hash of its first line (cleared + rewritten
    stores start differently)."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha1(fh.readline()).hexdigest()  # noqa: S324 - not security
    except OSError:
        return ""


def _row(
    ev: Event, line: str
) -> tuple[tuple[Any, ...], list[tuple[str, str, str, str, int]], str]:
    from src.soc.correlate import entity_of

    techs = ev.get("techniques") or []
    tech_rows = [
        ("T", str(t), str(t).upper(), str(t).upper().split(".")[0], k)
        for k, t in enumerate(techs)
    ]
    tech_rows += [
        ("C", str(t), str(t).upper(), str(t).upper().split(".")[0], k)
        for k, t in enumerate(ev.get("candidate_techniques") or [])
    ]
    row = (
        float(ev.get("ts") or 0),
        entity_of(ev),
        ev.get("src_ip") if isinstance(ev.get("src_ip"), str) else None,
        ev.get("dst_ip") if isinstance(ev.get("dst_ip"), str) else None,
        str(ev.get("user", "")).lower(),
        str(ev.get("host", "")).lower(),
        str(ev.get("sha256", "")).lower(),
        str(ev.get("kind", "?")),
        str(ev.get("source", "?")),  # as stored (status.py lists it as written)
        1 if techs else 0,
        ev.get("label") if isinstance(ev.get("label"), str) else None,
        None if ev.get("dst_port") is None else str(ev.get("dst_port")),
        line,
    )
    hay = f"{ev.get('detail', '')} {ev.get('kind', '')} {ev.get('file', '')}".lower()
    return row, tech_rows, hay


def _add(con: sqlite3.Connection, lines: list[str]) -> int:
    """Index these store lines; returns how many events were added."""
    from src.soc.riskboard import entities_of, family_of

    added = 0
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # same as engine.read_jsonl: unreadable lines are skipped
        if not isinstance(ev, dict):
            continue
        row, tech_rows, hay = _row(ev, line)
        added += 1
        cur = con.execute(
            "INSERT INTO ev(ts, entity, src_ip, dst_ip, user, host, sha, kind, source, "
            "alert, label, dport, js) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        i = cur.lastrowid
        if row[0] >= _FT_CUTOFF[0]:  # lean index: old events get no full-text row
            con.execute("INSERT INTO hay(rowid, h) VALUES (?, ?)", (i, hay))
        if tech_rows:
            con.executemany(
                "INSERT INTO tech(id, typ, raw, up, base, pos) VALUES (?,?,?,?,?,?)",
                [(i, *t) for t in tech_rows],
            )
        keys = entities_of(ev)
        pairs = [(a, b) for a in keys for b in keys if a != b]
        if pairs:
            con.executemany(
                "INSERT OR IGNORE INTO link(a, b, first_id) VALUES (?, ?, ?)",
                [(a, b, i) for a, b in pairs],
            )
        if ev.get("techniques"):
            f = (
                float(ev.get("ts", 0.0)),
                family_of(ev),
                json.dumps(list(ev["techniques"])),
                str(ev.get("detail", ""))[:120],
            )
            obs = str(ev.get("observer") or ev.get("host") or "").lower()
            con.executemany(
                "INSERT INTO finding(key, ts, family, techs, what) VALUES (?,?,?,?,?)",
                [(k, *f) for k in keys] + ([(f"obs:{obs}", *f)] if obs else []),
            )
    return added


BULK_BYTES = 256 * 2**20  # a catch-up bigger than this loads first, indexes after
# lean index (owner 2026-10-24): full-text rows only for the last N days (Performance
# page, 0 = all). Older events stay searchable by every field; a text search reaching
# further back reads their stored JSON instead (slower, same results).
_FT_CUTOFF = [0.0]  # events older than this have no full-text row (0 = all have one)
PRUNE_EVERY_S = 3600.0


def fulltext_days() -> int:
    from src.soc import perf

    try:
        return max(0, int(perf.load_settings().get("fulltext_days", 0)))
    except (TypeError, ValueError):
        return 0


def _refresh_cutoff(con: sqlite3.Connection) -> None:
    """Set the full-text cutoff from the setting and prune older rows (hourly)."""
    import time as _t

    days = fulltext_days()
    if not days:
        _FT_CUTOFF[0] = 0.0
        if _meta(con, "ft_cutoff", "0") != "0":  # switched back to 'all': rebuild the rows
            con.execute("BEGIN IMMEDIATE")
            for i, line in con.execute("SELECT id, js FROM ev WHERE ts < ?", (float(_meta(con, "ft_cutoff")),)).fetchall():
                ev = json.loads(line)
                hay = f"{ev.get('detail', '')} {ev.get('kind', '')} {ev.get('file', '')}".lower()
                con.execute("INSERT OR REPLACE INTO hay(rowid, h) VALUES (?, ?)", (i, hay))
            _set_meta(con, "ft_cutoff", "0")
            con.execute("COMMIT")
        return
    cutoff = _t.time() - days * 86_400
    _FT_CUTOFF[0] = cutoff
    if _t.time() - float(_meta(con, "ft_pruned", "0")) < PRUNE_EVERY_S:
        return
    con.execute("BEGIN IMMEDIATE")
    con.execute("DELETE FROM hay WHERE rowid IN (SELECT id FROM ev WHERE ts < ?)", (cutoff,))
    _set_meta(con, "ft_cutoff", cutoff)
    _set_meta(con, "ft_pruned", _t.time())
    con.execute("COMMIT")


def fulltext_cutoff() -> float:
    """Events before this time have no full-text row (0: every event has one)."""
    return _FT_CUTOFF[0]
_INDEX_NAMES = re.findall(r"CREATE INDEX IF NOT EXISTS (\w+)", _DDL)


def sync() -> sqlite3.Connection:
    """Catch the index up with the store (only the new bytes) and return it."""
    with _lock:
        con = _connect()
        path = _store()
        try:
            st = path.stat()
        except OSError:
            if _meta(con, "off", "0") != "0":
                con.execute("BEGIN IMMEDIATE")
                _reset(con)
                con.execute("COMMIT")
            return con
        off = int(_meta(con, "off", "0"))
        head = _head(path)
        _refresh_cutoff(con)
        if off == st.st_size and _meta(con, "head") == head:
            return con  # nothing new: the common case, ~0.1 ms
        con.execute("BEGIN IMMEDIATE")
        try:
            off = int(_meta(con, "off", "0"))  # another process may have caught up
            if st.st_size < off or (off and _meta(con, "head") != head):
                _reset(con)  # cleared or replaced store: rebuild
                off = 0
            _set_meta(con, "head", head)
            added = counted = 0
            bulk = st.st_size - off > BULK_BYTES
            if bulk:  # load first, build the lookup indexes once at the end
                for name in _INDEX_NAMES:
                    con.execute(f"DROP INDEX IF EXISTS {name}")
            with open(path, "rb") as fh:
                fh.seek(off)
                while True:
                    block = fh.read(CHUNK)
                    cut = block.rfind(b"\n") + 1  # only complete lines
                    if not cut:
                        break  # nothing new, or a half-written last line
                    added += _add(
                        con, block[:cut].decode("utf-8", errors="replace").splitlines()
                    )
                    _set_meta(con, "n", int(_meta(con, "n", "0")) + added - counted)
                    counted = added
                    off += cut
                    fh.seek(off)
                    # commit per chunk: a first build of millions of events is
                    # resumable and other processes are not locked out for minutes
                    _set_meta(con, "off", off)
                    con.execute("COMMIT")
                    con.execute("BEGIN IMMEDIATE")
            _set_meta(con, "off", off)
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        if bulk:  # (re)create every index in one pass (also after an interrupted build)
            con.executescript(_DDL)
        if added >= 20_000:  # fresh statistics so SQLite picks the right index
            con.execute("PRAGMA analysis_limit=400")
            con.execute("ANALYZE")
        return con


def _rows(cur: sqlite3.Cursor) -> list[Event]:
    return [json.loads(r[0]) for r in cur]


# --- stats -----------------------------------------------------------------------


def stats(since: float | None = None) -> dict[str, Any]:
    con = sync()
    with _lock:
        where, args = ("WHERE ts >= ?", [since]) if since is not None else ("", [])
        if since is None:  # the whole store: count kept by sync(), ends from the index
            n = int(_meta(con, "n", "0"))
            lo, hi = con.execute("SELECT min(ts), max(ts) FROM ev").fetchone()
        else:
            n, lo, hi = con.execute(
                f"SELECT count(*), min(ts), max(ts) FROM ev {where}",  # noqa: S608
                args,
            ).fetchone()
        srcs = sorted(
            r[0]
            for r in con.execute(f"SELECT DISTINCT source FROM ev {where}", args)  # noqa: S608
        )
    return {"events": int(n), "first": lo, "last": hi, "sources": srcs}


def recent(n: int) -> list[Event]:
    """The newest ``n`` stored events, oldest first (as the store lists them)."""
    con = sync()
    with _lock:
        rows = _rows(
            con.execute("SELECT js FROM ev ORDER BY id DESC LIMIT ?", (int(n),))
        )
    return rows[::-1]


def between(start: float, end: float) -> list[Event]:
    """Stored events with start <= ts < end, in store order."""
    con = sync()
    with _lock:
        return _rows(
            con.execute(
                "SELECT js FROM ev WHERE ts >= ? AND ts < ? ORDER BY id", (start, end)
            )
        )


def count() -> int:
    con = sync()
    with _lock:
        return int(_meta(con, "n", "0"))


# --- search ----------------------------------------------------------------------


def _fts_query(terms: list[str]) -> str:
    return " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)


def _distinct_like(con: sqlite3.Connection, col: str, needles: list[str]) -> list[str]:
    """Distinct stored values of ``col`` containing any needle (hosts, sources are
    few, so the substring test runs on the distinct values, not on every event)."""
    vals = [r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM ev")]  # noqa: S608
    return [v for v in vals if v is not None and any(n in v.lower() for n in needles)]


def _where(
    con: sqlite3.Connection, f: dict[str, Any]
) -> tuple[str, str, str, list[Any], bool]:
    """(FROM, WHERE, ORDER BY column, arguments, recheck) for the filters.

    The rows stream newest-stored first from the most useful index: the technique
    table when a technique is asked for, else the full-text index for text, else
    the events themselves - so a broad search stops as soon as it has CAP matches
    instead of collecting millions first. ``recheck``: text shorter than 3
    characters (trigrams cannot index it) is checked in Python."""
    from src.soc.correlate import technique_info

    conds: list[str] = []
    args: list[Any] = []
    frm, order = "ev", "ev.id"

    def among(col: str, vals: list[Any]) -> None:
        conds.append(f"{col} IN ({','.join('?' * len(vals))})" if vals else "0")
        args.extend(vals)

    if "techniques" in f:
        bases = sorted({x.split(".")[0] for x in f["techniques"]})
        frm, order = "tech JOIN ev ON ev.id = tech.id", "tech.id"
        conds.append("tech.typ = 'T'")
        among("tech.base", bases)
    long = [t for t in f.get("text", []) if len(t) >= 3]
    recheck = len(long) < len(f.get("text", []))
    cut = _FT_CUTOFF[0]
    if long and cut and float(f.get("start", 0.0)) < cut:
        # lean index: events older than the cutoff have no full-text row - read their
        # stored JSON (a superset; the caller re-checks every term exactly)
        plain = all(t.isascii() and '"' not in t and "\\" not in t for t in long)
        like = " AND ".join("lower(ev.js) LIKE ?" for _ in long) if plain else "1"
        conds.append(f"(ev.id IN (SELECT rowid FROM hay WHERE hay MATCH ?) OR (ev.ts < ? AND {like}))")
        args += [_fts_query(long), cut, *([f"%{t.lower()}%" for t in long] if plain else [])]
        recheck = True
    elif long:
        if frm == "ev":
            frm, order = "hay JOIN ev ON ev.id = hay.rowid", "hay.rowid"
            conds.append("hay MATCH ?")
        else:
            conds.append("ev.id IN (SELECT rowid FROM hay WHERE hay MATCH ?)")
        args.append(_fts_query(long))
    if "start" in f:
        conds.append("ev.ts >= ?")
        args.append(f["start"])
        # SQLite cannot estimate a time range; when the window holds few events,
        # walk the time index instead of a busier one (kind, user, ...)
        if frm == "ev":
            few = con.execute(
                "SELECT count(*) FROM (SELECT 1 FROM ev WHERE ts >= ? LIMIT 50001)",
                (f["start"],),
            ).fetchone()[0]
            if few <= 50_000:
                frm = "ev INDEXED BY ev_ts"
    if "end" in f:
        conds.append("ev.ts < ?")
        args.append(f["end"])
    if "ips" in f:
        q = ",".join("?" * len(f["ips"]))
        conds.append(
            f"ev.id IN (SELECT id FROM ev WHERE src_ip IN ({q}) "
            f"UNION SELECT id FROM ev WHERE dst_ip IN ({q}))"
        )
        args += list(f["ips"]) * 2
    if "user" in f:
        conds.append("ev.user = ?")
        args.append(f["user"])
    if "host" in f:
        among("ev.host", _distinct_like(con, "host", [f["host"]]))
    if "sha256" in f:
        among("ev.sha", list(f["sha256"]))
    if f.get("alerts_only"):
        conds.append("ev.alert = 1")
    if "tactics" in f:
        ups = [
            r[0]
            for r in con.execute("SELECT DISTINCT up FROM tech WHERE typ='T'")
            if technique_info(r[0])[1] in set(f["tactics"])
        ]
        conds.append(
            "ev.id IN (SELECT id FROM tech WHERE typ='T' AND up IN "
            f"({','.join('?' * len(ups))}))"
            if ups
            else "0"
        )
        args += ups
    if "sources" in f:
        among("ev.source", _distinct_like(con, "source", list(f["sources"])))
    if "kinds" in f:
        among("ev.kind", list(f["kinds"]))
    return frm, (" AND ".join(conds) or "1"), order, args, recheck


def search(q: str, now: float | None = None, exact: bool = False) -> dict[str, Any]:
    """src.soc.search.run(q, all_events) from the index: identical matches, newest
    first (same tie order). ``count`` is exact up to CAP; above it the newest CAP
    stored matches are used and ``capped`` is set, unless ``exact=True``. Only the
    rows shown are read in full."""
    from collections import Counter

    from src.soc import search as s

    f = s.parse(q, now)
    con = sync()
    with _lock:
        frm, where, order, args, recheck = _where(con, f)
        cols = "ev.id, ev.ts, ev.source, ev.kind" + (", ev.js" if recheck else "")
        cur = con.execute(
            f"SELECT {cols} FROM {frm} WHERE {where} ORDER BY {order} DESC",  # noqa: S608
            args,
        )
        seen: set[int] = set()
        cands: list[tuple[int, float, str, str]] = []
        capped = False
        for row in cur:
            if row[0] in seen:  # one event with two matching techniques
                continue
            seen.add(row[0])
            if recheck and not s._match(json.loads(row[4]), f):
                continue
            if len(cands) == CAP:
                capped = True
                break
            cands.append((row[0], row[1], row[2], row[3]))
        # newest first; equal times keep the store order, as run() does
        cands.sort(key=lambda r: (-r[1], r[0]))
        top = [r[0] for r in cands[: s.MAX_ROWS]]
        js = (
            dict(
                con.execute(
                    f"SELECT id, js FROM ev WHERE id IN ({','.join('?' * len(top))})",  # noqa: S608
                    top,
                ).fetchall()
            )
            if top
            else {}
        )
    shown = [json.loads(js[i]) for i in top]
    n = len(cands)
    if capped and exact:
        n = search_count(q, now)
    res = s.result(
        f,
        shown,
        n,
        by_source=dict(Counter(r[2] for r in cands).most_common(8)),
        by_kind=dict(Counter(r[3] for r in cands).most_common(8)),
    )
    # above CAP: the rows and per-source counts are from the newest CAP stored matches
    res["capped"] = capped and not exact
    res["sample"] = len(cands)
    return res


def search_count(q: str, now: float | None = None) -> int:
    """Exact number of matches (the 'count exactly' button)."""
    from src.soc import search as s

    f = s.parse(q, now)
    con = sync()
    with _lock:
        frm, where, _order, args, recheck = _where(con, f)
        if not recheck:
            sql = f"SELECT count(DISTINCT ev.id) FROM {frm} WHERE {where}"  # noqa: S608
            return int(con.execute(sql, args).fetchone()[0])
        sql = f"SELECT DISTINCT ev.id, ev.js FROM {frm} WHERE {where}"  # noqa: S608
        return sum(
            1 for _i, js in con.execute(sql, args) if s._match(json.loads(js), f)
        )


# --- evidence for "is X happening?" ------------------------------------------------


def _evidence_sql(
    techs: list[str], family: str | None, after: int, since: float | None
) -> tuple[str, list[Any]]:
    """Events that src.nlp.status._match could count: carrying one of the techniques
    (or candidate techniques), or a detector alert labelled ``family``."""
    cond = f"id IN (SELECT id FROM tech WHERE raw IN ({','.join('?' * len(techs))}) AND id > ?)"
    args: list[Any] = [*techs, after]
    if family is not None:
        cond = f"({cond} OR (source = 'detector' AND label = ?))"
        args.append(family)
    sql = f"SELECT js FROM ev WHERE id > ? AND {cond}"  # noqa: S608
    args = [after, *args]
    if since is not None:
        sql += " AND ts >= ?"
        args.append(since)
    return sql + " ORDER BY id", args


def evidence(techniques: list[str], family: str | None) -> list[Event]:
    """Every stored event that could count as evidence (see ``_evidence_sql``)."""
    con = sync()
    with _lock:
        sql, args = _evidence_sql(sorted(set(techniques)), family, 0, None)
        return _rows(con.execute(sql, args))


def status_state(
    techniques: list[str], family: str | None, since: float | None = None
) -> dict[str, Any]:
    """The running evidence summary of src.nlp.status for this question. Over the
    whole store it is kept in the index and only new events are added; with a time
    window (``since``) it is built from the events in that window."""
    from src.nlp import status

    techs = sorted(set(techniques))
    con = sync()
    with _lock:
        if since is not None:
            sql, args = _evidence_sql(techs, family, 0, since)
            return status.accumulate(
                status.new_state(), _rows(con.execute(sql, args)), techs, family
            )
        key = json.dumps([techs, family])
        top = int(con.execute("SELECT coalesce(max(id), 0) FROM ev").fetchone()[0])
        row = con.execute("SELECT upto, state FROM scache WHERE k=?", (key,)).fetchone()
        upto, state = (
            (int(row[0]), json.loads(row[1])) if row else (0, status.new_state())
        )
        if upto > top:  # the index was rebuilt
            upto, state = 0, status.new_state()
        if upto < top:
            sql, args = _evidence_sql(techs, family, upto, None)
            sql = sql.replace(" ORDER BY id", " AND id <= ? ORDER BY id")
            state = status.accumulate(
                state, _rows(con.execute(sql, [*args, top])), techs, family
            )
            con.execute(
                "INSERT OR REPLACE INTO scache(k, upto, state) VALUES (?, ?, ?)",
                (key, top, json.dumps(state)),
            )
        return state


# --- incidents --------------------------------------------------------------------


class _DbBoard:
    """riskboard.Board answered from the index: the same boost and reasons, but
    only the first finding per model family is read (not every finding)."""

    def __init__(self, con: sqlite3.Connection, **kw: Any) -> None:
        from src.soc.riskboard import Board

        self.b = Board(**kw)
        self.con = con
        # link look-ups, kept between calls (incidents() hands them over); keys that
        # received new links are dropped (forget) before they are used again
        self._deg: dict[str, int] = {}
        self._nb: dict[str, list[str]] = {}

    def forget(self, keys: set[str]) -> None:
        for k in keys:
            self._deg.pop(k, None)
            self._nb.pop(k, None)

    def degree(self, key: str) -> int:
        if key not in self._deg:
            self._deg[key] = int(
                self.con.execute(
                    "SELECT count(*) FROM link WHERE a=?", (key,)
                ).fetchone()[0]
            )
        return self._deg[key]

    def hub(self, key: str) -> bool:
        return self.degree(key) > self.b.max_degree

    def neighbours(self, key: str) -> list[str]:
        if key not in self._nb:
            self._nb[key] = [
                r[0] for r in self.con.execute("SELECT b FROM link WHERE a=?", (key,))
            ]
        return self._nb[key]

    def boost(
        self,
        keys: list[str],
        own: set[str],
        start: float,
        end: float,
        primary: str | None = None,
        victims: tuple[str, ...] | list[str] = (),
    ) -> tuple[float, list[str]]:
        """riskboard.Board.boost: families with findings on these entities or one
        link away (hubs skipped unless primary) within the window, plus (follow_s >
        0) the findings logged by the attacked machines from the start until
        follow_s after the end; for each family the reason quotes its first finding
        (entities in sorted order)."""
        from src.soc.riskboard import Finding

        usable = {k for k in keys if k == primary or not self.hub(k)}
        near = set(usable)
        for k in usable if self.b.hops else ():
            if not self.hub(k):
                near |= {n for n in self.neighbours(k) if not self.hub(n)}
        lo, hi = start - self.b.window_s, end + self.b.window_s
        first: dict[str, tuple[str, int]] = {}  # family -> (entity, finding rowid)
        ordered = sorted(near)
        for k0 in range(0, len(ordered), 900):
            part = ordered[k0 : k0 + 900]
            for key, fam, rid in self.con.execute(
                "SELECT key, family, min(rowid) FROM finding WHERE key IN "
                f"({','.join('?' * len(part))}) AND ts BETWEEN ? AND ? GROUP BY key, family",  # noqa: S608
                [*part, lo, hi],
            ):
                if fam not in first or key < first[fam][0]:
                    first[fam] = (key, rid)
        if self.b.follow_s > 0 and victims:
            vk = [f"obs:{str(v).lower()}" for v in victims]
            for key, fam, rid in self.con.execute(
                "SELECT key, family, min(rowid) FROM finding WHERE key IN "
                f"({','.join('?' * len(vk))}) AND ts BETWEEN ? AND ? GROUP BY key, family",  # noqa: S608
                [*vk, start, end + self.b.follow_s],
            ):
                if fam not in first:
                    first[fam] = (key.replace("obs:", "victim:", 1), rid)
        other = sorted(set(first) - own - {"unknown", "other"})
        pts = min(self.b.cap, self.b.weight * len(other))
        reasons = []
        for fam in other:
            ent, rid = first[fam]
            ts, techs, what = self.con.execute(
                "SELECT ts, techs, what FROM finding WHERE rowid=?", (rid,)
            ).fetchone()
            f = Finding(fam, ts, json.loads(techs), what)
            reasons.append(f"{fam} finding on {ent.split(':', 1)[0]} {ent.split(':', 1)[1][:60]}: "
                           f"{', '.join(f.techniques[:3]) or 'flagged'} ({f.what[:80]})")  # fmt: skip
        return pts, reasons


_COLS = "id, ts, alert, kind, source, dport"
SCAN_MIN = 20  # correlate.group_incident's scan threshold (distinct targets)


def _may_be_incident(g: list[tuple[Any, ...]]) -> bool:
    """Can this time-group become an incident? A superset of every rule in
    correlate.group_incident, read from columns (no event is decoded): an event
    with techniques, a failed logon (T1110 -> T1078 chain), cloud API calls
    (reconnaissance / denied probing), 50+ web requests (4xx scanning) or 20+
    events with a destination port (scan). Every other group returns None there."""
    ports = web = dns = ptr = 0
    for _i, _ts, alert, kind, source, dport in g:
        if alert or kind == "failed_logon" or source == "cloudtrail":
            return True
        ports += dport is not None
        web += kind == "http_request"
        dns += kind == "dns_query"
        ptr += kind == "dns_ptr_query"
    from src.soc.correlate import DNS_ENUM_MIN, PTR_SWEEP_MIN

    return ports >= SCAN_MIN or web >= 50 or dns >= DNS_ENUM_MIN or ptr >= PTR_SWEEP_MIN


def _groups(rows: list[tuple[Any, ...]], win: float) -> list[list[tuple[Any, ...]]]:
    """One entity's rows (sorted by ts, id) -> time groups split at gaps > win."""
    from src.soc.correlate import MAX_GROUP_S

    groups: list[list[tuple[Any, ...]]] = []
    for r in rows:  # same split as correlate._split_by_gap (gap > win or > MAX_GROUP_S long)
        if groups and r[1] - groups[-1][-1][1] <= win and r[1] - groups[-1][0][1] <= MAX_GROUP_S:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def _store_group(
    con: sqlite3.Connection, win: float, entity: str, g: list[tuple[Any, ...]]
) -> None:
    """Decode the group's events and store its incident (if it is one)."""
    from src.soc.correlate import group_incident

    if not _may_be_incident(g):
        return
    ids = [r[0] for r in g]
    evs: list[Event] = []
    for k in range(0, len(ids), 900):  # SQLite parameter limit
        part = ids[k : k + 900]
        rows = dict(
            con.execute(
                f"SELECT id, js FROM ev WHERE id IN ({','.join('?' * len(part))})",  # noqa: S608
                part,
            ).fetchall()
        )
        evs += [json.loads(rows[i]) for i in part]
    inc = group_incident(entity, evs)
    if inc is not None:
        con.execute(
            "INSERT INTO inc(win, entity, start, end_, data, sev) VALUES (?,?,?,?,?,?)",
            (win, entity, inc["start"], inc["end"], json.dumps(inc), float(inc["_sev"])),
        )


def _group_start(con: sqlite3.Connection, win: float, entity: str, new_from: int) -> float | None:
    """Start time of the time-group that holds the entity's earliest new event
    (2026-10-17: +1,000 events no longer re-read whole entity histories). Walks back
    from that event until a gap > ``win`` - exactly where _groups splits."""
    first = con.execute(
        "SELECT min(ts) FROM ev WHERE entity=? AND id>?", (entity, new_from)
    ).fetchone()[0]
    if first is None:
        return None
    start = float(first)
    for (t,) in con.execute(
        "SELECT ts FROM ev WHERE entity=? AND ts<? ORDER BY ts DESC", (entity, first)
    ):
        if start - t > win:
            break
        start = float(t)
    return start


def _recompute(con: sqlite3.Connection, win: float, entity: str, new_from: int) -> None:
    """Re-derive the incidents of ``entity`` whose time-groups contain events with
    id > ``new_from``."""
    since = _group_start(con, win, entity, new_from) if new_from else None
    if since is None:
        rows = con.execute(
            f"SELECT {_COLS} FROM ev WHERE entity=? ORDER BY ts, id",
            (entity,),  # noqa: S608
        ).fetchall()
    else:  # groups before the one holding the earliest new event cannot change
        rows = con.execute(
            f"SELECT {_COLS} FROM ev WHERE entity=? AND ts>=? ORDER BY ts, id",
            (entity, since),  # noqa: S608
        ).fetchall()
    for g in _groups(rows, win):
        if new_from and not any(r[0] > new_from for r in g):
            continue
        con.execute(
            "DELETE FROM inc WHERE win=? AND entity=? AND start>=? AND end_<=?",
            (win, entity, g[0][1], g[-1][1]),
        )
        _store_group(con, win, entity, g)


def _build_all(con: sqlite3.Connection, win: float) -> None:
    """First build for a window: one ordered pass over all events (entity, time),
    so millions of events need no per-entity queries; only groups that can be
    incidents are decoded."""
    cur = con.execute(f"SELECT entity, {_COLS} FROM ev ORDER BY entity, ts, id")  # noqa: S608
    ent: str | None = None
    rows: list[tuple[Any, ...]] = []
    for r in cur:
        if r[0] != ent:
            if ent is not None:
                for g in _groups(rows, win):
                    _store_group(con, win, ent, g)
            ent, rows = r[0], []
        rows.append(r[1:])
    if ent is not None:
        for g in _groups(rows, win):
            _store_group(con, win, ent, g)


# (db, window) -> what the last answer was built from, to update only what changed
_istate: dict[tuple[str, float], dict[str, Any]] = {}


def incidents(
    window_s: float = 900.0, scale: float | None = None
) -> list[dict[str, Any]]:
    """src.soc.correlate.correlate(all events, window_s, board=riskboard.board_for(...))
    from the index. New events only make their entities' time-groups be recomputed,
    and only incidents whose entities (or entities one link away) received new
    events get their risk-board boost re-read - same answer, milliseconds."""
    from src.soc import correlate, riskboard

    con = sync()
    win = float(window_s)
    if scale is None:
        from src.soc import places

        scale = places.board_scale()
    with _lock:
        top = int(con.execute("SELECT coalesce(max(id), 0) FROM ev").fetchone()[0])
        done = int(_meta(con, f"inc_upto:{win}", "0"))
        ents: list[str] = []
        if top != done:
            con.execute("BEGIN IMMEDIATE")
            try:
                done = int(_meta(con, f"inc_upto:{win}", "0"))
                if done > top:  # index rebuilt since: start over for this window
                    con.execute("DELETE FROM inc WHERE win=?", (win,))
                    done = 0
                if done == 0:  # first build for this window: one ordered pass
                    con.execute("DELETE FROM inc WHERE win=?", (win,))
                    _build_all(con, win)
                    ents = [r[0] for r in con.execute("SELECT DISTINCT entity FROM ev")]
                else:
                    ents = [
                        r[0]
                        for r in con.execute(
                            "SELECT DISTINCT entity FROM ev WHERE id > ?", (done,)
                        )
                    ]
                    for ent in ents:
                        _recompute(con, win, ent, done)
                _set_meta(con, f"inc_upto:{win}", top)
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        s = riskboard.settings()
        board = _DbBoard(
            con,
            weight=float(s["weight"]) * max(0.0, min(1.0, scale)),
            cap=float(s["cap"]),
            window_s=float(s["window_s"]),
            max_degree=int(s["max_degree"]),
            follow_s=float(s["follow_s"]),
            hops=int(s["hops"]),
        )
        cfg = (json.dumps(s, sort_keys=True), scale)
        key = (str(db_path()), win)
        n_inc = int(con.execute("SELECT count(*) FROM inc WHERE win=?", (win,)).fetchone()[0])
        if n_inc > MAX_IN_MEMORY:  # very large store: the most severe ones, memory bounded
            return _top_incidents(con, board, win, top, key, cfg, n_inc, float(s["cap"]))
        LAST_VIEW.update(total=n_inc, shown=n_inc)
        st = _istate.get(key)
        full = st is None or st["cfg"] != cfg or st["top"] > top
        if not full and st is not None:
            board._deg, board._nb = st["deg"], st["nb"]
        if not full and st is not None and st["top"] < top:
            full = _update(con, board, st, ents, top)
        if full:
            pre = {
                p["id"]: p
                for p in (
                    json.loads(r[0])
                    for r in con.execute("SELECT data FROM inc WHERE win=?", (win,))
                )
            }
            final = {p["id"]: correlate.finish_one(p, board) for p in pre.values()}
            st = {
                "cfg": cfg,
                "top": top,
                "win": win,
                "pre": pre,
                "final": final,
                "keys": _key_map(pre),
            }
            _istate[key] = st
        assert st is not None
        changed = full or st.get("sorted_at") != top
        st["top"] = top
        st["deg"], st["nb"] = board._deg, board._nb
        if changed or "sorted" not in st:  # re-sort only when something changed
            st["sorted"] = sorted(st["final"].values(), key=correlate.incident_order)
            st["sorted_at"] = top
        out = st["sorted"]
    # the cached incidents themselves (no per-call copies: at a million events there
    # are tens of thousands); callers only add the 'allowlisted' mark, set every call
    return list(out)


# above this many incidents only the most severe MAX_IN_MEMORY are held (2026-10-23: the
# synthetic 10M-event store had 4.98M incidents, ~20 GB as Python objects)
MAX_IN_MEMORY = 200_000
LAST_VIEW: dict[str, int] = {"total": 0, "shown": 0}  # what the last incidents() call covered


def _top_incidents(con: sqlite3.Connection, board: Any, win: float, top: int, key: tuple[str, float],
                   cfg: Any, n_inc: int, cap: float) -> list[dict[str, Any]]:  # fmt: skip
    """The MAX_IN_MEMORY most severe incidents, in the same order and with the same
    risk-board boosts as the full view. Rows come by pre-board severity (descending);
    the board adds at most ``cap``, so once a row's severity + cap cannot reach the
    weakest kept incident no later row can either. Memory: one heap of MAX_IN_MEMORY."""
    import heapq

    from src.soc import correlate

    st = _istate.get(key)
    if st is not None and st.get("cfg") == cfg and st.get("top") == top and st.get("bounded"):
        return list(st["sorted"])  # nothing new since the last view
    heap: list[tuple[tuple[float, float, str], int, dict[str, Any]]] = []
    seq = 0
    for data, sev in con.execute("SELECT data, sev FROM inc WHERE win=? ORDER BY sev DESC, end_ DESC", (win,)):
        if len(heap) >= MAX_IN_MEMORY and float(sev) + cap < heap[0][0][0]:
            break
        inc = correlate.finish_one(json.loads(data), board)
        k = correlate.incident_order(inc)  # smaller = more severe
        item = ((-k[0], -k[1], _neg(k[2])), seq, inc)  # heap keeps the least severe on top
        seq += 1
        if len(heap) < MAX_IN_MEMORY:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    out = sorted((x[2] for x in heap), key=correlate.incident_order)
    _istate[key] = {"cfg": cfg, "top": top, "bounded": True, "sorted": out}
    LAST_VIEW.update(total=n_inc, shown=len(out))
    return list(out)


def _neg(s: str) -> str:
    """Reverses string order inside a tuple key (ids sort ascending in incident_order)."""
    return "".join(chr(0x10FFFF - ord(c)) for c in s) + chr(0x10FFFF)  # end mark: a prefix sorts after


def _key_map(pre: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    """entity key -> ids of the incidents that mention it."""
    m: dict[str, set[str]] = {}
    for i, p in pre.items():
        for k in p["_keys"]:
            m.setdefault(k, set()).add(i)
    return m


def _update(
    con: sqlite3.Connection,
    board: _DbBoard,
    st: dict[str, Any],
    ents: list[str],
    top: int,
) -> bool:
    """Bring the cached answer from st['top'] up to ``top``. Returns True when only
    a full re-read is exact (an entity crossed the hub threshold)."""
    from src.soc import correlate
    from src.soc.riskboard import entities_of

    old = int(st["top"])
    recomputed = set(ents)
    # 1. incidents of the recomputed entities: replace their board-free form
    changed: set[str] = set()
    for i in [i for i, p in st["pre"].items() if p["src_ip"] in recomputed]:
        st["pre"].pop(i)
        st["final"].pop(i, None)
    for ent in ents:
        for (data,) in con.execute(
            "SELECT data FROM inc WHERE win=? AND entity=?", (st["win"], ent)
        ):
            p = json.loads(data)
            st["pre"][p["id"]] = p
            changed.add(p["id"])
    # 2. entity keys of the new events; a key that became a hub changes everything
    new_keys: set[str] = set()
    for (js,) in con.execute("SELECT js FROM ev WHERE id > ? AND id <= ?", (old, top)):
        new_keys |= set(entities_of(json.loads(js)))
    board.forget(new_keys)  # their links changed: look them up again
    hubs = set()
    for k in new_keys:
        if board.hub(k):
            before = int(
                con.execute(
                    "SELECT count(*) FROM link WHERE a=? AND first_id <= ?", (k, old)
                ).fetchone()[0]
            )
            if before <= board.b.max_degree:
                return True  # crossed the threshold: reachability changed everywhere
            hubs.add(k)
    touch = {k for k in new_keys if k not in hubs}
    for k in list(touch):
        touch |= {n for n in board.neighbours(k) if not board.hub(n)}
    st["keys"] = _key_map(st["pre"])
    affected = set(changed)
    for k in touch:
        affected |= st["keys"].get(k, set())
    affected |= {i for i, p in st["pre"].items() if p["_primary"] in hubs}
    for i in affected:
        if i in st["pre"]:
            st["final"][i] = correlate.finish_one(st["pre"][i], board)
    return False
