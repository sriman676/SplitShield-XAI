"""The event index (src/soc/eventindex.py) answers exactly like the whole-list code:
search, 'is X happening?' evidence and incidents (with the risk board), after every
append, out-of-order timestamps, a cleared store and a damaged index file."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.nlp import status
from src.soc import correlate, engine, eventindex, riskboard, search

NOW = 1_800_000_000.0
TECHS = ["T1110", "T1110.003", "T1059.001", "T1021", "T1003", "T1046", "T1071", "T1190",
         "T1078", "T1070", "T1053"]  # fmt: skip
KINDS = ["failed_logon", "successful_logon", "process_creation", "http_request",
         "fw_drop", "conn", "ListBuckets", "DescribeInstances", "service_installed"]  # fmt: skip
SOURCES = ["winsec", "sysmon", "suricata", "zeek", "cloudtrail", "firewall", "detector",
           "syslog", "phishing"]  # fmt: skip
WORDS = ["failed password for root", "powershell -enc AAAA", "mimikatz sekurlsa",
         "AccessDenied by policy", "GET /admin -> 404", "dns query example.com",
         "Invalid user admin", "rundll32 c:\\temp\\x.dll", "ok"]  # fmt: skip


def _event(rng: random.Random, t0: float) -> dict:
    ev: dict = {
        "ts": t0 + rng.uniform(-7200, 7200),  # out of order on purpose
        "source": rng.choice(SOURCES),
        "kind": rng.choice(KINDS),
        "detail": rng.choice(WORDS) + f" #{rng.randrange(50)}",
        "techniques": rng.sample(TECHS, rng.choice([0, 0, 0, 1, 2])),
    }
    if rng.random() < 0.4:  # whole seconds: many events share a timestamp
        ev["ts"] = float(int(ev["ts"]) // 60 * 60)
    if rng.random() < 0.7:
        ev["src_ip"] = f"10.0.0.{rng.randrange(12)}"
    if rng.random() < 0.5:
        ev["user"] = rng.choice(["bob", "Alice", "root", "svc_backup", "admin"])
    if rng.random() < 0.5:
        ev["host"] = rng.choice(["WS01", "ws02", "SRV-DB", "pc7"])
    if rng.random() < 0.3:
        ev["dst_ip"] = f"192.168.1.{rng.randrange(40)}"
        ev["dst_port"] = rng.choice([22, 80, 443, 445, 3389, rng.randrange(1, 65535)])
    if rng.random() < 0.05:
        ev["sha256"] = rng.choice(["a" * 64, "B" * 64])
    if rng.random() < 0.05:
        ev["detail"] += " http://evil.example/x"
    if ev["source"] == "detector":
        ev["label"] = rng.choice(["DDoS", "PortScan", "Bot"])
        ev["tier"] = rng.choice(["confirmed", "low"])
        ev["candidate_techniques"] = rng.sample(TECHS, 1)
    return ev


@pytest.fixture(params=[6, 1000], ids=["hubs", "no-hubs"])
def store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Path]:
    monkeypatch.setattr(engine, "EVENTS", tmp_path / "events.jsonl")
    # a non-zero board weight so the risk-board boost is exercised too
    monkeypatch.setattr(
        riskboard,
        "settings",
        lambda: riskboard.DEFAULT
        | {"weight": 1.0, "cap": 2.0, "max_degree": request.param},
    )
    eventindex.close()
    yield engine.EVENTS
    eventindex.close()


def _append(path: Path, evs: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for e in evs:
            fh.write(json.dumps(e) + "\n")


def _all(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


# "after <clock>" is read in local time: a fixed "3 pm" missed every event on a UTC
# CI runner (17 < 18 hits) but not on a UTC+5:30 PC, so use the hour NOW falls in
CLOCK = time.strftime("%I %p", time.localtime(NOW)).lower()

QUERIES = [
    "failed logins", "show failed logins for bob", '"failed password"', "find powershell -enc",
    "events from 10.0.0.3", "events on host ws", "list T1110 events", "show alerts",
    "credential-access events", "zeek events", "events from sysmon", "show mimikatz",
    "events with ok", "logins from 10.0.0.1", "show processes on host SRV-DB",
    "events after " + CLOCK, "show T1059 alerts on host ws01", "find c:\\temp", "find " + "a" * 64,
    "show events", "find 404", "search x.dll",
]  # fmt: skip


def _same_search(path: Path) -> int:
    evs = _all(path)
    total = 0
    for q in QUERIES:
        a, b = search.run(q, evs, now=NOW), search.run_store(q, now=NOW)
        assert not b["capped"]
        for k in ("filters", "count", "by_source", "by_kind", "rows"):
            assert a[k] == b[k], (q, k)
        total += a["count"] > 0
    return total


def _same_incidents(path: Path, win: float) -> list[dict]:
    evs = _all(path)
    want = correlate.correlate(evs, win, board=riskboard.board_for(evs, scale=1.0))
    got = eventindex.incidents(win, scale=1.0)
    if got != want:  # name exactly what differs (a rare mismatch was seen under heavy load)
        g, w = {i["id"]: i for i in got}, {i["id"]: i for i in want}
        only = sorted(set(g) ^ set(w))[:5]
        fields = {i: sorted(k for k in set(g[i]) | set(w[i]) if g[i].get(k) != w[i].get(k)
                            or (k in g[i]) != (k in w[i]) or repr(g[i].get(k)) != repr(w[i].get(k)))
                  for i in set(g) & set(w) if g[i] != w[i]}  # fmt: skip
        order = [i["id"] for i in got] != [i["id"] for i in want]
        detail = {i: {k: (repr(g[i].get(k))[:200], repr(w[i].get(k))[:200]) for k in f} for i, f in list(fields.items())[:3]}
        raise AssertionError(f"index != full scan: only in one {only}; differing fields {fields}; "
                             f"order differs {order}; first differences {detail}")
    return got


def test_index_equals_full_scan_after_every_append(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    updates: list[bool] = []
    real_update = eventindex._update

    def spy(*a: object) -> bool:
        full = real_update(*a)  # type: ignore[arg-type]
        updates.append(full)
        return full

    monkeypatch.setattr(eventindex, "_update", spy)
    rng = random.Random(7)
    for batch in range(6):
        _append(
            store,
            [_event(rng, NOW + batch * 1800) for _ in range(rng.randrange(40, 160))],
        )
        hits = _same_search(store)
        incs = _same_incidents(store, 900.0)
        if batch % 2:
            _same_incidents(store, 3600.0)  # a second window kept side by side
        evs = _all(store)
        for techs, fam in (
            (["T1110"], None),
            (["T1046", "T1021"], "PortScan"),
            ([], "DDoS"),
        ):
            for since in (None, NOW):
                a = status.check(
                    evs, techs, fam, since=since, now=NOW, include_low=True
                )
                b = status.check_store(
                    techs, fam, since=since, now=NOW, include_low=True
                )
                assert a == b
    # not vacuous: real incidents, boosts from the risk board, most queries hit
    assert len(incs) > 20 and any(i["board_boost"] for i in incs)
    assert any(i["linked_findings"] for i in incs) and hits >= len(QUERIES) - 4
    assert False in updates  # the incremental path really ran (not only full re-reads)


def test_half_written_line_waits(store: Path) -> None:
    rng = random.Random(1)
    _append(store, [_event(rng, NOW) for _ in range(20)])
    with open(store, "a", encoding="utf-8") as fh:
        fh.write('{"ts": 1, "source": "zeek", "kind"')  # the monitor is mid-write
    assert eventindex.count() == 20
    with open(store, "a", encoding="utf-8") as fh:
        fh.write(': "conn", "detail": "x", "techniques": []}\n')
    assert eventindex.count() == 21


def test_cleared_or_replaced_store_rebuilds(store: Path) -> None:
    rng = random.Random(2)
    _append(store, [_event(rng, NOW) for _ in range(80)])
    _same_incidents(store, 900.0)
    store.unlink()  # engine.clear_store()
    assert eventindex.count() == 0 and eventindex.incidents(900.0, scale=1.0) == []
    _append(store, [_event(rng, NOW + 99999) for _ in range(120)])  # bigger than before
    assert eventindex.count() == 120
    _same_search(store)
    _same_incidents(store, 900.0)


def test_damaged_index_file_is_rebuilt(store: Path) -> None:
    rng = random.Random(3)
    _append(store, [_event(rng, NOW) for _ in range(30)])
    assert eventindex.count() == 30
    eventindex.close()
    for suffix in ("-wal", "-shm"):
        Path(f"{eventindex.db_path()}{suffix}").unlink(missing_ok=True)
    eventindex.db_path().write_bytes(b"this is not a database" * 100)
    assert eventindex.count() == 30


def test_big_counts_are_capped_unless_exact(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eventindex, "CAP", 25)
    rng = random.Random(4)
    evs = [_event(rng, NOW) for _ in range(200)]
    for e in evs:
        e["detail"] = "failed password for root"
    _append(store, evs)
    res = search.run_store('"failed password"', now=NOW)
    assert res["capped"] and res["count"] == 25 and len(res["rows"]) == 25
    assert "25+" in search.answer_text(res)
    exact = search.run_store('"failed password"', now=NOW, exact=True)
    assert not exact["capped"] and exact["count"] == 200
    assert eventindex.search_count('"failed password"', now=NOW) == 200


def test_bulk_first_build_gives_the_same_answers(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2026-10-16: big first builds load first and build the indexes at the end
    monkeypatch.setattr(eventindex, "BULK_BYTES", 1_000)
    rng = random.Random(11)
    _append(store, [_event(rng, NOW) for _ in range(400)])
    _same_search(store)
    _same_incidents(store, 900.0)
    con = eventindex.sync()
    names = {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert set(eventindex._INDEX_NAMES) <= names  # every index back after the load


def test_late_events_with_older_times_give_the_same_incidents(store: Path) -> None:
    """2026-10-17: new events only re-read their entity from the start of the
    time-group they land in - also when they are older than what is stored."""
    rng = random.Random(11)
    _append(store, [_event(rng, NOW + k * 600) for k in range(150)])
    _same_incidents(store, 900.0)
    for back in (5000, 300, 86400 * 2):  # late logs inside / between / before groups
        _append(store, [_event(rng, NOW - back + k * 60) for k in range(60)])
        _same_incidents(store, 900.0)


def test_nonstop_and_entityless_events_give_the_same_incidents(store: Path) -> None:
    """2026-10-23: a source active for days without a pause is split per day (MAX_GROUP_S)
    and entity-less events are grouped per source + file - the index agrees with the
    whole-list code, also when new events arrive in the middle of a long run."""
    rng = random.Random(23)
    run: list[dict[str, Any]] = [{"ts": NOW + k * 300.0, "source": "winsec", "kind": "failed_logon", "src_ip": "10.9.9.9",
            "detail": "failed", "techniques": ["T1110"]} for k in range(0, 700, 2)]  # ~2.4 days
    loose = [{"ts": NOW + k * 120.0, "source": rng.choice(["syslog", "zeek"]), "kind": "audit_log_cleared",
              "file": rng.choice(["a.log", "b.log"]), "detail": "cleared", "techniques": ["T1070"]}
             for k in range(200)]  # fmt: skip
    _append(store, run + loose)
    _same_incidents(store, 900.0)
    _append(store, [dict(e, ts=e["ts"] + 150.0) for e in run[::7]])  # fills gaps mid-run
    _same_incidents(store, 900.0)


def test_bounded_view_equals_the_top_of_the_full_view(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-10-23: above MAX_IN_MEMORY incidents only the most severe are held; they are
    exactly the first ones of the full view (same order, same risk-board boosts)."""
    rng = random.Random(31)
    _append(store, [_event(rng, NOW + k * 900) for k in range(900)])
    full = eventindex.incidents(900.0, scale=1.0)
    assert len(full) > 20
    for cap in (5, 20, len(full) - 1):
        monkeypatch.setattr(eventindex, "MAX_IN_MEMORY", cap)
        eventindex._istate.clear()
        top = eventindex.incidents(900.0, scale=1.0)
        assert [i["id"] for i in top] == [i["id"] for i in full[:cap]]
        assert top == full[:cap] and eventindex.LAST_VIEW == {"total": len(full), "shown": cap}
    assert eventindex._neg("ab") > eventindex._neg("abc") > eventindex._neg("abd")


def test_victim_follow_gives_the_same_incidents(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-10-24: with the risk board's victim follow on, the index (findings under
    obs:<observer>) agrees with the whole-list board."""
    monkeypatch.setattr(riskboard, "settings", lambda: riskboard.DEFAULT
                        | {"weight": 1.0, "cap": 2.0, "follow_s": 3600.0})  # fmt: skip
    rng = random.Random(31)
    evs = []
    for k in range(120):
        ev = _event(rng, NOW + k * 90)
        ev["observer"] = rng.choice(["web01", "db01", "mail01"])
        evs.append(ev)
    evs += [{"ts": NOW + k * 200.0, "source": "auditd", "kind": "process", "observer": "web01",
             "detail": "execve sh", "techniques": ["T1059.004"]} for k in range(30)]  # fmt: skip
    _append(store, evs)
    got = _same_incidents(store, 900.0)
    assert any(any("victim" in r for r in i["linked_findings"]) for i in got)


def test_lean_fulltext_index_gives_the_same_search_results(store: Path) -> None:
    """2026-10-24: with full-text rows only for the last day, text searches that reach
    older events read their stored JSON instead - the same matches as the full index."""
    import time as _t

    from src.soc import perf

    rng = random.Random(41)
    now = _t.time()
    _append(store, [_event(rng, now - 10 * 86400 + k * 30) for k in range(200)]
            + [_event(rng, now - 600 + k) for k in range(200)])  # fmt: skip
    full = {q: search.run_store(q, now=now)["count"] for q in QUERIES}
    perf.save_settings("auto", fulltext_days=1)
    eventindex.close()
    con = eventindex.sync()
    with_ft = con.execute("SELECT count(*) FROM hay").fetchone()[0]
    assert 0 < with_ft < 400 and eventindex.fulltext_cutoff() > 0
    lean = {q: search.run_store(q, now=now)["count"] for q in QUERIES}
    assert lean == full
    evs = _all(store)
    for q in QUERIES:
        assert lean[q] == search.run(q, evs, now=now)["count"], q
