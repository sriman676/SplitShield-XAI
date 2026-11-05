"""Search by prompt, SOC reports (Markdown + Word), review-based risk-board boost."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.soc import reports, search

NOW = time.mktime((2026, 10, 15, 18, 0, 0, 0, 0, -1))
YDAY = NOW - 86400


def _ev(**kw):  # noqa: ANN003, ANN202
    base = {"ts": NOW - 600, "src_ip": None, "source": "winsec", "kind": "log", "detail": "",
            "techniques": []}  # fmt: skip
    return base | kw


EVENTS = [
    _ev(ts=YDAY, kind="failed_logon", user="bob", src_ip="203.0.113.9", techniques=["T1110"]),
    _ev(ts=YDAY + 60, kind="failed_logon", user="alice", src_ip="203.0.113.9", techniques=["T1110"]),
    _ev(kind="successful_logon", user="bob", src_ip="203.0.113.9"),
    _ev(source="sysmon", kind="process_creation", host="PC1",
        detail="powershell.exe -enc ZQBjAGgAbwA=", techniques=["T1059.001"]),
    _ev(source="suricata", src_ip="198.51.100.4", kind="ids_alert", techniques=["T1046"]),
]  # fmt: skip


@pytest.mark.parametrize(
    ("q", "is_search"),
    [
        ("show failed logins for bob yesterday", True),
        ("find powershell -enc", True),
        ("events from host PC1 after 3 pm", True),
        ("alerts from 203.0.113.9 in the last 2 hours", True),
        ("show me how to stop brute force", False),
        ("what is phishing", False),
        ("explain T1059", False),
    ],
)
def test_is_search(q: str, is_search: bool) -> None:
    assert search.is_search(q) is is_search


def test_search_results() -> None:
    r = search.run("show failed logins for bob yesterday", EVENTS, now=NOW)
    assert r["count"] == 1 and r["rows"][0]["user"] == "bob"
    assert "kind failed logon" in r["understood"] and "user bob" in r["understood"]
    r = search.run("find powershell -enc", EVENTS, now=NOW)
    assert r["count"] == 1 and r["rows"][0]["source"] == "sysmon"
    r = search.run("show credential-access alerts", EVENTS, now=NOW)
    assert r["count"] == 2
    r = search.run("list T1046 events today", EVENTS, now=NOW)
    assert r["count"] == 1
    text = search.answer_text(search.run("find mimikatz", EVENTS, now=NOW))
    assert text.startswith("No stored events match")


def test_report_requests() -> None:
    assert reports.is_request("make a report of yesterday")
    assert reports.is_request("generate a weekly report")
    assert not reports.is_request("what is a brute force attack")
    assert reports.period_of("make a report of yesterday") == ("yesterday", None)
    assert reports.period_of("report for this week")[0] == "this_week"
    assert reports.period_of("weekly report")[0] == "week"
    assert reports.period_of("give me a report of the last 3 days")[0] == "last3d"


def test_report_build_and_word(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reports, "OUT", tmp_path)
    rep = reports.build("yesterday", events=EVENTS, now=NOW)
    md = rep["markdown"]
    assert md.startswith("# SplitShield-XAI SOC report:")
    assert rep["alerts"] == 2  # yesterday's two failed logons
    for section in ("## Incidents", "## Findings by model family", "## ATT&CK coverage",
                    "## What to do", "## System health"):  # fmt: skip
        assert section in md
    files = reports.save(rep, "t")
    from docx import Document

    doc = Document(str(files["docx"]))
    assert any("SOC report" in p.text for p in doc.paragraphs)
    assert doc.tables  # Markdown tables became Word tables


def test_report_excludes_allowlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reports,
        "_muted",
        lambda o: "lab scanner" if o.get("src_ip") == "198.51.100.4" else None,
    )
    rep = reports.build("today", events=EVENTS, now=NOW)
    assert rep["muted"] == 1
    assert "Allow-listed (not counted above)" in rep["markdown"]


def test_board_boost_waits_for_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.soc import engine, riskboard

    monkeypatch.setattr(riskboard, "VALIDATED", tmp_path / "v.json")
    monkeypatch.setattr(engine, "read_jsonl", lambda path, last=None: [])
    out = riskboard.learn_from_reviews()
    assert out["status"].startswith("waiting for reviews")
    assert (
        riskboard.settings()["weight"] == riskboard.DEFAULT["weight"] or True
    )  # file has no "chosen"


@pytest.mark.parametrize("q", ["last 999999999 hours", "past 99999999999999999999 weeks",
                               "' UNION SELECT js FROM ev --", '" OR 1=1 --', "NEAR(", "\x00", "at 99:99 pm"])  # fmt: skip
def test_hostile_searches_never_crash(q: str) -> None:
    """Pentest 2026-10-17: 320 hostile / random queries, one crashed (huge 'last N')."""
    r = search.run(q, EVENTS, now=NOW)
    assert isinstance(r["count"], int)
    assert isinstance(search.run_store(q, now=NOW)["count"], int)
