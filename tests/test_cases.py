"""Incident case management: status / assignee / notes, permissions, audit, API,
reports (gap audit 2026-10-22)."""

from __future__ import annotations

import pytest

from src.soc import audit, cases, reports, users
from tests.test_search_reports import EVENTS, NOW

PW = "correct-horse-1"


@pytest.fixture()
def team() -> None:
    users.create_first_admin("boss", PW)
    users.add_user("boss", "ana", PW, "analyst")
    users.add_user("boss", "vic", PW, "viewer")


def test_case_lifecycle_and_audit(team: None) -> None:
    assert cases.get("inc-1")["status"] == "new"
    c = cases.assign("ana", "inc-1", "ana")
    assert c["assignee"] == "ana" and c["status"] == "investigating"  # assigning starts work
    cases.add_note("ana", "inc-1", "blocked 203.0.113.9 at the edge")
    c = cases.set_status("ana", "inc-1", "resolved")
    assert c["status"] == "resolved" and c.get("closed") and len(c["notes"]) == 1
    acts = [r["action"] for r in audit.read()]
    assert {"case_assigned", "case_note", "case_status"} <= set(acts)
    assert cases.summary()["resolved"] == 1


def test_viewers_cannot_change_cases_and_input_is_checked(team: None) -> None:
    with pytest.raises(PermissionError):
        cases.set_status("vic", "inc-2", "resolved")
    with pytest.raises(ValueError):
        cases.set_status("ana", "inc-2", "done")
    with pytest.raises(ValueError):
        cases.assign("ana", "inc-2", "nobody")
    with pytest.raises(ValueError):
        cases.add_note("ana", "inc-2", "x" * (cases.MAX_NOTE + 1))


def test_reports_show_the_case(team: None) -> None:
    rep = reports.build("week", events=EVENTS, now=NOW, template={"sections": ["incidents"]})
    assert "| Case |" in rep["markdown"]
