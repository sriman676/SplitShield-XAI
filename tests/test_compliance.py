"""Compliance pack: NIST CSF 2.0 / ISO 27001:2022 evidence from the SOC's own records."""

from __future__ import annotations

from src.soc import compliance, reports, users
from tests.test_search_reports import EVENTS, NOW


def test_every_control_gets_a_state_and_gaps_say_what_to_do() -> None:
    rows = compliance.assess(compliance.evidence(NOW - 7 * 86400, NOW, EVENTS, []))
    assert len(rows) == len(compliance.CONTROLS)
    assert {r["framework"] for r in rows} == {"NIST CSF 2.0", "ISO/IEC 27001:2022"}
    assert all(r["state"] in ("evidence", "partial", "gap") for r in rows)
    assert all(r["todo"] for r in rows if r["state"] == "gap")
    by = {r["id"]: r for r in rows}
    assert by["DE.CM-03"]["state"] == "evidence"  # the failed logons in EVENTS
    assert by["RS.MA-02"]["state"] == "gap"  # no reviews yet
    assert by["A.5.28"]["state"] == "evidence"  # audit chain intact


def test_access_control_evidence_follows_the_users() -> None:
    users.create_first_admin("boss", "correct-horse-1")
    ev = compliance.evidence(NOW - 86400, NOW + 86400, EVENTS, [])
    assert {r["id"]: r for r in compliance.assess(ev)}["A.5.15"]["state"] == "partial"
    users.add_user("boss", "ana", "correct-horse-1", "analyst")
    ev = compliance.evidence(NOW - 86400, NOW + 86400, EVENTS, [])
    assert {r["id"]: r for r in compliance.assess(ev)}["A.5.15"]["state"] == "evidence"


def test_weekly_compliance_report() -> None:
    rep = reports.build("week", events=EVENTS, now=NOW, template={"sections": ["summary", "compliance"]})
    assert "## Compliance evidence (NIST CSF 2.0, ISO/IEC 27001:2022)" in rep["markdown"]
    assert "not a certification" in rep["markdown"] and "| A.8.15 |" in rep["markdown"]
    assert any(s["name"] == "Weekly compliance" for s in reports.load_schedules())
