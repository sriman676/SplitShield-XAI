"""Correlation by user / host and behaviour-over-time rules (owner 2026-10-15)."""

from __future__ import annotations

from typing import Any

from src.soc.correlate import correlate, entity_of
from src.soc.parsers import _event


def test_entity_falls_back_to_user_then_host() -> None:
    assert entity_of(_event(src_ip="10.0.0.5")) == "10.0.0.5"
    assert entity_of(_event(user="alice")) == "user:alice"
    assert entity_of(_event(host="WS1")) == "host:WS1"
    assert entity_of(_event()) == "unknown:syslog:"
    assert entity_of(_event(source="", file="")) == "unknown"


def test_entityless_events_group_per_source_and_file() -> None:
    """Owner 2026-10-23: events without IP / user / host from different log files are
    no longer one entity (they merged into one fake incident)."""
    from src.soc.correlate import is_unknown, unknown_label

    a = [_event(ts=1000.0 + i, file="a.log", kind="audit_log_cleared", techniques=["T1070"]) for i in range(3)]
    b = [_event(ts=1000.0 + i, file="b.log", kind="audit_log_cleared", techniques=["T1070"]) for i in range(3)]
    incs = correlate(a + b)
    assert sorted(i["src_ip"] for i in incs) == ["unknown:syslog:a.log", "unknown:syslog:b.log"]
    assert all(is_unknown(i["src_ip"]) for i in incs) and not is_unknown("10.0.0.5")
    assert unknown_label("unknown:syslog:a.log") == "no IP / user / host (syslog, a.log)"


def test_nonstop_activity_splits_after_a_day() -> None:
    """A source that never pauses for the window becomes one incident per day, not one
    ever-growing group (memory, 10M benchmark 2026-10-23)."""
    from src.soc.correlate import MAX_GROUP_S

    evs = [_event(ts=float(t), src_ip="198.51.100.9", kind="failed_logon", techniques=["T1110"])
           for t in range(0, int(2.5 * MAX_GROUP_S), 600)]  # every 10 min for 2.5 days
    starts = sorted(i["start"] for i in correlate(evs))
    assert len(starts) == 3 and all(b - a > MAX_GROUP_S - 600 for a, b in zip(starts, starts[1:], strict=False))


def test_cloud_recon_burst_becomes_an_incident() -> None:
    names = [f"Describe{i}" for i in range(10)] + [f"List{i}" for i in range(8)]
    evs = [
        _event(ts=1000.0 + i, source="cloudtrail", kind=n, user="arn:aws:iam::1:user/x",
               detail=f"ec2:{n} by x")
        for i, n in enumerate(names)
    ]  # fmt: skip
    incs = correlate(evs)
    assert len(incs) == 1 and incs[0]["src_ip"] == "user:arn:aws:iam::1:user/x"
    assert "T1580" in {t["id"] for t in incs[0]["techniques"]}
    assert any("reconnaissance" in f for f in incs[0]["findings"])


def test_password_spraying_and_web_scan_findings() -> None:
    spray = [
        _event(ts=2000.0 + i, src_ip="203.0.113.7", kind="failed_logon", user=f"user{i}",
               techniques=["T1110"])
        for i in range(6)
    ]  # fmt: skip
    scan = [
        _event(ts=3000.0 + i, src_ip="198.51.100.9", source="web", kind="http_request",
               detail=f"GET /admin{i} -> 404")
        for i in range(60)
    ]  # fmt: skip
    by_ip = {i["src_ip"]: i for i in correlate(spray + scan)}
    assert any("spraying" in f for f in by_ip["203.0.113.7"]["findings"])
    assert "T1595" in {t["id"] for t in by_ip["198.51.100.9"]["techniques"]}


def test_a_few_routine_calls_do_not_trigger() -> None:
    evs = [
        _event(ts=1.0 + i, source="cloudtrail", kind=f"Describe{i}", user="u")
        for i in range(5)
    ]
    assert correlate(evs) == []


def test_every_attack_tactic_can_be_in_an_incident() -> None:
    """2026-10-23: a resource-development technique crashed incident building (the tactic
    was missing from the order); all 14 tactics, and unknown ones, must work."""
    from src.knowledge.explain import kb
    from src.soc.correlate import TACTIC_ORDER, tactic_rank

    seen = {t for v in kb().values() for t in v.get("tactics", [])}
    assert seen <= set(TACTIC_ORDER), seen - set(TACTIC_ORDER)
    assert tactic_rank("not-a-tactic") == len(TACTIC_ORDER)
    one: dict[str, str] = {}
    for tid, v in kb().items():  # one technique per tactic
        for t in v.get("tactics", []):
            one.setdefault(t, tid)
    evs = [_event(ts=1000.0 + i, host="WS9", kind="process_creation", techniques=[tid])
           for i, tid in enumerate(one.values())]  # fmt: skip
    incs = correlate(evs)  # raised ValueError before the fix
    assert incs and all(tactic_rank(t) < len(TACTIC_ORDER) for t in incs[0]["tactics"])
    assert len(incs[0]["tactics"]) >= 12  # every tactic family shows up (old / new names may merge)


def test_a_new_attack_tactic_is_ordered_weighted_and_reported() -> None:
    """Owner 2026-10-23: the tactic list follows the ATT&CK knowledge base (monthly update);
    a tactic with no weight gets the neutral 5 and is reported for review."""
    from src.soc import correlate, updater

    known = ["reconnaissance", "defense-evasion", "credential-access"]
    kb_order = ["reconnaissance", "quantum-ops", "credential-access"]
    merged = correlate._merged_order(known, kb_order)
    assert merged.index("quantum-ops") < merged.index("credential-access")
    assert "defense-evasion" in merged  # older names stay for older rule tags
    assert updater.new_tactics({"tactics": [{"shortname": "quantum-ops"}, {"shortname": "impact"}]}) == ["quantum-ops"]
    assert correlate.NEW_TACTIC_WEIGHT == 5
    objs: list[dict[str, Any]] = [{"type": "x-mitre-tactic", "id": "x-1", "x_mitre_shortname": "quantum-ops", "name": "Quantum Ops",
             "external_references": [{"source_name": "mitre-attack", "external_id": "TA0099"}]},
            {"type": "x-mitre-matrix", "id": "m-1", "tactic_refs": ["x-1"]}]  # fmt: skip
    assert updater.attack_tactics(objs) == [{"shortname": "quantum-ops", "name": "Quantum Ops", "id": "TA0099"}]
