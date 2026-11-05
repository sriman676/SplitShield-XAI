"""Users adapt built-in rules and engine settings (owner 2026-10-24)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.soc import rulectl
from src.soc.correlate import correlate

T0 = 1_760_000_000.0


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OVERRIDES", "SETTINGS", "PROPOSALS", "HISTORY"):
        monkeypatch.setattr(rulectl, name, tmp_path / Path(getattr(rulectl, name)).name)
    monkeypatch.setattr("src.soc.audit.path", lambda: tmp_path / "audit.jsonl")
    rulectl._CACHE.update(key=None, doc={})
    rulectl._ES.update(key=None, val={})


def _ev(**kw):  # noqa: ANN003, ANN202
    ev = {"ts": T0, "src_ip": "198.51.100.3", "source": "syslog", "kind": "log", "detail": "x", "techniques": [],
          "host": "web01"} | kw
    for rid, techs in kw.pop("hits", []):
        rulectl.hit(ev, rid, techs, rid)
    return ev


def test_switch_off_removes_only_that_rules_techniques() -> None:
    ev = _ev(techniques=["T1110", "T1059"], hits=[("sigma:a", ["T1110"]), ("own:b", ["T1059"])])
    rulectl.set_override("sigma:a", {"enabled": False}, reason="noisy here")
    assert rulectl.apply([ev]) == 1
    assert ev["techniques"] == ["T1059"] and ev["rules_off"] == ["sigma:a"]
    # a technique two rules give stays while one of them is still on
    ev2 = _ev(techniques=["T1110"], hits=[("sigma:a", ["T1110"]), ("own:b", ["T1110"])])
    rulectl.apply([ev2])
    assert ev2["techniques"] == ["T1110"]
    assert rulectl.history()[-1]["action"] == "rule_override"


def test_exceptions_technique_and_severity_overrides() -> None:
    rulectl.set_override("sigma:a", {"exceptions": [{"field": "host", "op": "equals", "value": "backup01"}],
                                     "techniques": ["T1078"], "severity": 9})  # fmt: skip
    backup = _ev(host="backup01", techniques=["T1110"], hits=[("sigma:a", ["T1110"])])
    other = _ev(techniques=["T1110"], hits=[("sigma:a", ["T1110"])])
    rulectl.apply([backup, other])
    assert backup["techniques"] == [] and other["techniques"] == ["T1078"] and other["severity_hint"] == 9
    inc = correlate([other])[0]
    assert inc["severity"] == 9
    with pytest.raises(ValueError):
        rulectl.set_override("sigma:a", {"severity": 11})
    with pytest.raises(ValueError):
        rulectl.set_override("sigma:a", {"techniques": ["not-a-technique"]})


def test_engine_settings_within_limits_and_validated_shown() -> None:
    base = rulectl.validated_engine()["correlation_gap_s"]
    s = rulectl.set_engine("correlation_gap_s", 1800, reason="slow attacks here")
    assert s["correlation_gap_s"] == 1800 and base == 900
    with pytest.raises(ValueError):
        rulectl.set_engine("crs_paranoia", 9)
    assert rulectl.set_engine("correlation_gap_s", None)["correlation_gap_s"] == base


def test_pack_round_trip_is_checked_before_switching_on() -> None:
    rulectl.set_override("crs:942100", {"enabled": False})
    rulectl.set_engine("crs_threshold", 7)
    pack = rulectl.export_pack()
    assert pack["overrides"]["crs:942100"]["enabled"] is False
    rep = rulectl.import_pack(pack, events=[], apply_now=False)
    assert rep["ok"] and not rep.get("applied")
    bad = pack | {"engine_settings": {"crs_paranoia": 99}}
    assert not rulectl.import_pack(bad, events=[])["ok"]
    with pytest.raises(ValueError):
        rulectl.import_pack({"format": "other"}, events=[])


def test_proposals_from_reviews_need_approval() -> None:
    alerts = [{"id": f"a{i}", "host": "backup01" if i < 6 else "web01",
               "rule_hits": [{"id": "sigma:noisy", "techniques": ["T1110"]}]} for i in range(7)]  # fmt: skip
    reviews = [{"alert_id": f"a{i}", "correct": i == 6} for i in range(7)]
    props = rulectl.propose(alerts, reviews)
    assert len(props) == 1 and props[0]["change"]["exceptions"][0]["value"] == "backup01"
    assert rulectl.overrides() == {}  # nothing changes until a user accepts
    rulectl.decide(props[0]["id"], accept=True)
    assert rulectl.overrides()["sigma:noisy"]["exceptions"][0]["value"] == "backup01"


def test_parsers_attribute_rules() -> None:
    from src.soc.parsers import parse_line

    ev = parse_line('{"timestamp": "2024-01-01T00:00:00Z", "event_type": "alert", "src_ip": "10.0.0.1", '
                    '"dest_ip": "10.0.0.2", "dest_port": 80, "alert": {"signature": "ET SCAN Nmap Scripting Engine", '
                    '"signature_id": 2009358, "category": "Attempted Information Leak"}}')  # fmt: skip
    assert ev is not None
    if ev["techniques"]:
        assert ev["rule_hits"][0]["id"].startswith("ids:")


def test_reapply_to_stored_events_both_ways() -> None:
    from src.soc import engine, search

    ev = _ev(techniques=["T1110"], hits=[("sigma:a", ["T1110"])], detail="failed password for root")
    engine._append(engine.EVENTS, [ev, _ev(detail="plain")])
    rulectl.set_override("sigma:a", {"enabled": False})
    r = rulectl.reapply_store()
    assert r == {"events": 2, "changed": 1}
    stored = engine.read_jsonl(engine.EVENTS)
    assert stored[0]["techniques"] == [] and stored[0]["rules_off"] == ["sigma:a"]
    assert search.run_store("failed password")["count"] == 1  # the index was rebuilt
    rulectl.set_override("sigma:a", None)  # undo -> the rule's technique comes back
    assert rulectl.reapply_store()["changed"] == 1
    assert engine.read_jsonl(engine.EVENTS)[0]["techniques"] == ["T1110"]
