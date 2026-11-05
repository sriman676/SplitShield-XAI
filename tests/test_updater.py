"""Internet update agent (src/soc/updater.py), offline: every download is mocked."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import pytest

from src.soc import intel, updater

FEODO_OLD = '"first_seen_utc","dst_ip","dst_port","c2_status","last_online","malware"\n"2022-06-04","162.243.103.246","8080","offline","2026-03-07","Emotet"\n'
FEODO_NEW = (
    FEODO_OLD + '"2026-10-01","50.16.16.211","443","online","2026-10-12","QakBot"\n'
)
FEODO_BAD = FEODO_OLD + '"2026-10-01","8.8.8.8","53","online","2026-10-12","Oops"\n'


def _lay(tmp_path: Path) -> updater.Layout:
    root = tmp_path / "proj"
    (root / "artifacts" / "intel").mkdir(parents=True)
    sample = root / "artifacts" / "intel" / "benign_sample.jsonl"
    rows = [
        {"source": "t", "line": "Jun  9 06:06:20 combo syslog: klogd startup succeeded"},
        {"source": "t", "line": json.dumps({
            "Id": 7036, "ProviderName": "Service Control Manager", "LogName": "System",
            "Message": "The Windows Update service entered the running state."})},
    ]  # fmt: skip
    sample.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    return updater.Layout(root=root, out=root)


def _fetcher(table: dict[str, Any]) -> Any:
    def fetch(url: str) -> bytes:
        v = table.get(url)
        if v is None:
            raise urllib.error.URLError("offline")
        if isinstance(v, Exception):
            raise v
        return v if isinstance(v, bytes) else str(v).encode()

    return fetch


# --- settings ----------------------------------------------------------------------


def test_on_by_default_and_toggle(tmp_path: Path) -> None:
    p = tmp_path / "update_agent.json"
    assert updater.get_enabled(p) is True  # owner 2026-10-25: on by default
    updater.set_enabled(True, p)
    assert updater.get_enabled(p) is True
    updater.set_enabled(False, p)
    st = updater.load_state(p)
    assert st["enabled"] is False and [h["action"] for h in st["history"]] == [
        "enabled",
        "disabled",
    ]


def test_disabled_run_does_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _lay(tmp_path)
    monkeypatch.setattr(updater, "FETCH", _fetcher({}))
    updater.set_enabled(False, lay.state)  # switched off by a user (on by default since 2026-10-25)
    rep = updater.run_once(lay=lay)
    assert rep["status"] == "disabled"


# --- intel feeds -------------------------------------------------------------------


def test_feed_update_applies_and_keeps_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _lay(tmp_path)
    intel.write_feed(
        intel.build_feed("feodo", FEODO_OLD.encode(), "old"), lay.intel_dir()
    )
    url = intel.FEEDS["feodo"]["url"]
    monkeypatch.setattr(updater, "FETCH", _fetcher({url: FEODO_NEW}))
    updater.set_enabled(True, lay.state)
    rep = updater.run_once(lay=lay, sources=("feodo",), run_tests_gate=False)
    s = rep["sources"]["feodo"]
    assert s["status"] == "applied", s
    assert s["checks"]["false_alarm"]["ok"]
    assert intel.lookup_ip("50.16.16.211", lay.intel_dir())
    assert (lay.previous / "feodo" / "feodo.json").is_file()
    # same day again: nothing happens
    assert (
        updater.run_once(lay=lay, sources=("feodo",))["status"] == "already ran today"
    )
    # rollback restores the old feed
    assert updater.rollback("feodo", lay)["ok"]
    assert intel.lookup_ip("50.16.16.211", lay.intel_dir()) == []


def test_feed_flagging_benign_indicator_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _lay(tmp_path)
    url = intel.FEEDS["feodo"]["url"]
    monkeypatch.setattr(updater, "FETCH", _fetcher({url: FEODO_BAD}))
    rep = updater.run_once(
        force=True, lay=lay, sources=("feodo",), run_tests_gate=False
    )
    s = rep["sources"]["feodo"]
    assert s["status"] == "rejected" and "benign" in s["reason"]
    assert not (lay.intel_dir() / "feodo.json").exists()


def test_empty_feed_rejected_and_network_errors_never_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _lay(tmp_path)
    tf = intel.FEEDS["threatfox"]["url"]
    oph = intel.FEEDS["openphish"]["url"]
    monkeypatch.setattr(
        updater,
        "FETCH",
        _fetcher(
            {
                tf: urllib.error.HTTPError(tf, 401, "key", None, None),  # type: ignore[arg-type]
                oph: b"",
            }
        ),  # fmt: skip
    )
    rep = updater.run_once(force=True, lay=lay, run_tests_gate=False)
    src = rep["sources"]
    assert src["threatfox"]["status"] == "skipped"
    assert src["openphish"]["status"] == "rejected"
    assert (
        src["sigma"]["status"] == "error"
        and "download failed" in src["sigma"]["reason"]
    )
    assert src["attack"]["status"] == "error"
    assert rep["status"] == "ok"
    assert updater.load_state(lay.state)["last_run"]


def test_tests_gate_failure_blocks_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _lay(tmp_path)
    url = intel.FEEDS["feodo"]["url"]
    monkeypatch.setattr(updater, "FETCH", _fetcher({url: FEODO_NEW}))
    monkeypatch.setattr(
        updater, "run_tests", lambda lay, c: {"ok": False, "summary": "1 failed"}
    )
    rep = updater.run_once(force=True, lay=lay, sources=("feodo",))
    s = rep["sources"]["feodo"]
    assert s["status"] == "rejected" and "tests fail" in s["reason"]
    assert not (lay.intel_dir() / "feodo.json").exists()


def test_dry_run_writes_only_to_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _lay(tmp_path)
    dry = updater.Layout(root=lay.root, out=tmp_path / "dry")
    url = intel.FEEDS["feodo"]["url"]
    monkeypatch.setattr(updater, "FETCH", _fetcher({url: FEODO_NEW}))
    rep = updater.run_once(
        force=True, dry_run=True, lay=dry, sources=("feodo",), run_tests_gate=False
    )
    assert rep["sources"]["feodo"]["status"] == "dry_run"
    assert (tmp_path / "dry" / "artifacts" / "intel" / "feodo.json").is_file()
    assert not (lay.intel_dir() / "feodo.json").exists()
    assert not lay.state.exists()


# --- ATT&CK ------------------------------------------------------------------------


def _bundle() -> dict[str, Any]:
    ref = [{"source_name": "mitre-attack", "external_id": "T1059.001"}]
    return {
        "objects": [
            {"type": "x-mitre-collection", "id": "x-mitre-collection--1", "x_mitre_version": "20.0"},
            {"type": "x-mitre-tactic", "id": "x-mitre-tactic--1", "name": "Execution", "x_mitre_shortname": "execution",
             "external_references": [{"source_name": "mitre-attack", "external_id": "TA0002"}]},
            {"type": "x-mitre-matrix", "id": "x-mitre-matrix--1", "tactic_refs": ["x-mitre-tactic--1"]},
            {"type": "attack-pattern", "id": "attack-pattern--1", "name": "PowerShell",
             "external_references": ref,
             "description": "Adversaries may abuse PowerShell (Citation: X). See [docs](https://e.x/y) <code>iex</code>.",
             "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "execution"}],
             "x_mitre_platforms": ["Windows"]},
            {"type": "attack-pattern", "id": "attack-pattern--2", "name": "Old", "revoked": True,
             "external_references": [{"source_name": "mitre-attack", "external_id": "T9999"}]},
            {"type": "course-of-action", "id": "course-of-action--1", "name": "Code Signing",
             "description": "Sign scripts."},
            {"type": "relationship", "id": "relationship--1", "relationship_type": "mitigates",
             "source_ref": "course-of-action--1", "target_ref": "attack-pattern--1",
             "description": "Allow only signed scripts."},
            {"type": "x-mitre-analytic", "id": "x-mitre-analytic--1", "x_mitre_platforms": ["Windows"],
             "description": "Encoded PowerShell launched by Office.",
             "x_mitre_log_source_references": [{"name": "WinEventLog:Sysmon"}]},
            {"type": "x-mitre-detection-strategy", "id": "x-mitre-detection-strategy--1",
             "x_mitre_analytic_refs": ["x-mitre-analytic--1"]},
            {"type": "relationship", "id": "relationship--2", "relationship_type": "detects",
             "source_ref": "x-mitre-detection-strategy--1", "target_ref": "attack-pattern--1"},
        ]
    }  # fmt: skip


def test_build_attack_kb_format() -> None:
    kb = updater.build_attack_kb(_bundle())
    assert kb["version"] == "20.0" and kb["source"] == updater.ATTACK_SOURCE
    assert list(kb["techniques"]) == ["T1059.001"]
    assert kb["tactics"] == [{"shortname": "execution", "name": "Execution", "id": "TA0002"}]
    t = kb["techniques"]["T1059.001"]
    assert set(t) == set(updater.KB_KEYS)
    assert t["description"] == "Adversaries may abuse PowerShell . See docs `iex`."
    assert t["tactics"] == ["execution"]
    assert t["mitigations"] == ["Code Signing: Allow only signed scripts."]
    assert t["detection_notes"] == [
        "[Windows] Encoded PowerShell launched by Office. (logs: WinEventLog:Sysmon)"
    ]
    assert t["url"] == "https://attack.mitre.org/techniques/T1059/001/"


def test_attack_update_and_shrink_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _lay(tmp_path)
    api = {"tag_name": "ATT&CK-v20.0",
           "assets": [{"name": "enterprise-attack.json", "browser_download_url": "https://x/e.json"}]}  # fmt: skip
    monkeypatch.setattr(
        updater,
        "FETCH",
        _fetcher(
            {
                updater.ATTACK_API: json.dumps(api),
                "https://x/e.json": json.dumps(_bundle()),
            }
        ),
    )
    monkeypatch.setattr(updater, "ATTACK_MIN_TECHNIQUES", 1)
    rep = updater.run_once(
        force=True, lay=lay, sources=("attack",), run_tests_gate=False
    )
    assert rep["sources"]["attack"]["status"] == "applied"
    kb = json.loads(lay.attack_json().read_text("utf-8"))
    assert kb["version"] == "20.0" and "T1059.001" in kb["techniques"]
    # a release with far fewer techniques than now is rejected
    big = {f"T{1000 + i}": kb["techniques"]["T1059.001"] for i in range(50)}
    updater.write_attack_kb(
        kb | {"version": "19", "techniques": big}, lay.attack_json()
    )
    st = updater.load_state(lay.state)
    st["versions"] = {}
    updater.save_state(st, lay.state)
    rep = updater.run_once(
        force=True, lay=lay, sources=("attack",), run_tests_gate=False
    )
    assert rep["sources"]["attack"]["status"] == "rejected"
    assert "techniques fell" in rep["sources"]["attack"]["reason"]


# --- Sigma -------------------------------------------------------------------------

RULE_OK = """title: Test Benign Never
id: 1
status: test
author: Someone
level: high
tags: [attack.execution, attack.t1059]
logsource: {product: windows, service: system}
detection:
  keywords: ['nothing-like-this-ever-xyz']
  condition: keywords
"""
RULE_NOISY = RULE_OK.replace("Test Benign Never", "Noisy").replace(
    "nothing-like-this-ever-xyz", "entered the running state"
)
RULE_FAIR_TEST = RULE_NOISY.replace("author: Someone", "author: Roberto Rodriguez")


def _zip(rules: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("version.txt", "Release: test\n")
        for name, text in rules.items():
            z.writestr(f"rules/windows/{name}", text)
    return buf.getvalue()


def _sigma_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_rules: dict[str, str]
) -> updater.Layout:
    lay = _lay(tmp_path)
    sd = lay.sigma_dir()
    (sd / "rules" / "rules" / "windows").mkdir(parents=True)
    (sd / "rules" / "rules" / "windows" / "a.yml").write_text(RULE_OK, "utf-8")
    (sd / "sigma_core+_r2026-07-01.zip").write_bytes(_zip({"a.yml": RULE_OK}))
    api = {"tag_name": "r2026-10-01",
           "assets": [{"name": "sigma_all_rules.zip", "browser_download_url": "https://x/s.zip"}]}  # fmt: skip
    monkeypatch.setattr(
        updater,
        "FETCH",
        _fetcher(
            {updater.SIGMA_API: json.dumps(api), "https://x/s.zip": _zip(new_rules)}
        ),
    )
    monkeypatch.setattr(updater, "SIGMA_MIN_FILES", 1)
    return lay


def test_sigma_current_tag_from_zip_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _sigma_setup(tmp_path, monkeypatch, {"a.yml": RULE_OK})
    assert updater.current_sigma_tag(lay, {}) == "r2026-07-01"
    updater.set_enabled(False, lay.state)
    rep = updater.run_once(
        force=False, lay=lay, sources=("sigma",), run_tests_gate=False
    )
    assert rep["status"] == "disabled"


def test_sigma_update_applies_with_fair_test_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _sigma_setup(
        tmp_path,
        monkeypatch,
        {"a.yml": RULE_OK, "b.yml": RULE_OK, "c.yml": RULE_FAIR_TEST},
    )
    rep = updater.run_once(
        force=True, lay=lay, sources=("sigma",), run_tests_gate=False
    )
    s = rep["sources"]["sigma"]
    assert s["status"] == "applied", s
    assert s["old_version"] == "r2026-07-01" and s["new_version"] == "r2026-10-01"
    assert s["checks"]["validate"]["new"]["excluded_fair_test"] == 1
    assert s["checks"]["false_alarm"]["rise_pp"] == 0
    sd = lay.sigma_dir()
    assert (sd / "sigma_all_rules_r2026-10-01.zip").is_file()
    assert not (sd / "sigma_core+_r2026-07-01.zip").exists()
    assert (lay.previous / "sigma" / "sigma_core+_r2026-07-01.zip").is_file()
    assert updater.load_state(lay.state)["versions"]["sigma"] == "r2026-10-01"
    assert updater.rollback("sigma", lay)["restored"] == "r2026-07-01"
    assert (sd / "sigma_core+_r2026-07-01.zip").is_file()


def test_sigma_rule_raising_benign_alerts_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay = _sigma_setup(tmp_path, monkeypatch, {"a.yml": RULE_OK, "n.yml": RULE_NOISY})
    rep = updater.run_once(
        force=True, lay=lay, sources=("sigma",), run_tests_gate=False
    )
    s = rep["sources"]["sigma"]
    assert s["status"] == "rejected" and "benign alert rate" in s["reason"], s
    assert s["checks"]["false_alarm"]["after"] > s["checks"]["false_alarm"]["before"]
    assert (lay.sigma_dir() / "sigma_core+_r2026-07-01.zip").is_file()  # kept


def test_zip_slip_rejected(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.yml", "x")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z, pytest.raises(ValueError):
        updater._safe_extract(z, tmp_path / "x")


def test_content_monthly_feeds_daily() -> None:
    """Owner 2026-10-23: Sigma / ATT&CK are checked once a month, threat-intel feeds daily."""
    import calendar
    import time as _t

    now = calendar.timegm(_t.strptime("2026-11-30T04:00:00", "%Y-%m-%dT%H:%M:%S"))
    st = {"checked": {"sigma": "2026-11-10T04:00:00Z", "attack": "2026-10-20T04:00:00Z"}}
    assert updater.is_due(st, "attack", now) and not updater.is_due(st, "sigma", now)
    assert updater.is_due(st, "urlhaus", now)  # feeds every day
    assert updater.is_due({}, "sigma", now)  # never checked
