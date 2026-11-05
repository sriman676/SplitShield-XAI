"""The owner's custom rules, allow-list and stricter-only sensitivity (2026-10-15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.soc import allowlist, custom

T0 = 1_760_000_000.0


def _ev(**kw):  # noqa: ANN003, ANN202
    base = {"ts": T0, "src_ip": None, "source": "winsec", "kind": "log", "detail": "",
            "techniques": []}  # fmt: skip
    return base | kw


RULE = {"name": "encoded powershell", "field": "detail", "op": "contains",
        "value": "-enc", "techniques": ["T1059.001"], "severity": 7}  # fmt: skip


# --- custom rules ----------------------------------------------------------------------


def test_simple_rule_fires_and_marks(tmp_path: Path) -> None:
    evs = [_ev(detail="powershell -enc AAAA"), _ev(detail="notepad.exe")]
    custom.save_rule(RULE, evs * 30, accept_noisy=True, folder=tmp_path)
    batch = [_ev(detail="powershell.exe -enc ZQBj"), _ev(detail="explorer.exe")]
    assert custom.apply(batch, folder=tmp_path) == 1
    assert batch[0]["techniques"] == ["T1059.001"]
    assert batch[0]["custom_rules"] == ["custom:encoded powershell"]
    assert batch[0]["severity"] == 7
    assert batch[1]["techniques"] == []


def test_noisy_rule_is_refused_unless_accepted(tmp_path: Path) -> None:
    evs = [_ev(detail="powershell -enc x")] * 10  # fires on 100 %
    with pytest.raises(ValueError, match="I understand"):
        custom.save_rule(RULE, evs, folder=tmp_path)
    assert (
        custom.save_rule(RULE, evs, accept_noisy=True, folder=tmp_path)["fired"] == 10
    )


def test_threshold_rule_counts_per_entity_in_window() -> None:
    doc = {"name": "many failures", "field": "kind", "op": "equals", "value": "failed_logon",
           "techniques": ["T1110"], "count": 3, "minutes": 5}  # fmt: skip
    r = custom.compile_doc(doc)
    evs = [_ev(ts=T0 + i * 10, kind="failed_logon", src_ip="1.2.3.4") for i in range(3)]
    assert [r.fires(e) for e in evs] == [False, False, True]
    late = _ev(ts=T0 + 3600, kind="failed_logon", src_ip="5.6.7.8")
    assert r.fires(late) is False  # another entity, fresh count


@pytest.mark.parametrize(
    ("bad", "msg"),
    [
        ({"techniques": ["X1"]}, "technique"),
        ({"techniques": []}, "technique"),
        ({"name": "x"}, "name"),
        ({"op": "regex", "value": "("}, ""),
        ({"count": 3, "minutes": 0}, "window"),
    ],
)
def test_invalid_rules_rejected(bad: dict, msg: str) -> None:
    with pytest.raises(Exception, match=msg):
        custom.compile_doc(RULE | bad)


def test_sigma_rule(tmp_path: Path) -> None:
    sig = """title: Mimikatz in text
detection:
  keywords:
    - mimikatz
  condition: keywords
tags: [attack.t1003]
level: high
"""
    r = custom.compile_doc({"name": "sigma mimi", "kind": "sigma", "sigma": sig})
    assert r.techniques == ["T1003"]
    assert r.fires(_ev(detail="C:\\tools\\mimikatz.exe sekurlsa"))
    assert not r.fires(_ev(detail="notepad"))


def test_broken_rule_file_never_stops_ingestion(tmp_path: Path) -> None:
    (tmp_path / "bad.yml").write_text("name: [unclosed", "utf-8")
    assert custom.apply([_ev(detail="x")], folder=tmp_path) == 0
    assert custom.rules(tmp_path)[1]  # reported as an error


# --- allow-list ------------------------------------------------------------------------


def test_allowlist_mutes_but_critical_still_forwards(tmp_path: Path) -> None:
    p = tmp_path / "al.json"
    allowlist.add("ip", "10.0.0.5", "our vulnerability scanner", 30, path=p)
    alert = _ev(src_ip="10.0.0.5", severity=6, techniques=["T1046"])
    assert allowlist.is_muted(alert, p)[0]
    assert allowlist.forward(alert, p) == (False, allowlist.is_muted(alert, p)[1])
    ok, mark = allowlist.forward(alert | {"severity": 9}, p)
    assert ok and mark is not None and mark.startswith("allow-listed entity involved")
    inc = {"src_ip": "10.0.0.5", "start": T0, "end": T0 + 1, "severity": 6}
    assert allowlist.is_muted(inc, p)[0]


def test_allowlist_requires_reason_and_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "al.json"
    with pytest.raises(ValueError, match="reason"):
        allowlist.add("user", "bob", " ", path=p)
    with pytest.raises(ValueError, match="1-365"):
        allowlist.add("user", "bob", "admin", days=400, path=p)
    allowlist.add("user", "Bob", "IT admin", days=1, path=p)
    assert allowlist.is_muted(_ev(user="bob"), p)[0]
    monkeypatch.setattr(allowlist, "_now", lambda: T0 + 10 * 86400 + 1e9)
    assert not allowlist.is_muted(_ev(user="bob"), p)[0]  # expired
    assert allowlist.entries(p)[0]["active"] is False


def test_allowlist_domain_covers_subdomains(tmp_path: Path) -> None:
    p = tmp_path / "al.json"
    allowlist.add("domain", "https://example.com/", "our site", path=p)
    assert allowlist.is_muted(_ev(detail="GET https://cdn.example.com/x"), p)[0]
    assert not allowlist.is_muted(_ev(detail="GET https://example.com.evil.net/"), p)[0]


# --- sensitivity: stricter only ----------------------------------------------------------


def test_sensitivity_stricter_or_looser(tmp_path: Path) -> None:
    """Owner 2026-10-24: looser than validated is allowed (with a warning on the page),
    never below LOOSEST."""
    p = tmp_path / "s.json"
    with pytest.raises(ValueError, match="coin flip"):
        custom.set_threshold("phishing", 0.4, validated=0.9, path=p)
    assert custom.set_threshold("phishing", 0.97, validated=0.9, path=p) == 0.97
    assert custom.threshold("phishing", 0.9, p) == 0.97
    assert custom.set_threshold("phishing", 0.7, validated=0.9, path=p) == 0.7  # looser
    assert custom.threshold("phishing", 0.98, p) == 0.7  # the user's value is kept
    custom.set_threshold("phishing", None, validated=0.9, path=p)
    assert custom.threshold("phishing", 0.9, p) == 0.9


def test_looser_flow_threshold_promotes_possible_alerts(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    custom.set_threshold("flow", 0.6, validated=0.9, path=p)
    ev = _ev(source="detector", tier="low", confidence=0.7, label="PortScan", kind="flow_detection_low")
    custom.demote(ev, sens_path=p)
    assert ev["tier"] == "confirmed" and ev["techniques"]


def test_demote_keeps_record_but_drops_alert(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    custom.set_threshold("phishing", 0.97, validated=0.9, path=p)
    ev = _ev(
        source="phishing",
        detail="phishing model: http://x (p=0.95)",
        techniques=["T1566.002"],
    )
    assert custom.demote(ev, p) is True
    assert ev["techniques"] == [] and "below your stricter threshold" in ev["detail"]
    strong = _ev(
        source="phishing",
        detail="phishing model: http://y (p=0.99)",
        techniques=["T1566.002"],
    )
    assert custom.demote(strong, p) is False and strong["techniques"]
