"""Shared risk board, cross-checks, place profiles, hash intel, new-model scaffold
(owner 2026-10-15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.soc import crosscheck, intel, places, riskboard
from src.soc.correlate import correlate

T0 = 1_760_000_000.0


def _ev(**kw):  # noqa: ANN003, ANN202
    base = {"ts": T0, "src_ip": None, "dst_ip": None, "dst_port": None,
            "source": "syslog", "kind": "log", "detail": "", "techniques": []}  # fmt: skip
    return base | kw


# --- entities and families -----------------------------------------------------------


def test_families_and_entities() -> None:
    assert riskboard.family_of(_ev(source="suricata")) == "ids"
    assert riskboard.family_of(_ev(source="winsec")) == "endpoint"
    assert riskboard.family_of(_ev(source="cloudtrail")) == "cloud"
    assert riskboard.family_of(_ev(source="phishing")) == "phishing"
    assert riskboard.family_of(_ev(source="unknown:kv")) == "unknown"
    keys = riskboard.entities_of(_ev(src_ip="1.2.3.4", user="Bob", host="PC1",
                                     detail="GET http://evil.example/x"))  # fmt: skip
    assert {"ip:1.2.3.4", "user:bob", "host:pc1", "url:http://evil.example/x",
            "domain:evil.example"} <= set(keys)  # fmt: skip


def test_dst_ip_is_not_an_entity() -> None:
    assert riskboard.entities_of(_ev(dst_ip="10.0.0.1")) == []


# --- coordination: a finding by another family raises severity ---------------------------


def _phish_then_logon() -> list[dict]:
    return [
        # the mail / phishing model flags a link the user clicked (user entity)
        _ev(source="phishing", user="alice", techniques=["T1566.002"], detail="phishing url"),
        # later: brute force + logon from an IP that is tied to alice by the logon event
        _ev(ts=T0 + 60, source="winsec", src_ip="203.0.113.9", kind="failed_logon",
            techniques=["T1110"]),
        _ev(ts=T0 + 90, source="winsec", src_ip="203.0.113.9", user="alice",
            kind="successful_logon"),
    ]  # fmt: skip


def test_linked_family_raises_severity() -> None:
    evs = _phish_then_logon()
    off = {i["src_ip"]: i for i in correlate(evs)}
    board = riskboard.Board(weight=1, cap=2).add_all(evs)
    on = {i["src_ip"]: i for i in correlate(evs, board=board)}
    ip = "203.0.113.9"
    assert on[ip]["severity"] == min(10, off[ip]["severity"] + 1)
    assert on[ip]["board_boost"] == 1
    assert any("phishing" in r for r in on[ip]["linked_findings"])
    assert on[ip]["corroborated"]


def test_board_never_lowers_and_respects_cap_and_window() -> None:
    evs = _phish_then_logon()
    evs[0]["ts"] = T0 - 10 * 86400  # phishing long before: outside the 24 h window
    board = riskboard.Board(weight=5, cap=1).add_all(evs)
    inc = next(i for i in correlate(evs, board=board) if i["src_ip"] == "203.0.113.9")
    assert inc["board_boost"] == 0
    evs = _phish_then_logon()
    board = riskboard.Board(weight=5, cap=1).add_all(evs)
    inc = next(i for i in correlate(evs, board=board) if i["src_ip"] == "203.0.113.9")
    assert inc["board_boost"] == 1  # capped


def test_hub_entities_are_not_followed() -> None:
    evs = [_ev(source="phishing", user="u0", host="proxy", techniques=["T1566.002"])]
    evs += [
        _ev(ts=T0 + i, source="winsec", user=f"u{i}", host="proxy")
        for i in range(1, 80)
    ]
    evs.append(_ev(ts=T0 + 100, source="suricata", src_ip="198.51.100.7", host="proxy",
                   techniques=["T1046"]))  # fmt: skip
    board = riskboard.Board(weight=1, cap=2, max_degree=50).add_all(evs)
    inc = next(i for i in correlate(evs, board=board) if i["src_ip"] == "198.51.100.7")
    assert inc["board_boost"] == 0  # "proxy" links to 80 entities: a hub


def test_place_scale_cannot_exceed_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        riskboard, "settings", lambda: riskboard.DEFAULT | {"weight": 1, "cap": 2}
    )
    assert riskboard.board_for([], scale=3.0).weight == 1.0
    assert riskboard.board_for([], scale=0.5).weight == 0.5


# --- cross-checks --------------------------------------------------------------------


def test_crosscheck_hash_intel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = "a" * 64
    doc = intel.build_feed("malwarebazaar", f"# header\n{h}\n".encode(), "2026-10-15")
    intel.write_feed(doc, tmp_path)
    monkeypatch.setattr(intel, "INTEL_DIR", tmp_path)
    monkeypatch.setattr(intel, "status", lambda folder=None: {"malwarebazaar": {}})
    monkeypatch.setattr(
        intel, "index", lambda folder=None: intel.Index(intel.read_feeds(tmp_path))
    )
    ev = _ev(source="sysmon", host="pc1", user="bob", sha256=h, file=r"C:\x\evil.exe")
    out = crosscheck._intel([ev])
    assert len(out) == 1 and out[0]["source"] == "intel"
    assert out[0]["techniques"] == ["T1204.002"]
    assert out[0]["user"] == "bob" and out[0]["host"] == "pc1"  # carried for the board


def test_crosscheck_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_evs):  # noqa: ANN001, ANN202
        raise RuntimeError("model missing")

    monkeypatch.setattr(crosscheck, "_urls", boom)
    monkeypatch.setattr(crosscheck, "_intel", boom)
    monkeypatch.setattr(crosscheck, "_files", boom)
    assert crosscheck.run([_ev(detail="http://x.example/")]) == []


def test_derived_events_are_not_rechecked() -> None:
    assert crosscheck.run([_ev(source="phishing", detail="http://x.example/")]) == []


def test_malwarebazaar_parser() -> None:
    items = intel.parse_malwarebazaar("# c\n" + "B" * 64 + "\nnot-a-hash\n")
    assert [i["value"] for i in items] == ["b" * 64]
    assert (
        intel.parse_threatfox(
            '# "first_seen_utc","ioc_id","ioc_value","ioc_type","threat_type","fk_malware",'
            '"malware_alias","malware_printable","last_seen_utc","confidence_level",'
            '"reference","tags","anonymous","reporter"\n'
            f'"2026-10-15","1","{"c" * 64}","sha256_hash","payload","x","","Lumma","","100","","","0","a"\n'
        )[0]["type"]
        == "sha256"
    )


# --- place profiles -------------------------------------------------------------------


@pytest.fixture
def place_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(places, "STATE", tmp_path / "place.json")
    monkeypatch.setattr(places, "BUDGET", tmp_path / "budget.json")
    return tmp_path


def test_place_guess() -> None:
    assert places.guess([])[0] == "home_pc"
    web = [_ev(source="apache", ts=1e12) for _ in range(10)]
    assert places.guess(web)[0] == "server"
    office = [_ev(source="winsec", host=f"pc{i}", ts=1e12) for i in range(6)]
    assert places.guess(office)[0] == "small_office"
    lab = [
        _ev(source="suricata", techniques=["T1046"]) for _ in range(10)
    ]  # 2025 timestamps
    assert places.guess(lab)[0] == "lab"


def test_place_modes(place_files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    sig = inspect.signature(places.set_place)
    assert sig.parameters["mode"].default == "manual"
    places.set_place("home_pc", mode="confirm")
    office = [_ev(source="winsec", host=f"pc{i}", ts=1e12) for i in range(6)]
    st = places.check(office, force=True)
    assert st["place"] == "home_pc" and "notice" in st  # confirm mode: asks, no switch
    places.set_mode("auto")
    st = places.check(office, force=True)
    assert st["place"] == "small_office"  # auto mode: switches
    places.set_place("lab")  # manual pick always wins
    assert places.check(office, force=True)["place"] == "lab"
    with pytest.raises(ValueError):
        places.set_place("moon_base")


def test_push_budget_lets_critical_through(
    place_files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(places, "current", lambda: {"name": "t", "notify_min_severity": 6,
                                                    "max_pushes_per_day": 1})  # fmt: skip
    assert places.may_push(5) is False  # below the profile minimum
    assert places.may_push(6) is True
    assert places.may_push(7) is False  # budget used up
    assert places.may_push(9) is True  # critical: always


def test_profiles_never_switch_detectors_off() -> None:
    for name, p in places.profiles().items():
        assert set(p) <= {"description", "board_scale", "notify_min_severity",
                          "max_pushes_per_day"}, name  # fmt: skip
        assert 0 <= p["board_scale"] <= 1


# --- new-model scaffold --------------------------------------------------------------


def test_new_model_scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from scripts import new_model
    from src.soc import controller

    root = tmp_path
    for d in ("config", "src/mlareas", "tests"):
        (root / d).mkdir(parents=True)
    reg = root / "config" / "models.yaml"
    reg.write_text(Path(controller.REGISTRY).read_text("utf-8"), "utf-8")
    monkeypatch.setattr(new_model, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["new_model", "dns_tunnel", "--title", "DNS tunnelling",
                                      "--family", "network", "--entities", "ip,domain"])  # fmt: skip
    new_model.main()
    models = controller.registry(reg)
    assert models["dns_tunnel"].extra["family"] == "network"
    assert models["dns_tunnel"].extra["entities"] == ["ip", "domain"]
    assert (root / "src/mlareas/dns_tunnel.py").is_file()
    assert "test_dns_tunnel_is_registered" in (
        root / "tests/test_mlarea_dns_tunnel.py"
    ).read_text("utf-8")
    with pytest.raises(SystemExit):
        new_model.main()  # refuses to add it twice
    assert len(json.dumps(sorted(models))) > 0


# --- victim follow (owner 2026-10-24) -------------------------------------------------------


def _web_attack_then_host_activity(gap: float = 600.0) -> list[dict]:
    return [
        # the web server logs an exploit attempt from the attacker
        _ev(source="apache", src_ip="198.51.100.7", observer="web01", kind="http_request",
            techniques=["T1190"], detail="GET /cgi-bin/x?cmd=id"),
        # minutes later the same server's audit log shows a shell (no attacker IP in it)
        _ev(ts=T0 + gap, source="auditd", observer="web01", kind="process",
            techniques=["T1059.004"], detail="execve /bin/sh -c id"),
    ]  # fmt: skip


def test_victim_follow_links_attack_to_its_victim() -> None:
    evs = _web_attack_then_host_activity()
    off = riskboard.Board(weight=1, cap=2).add_all(evs)
    inc_off = next(i for i in correlate(evs, board=off) if i["src_ip"] == "198.51.100.7")
    assert inc_off["board_boost"] == 0  # the shell has no attacker IP: not linked
    on = riskboard.Board(weight=1, cap=2, follow_s=3600).add_all(evs)
    inc_on = next(i for i in correlate(evs, board=on) if i["src_ip"] == "198.51.100.7")
    assert inc_on["board_boost"] == 1
    assert any("victim web01" in r for r in inc_on["linked_findings"])


def test_victim_follow_is_time_bounded_and_forward_only() -> None:
    late = _web_attack_then_host_activity(gap=3 * 3600)
    on = riskboard.Board(weight=1, cap=2, follow_s=3600).add_all(late)
    assert next(i for i in correlate(late, board=on) if i["src_ip"] == "198.51.100.7")["board_boost"] == 0
    before = _web_attack_then_host_activity(gap=-600)  # host activity BEFORE the attack
    on = riskboard.Board(weight=1, cap=2, follow_s=3600).add_all(before)
    assert next(i for i in correlate(before, board=on) if i["src_ip"] == "198.51.100.7")["board_boost"] == 0
