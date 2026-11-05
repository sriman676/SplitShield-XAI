"""SOC layer: parsers, correlation (multi-stage chains), CEF, sinks, RCL learning."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import pytest

from src.soc.correlate import correlate
from src.soc.parsers import parse_line, parse_text
from src.soc.rcl import RCL
from src.soc.samples import make_samples
from src.soc.sinks import DEFAULTS, send, to_cef


def test_parsers_detect_formats(tmp_path: Path) -> None:
    files = {
        p.name: parse_text(p.read_text("utf-8"), p.name)
        for p in make_samples(tmp_path)
        if p.suffix != ".csv"
    }
    assert {e["source"] for e in files["conn.log"]} == {"zeek"}
    assert {e["source"] for e in files["pfirewall.log"]} == {"firewall"}
    assert any(e["techniques"] == ["T1190"] for e in files["eve.json"])
    win = files["winsec.json"]
    assert sum(e["kind"] == "failed_logon" for e in win) == 8
    assert any(
        e["kind"] == "suspicious_process" and e["techniques"] == ["T1059"] for e in win
    )
    ev = parse_line(
        "Jan 01 00:00:00 h sshd[1]: Failed password for root from 192.0.2.9 port 1 ssh2"
    )
    assert ev is not None and ev["src_ip"] == "192.0.2.9"


def test_correlation_finds_compromise_chain_and_scan(tmp_path: Path) -> None:
    events = [
        e
        for p in make_samples(tmp_path)
        if p.suffix != ".csv"
        for e in parse_text(p.read_text("utf-8"), p.name)
    ]
    incs = {i["src_ip"]: i for i in correlate(events, window_s=3600)}
    rdp = incs["203.0.113.45"]
    ids = {t["id"] for t in rdp["techniques"]}
    assert {
        "T1046",
        "T1190",
        "T1110",
        "T1078",
    } <= ids  # scan -> exploit -> brute force -> valid account
    assert rdp["corroborated"] and rdp["level"] == "critical"
    assert "T1078" in {
        t["id"] for t in incs["192.0.2.10"]["techniques"]
    }  # SSH brute force succeeded
    assert "T1046" in {
        t["id"] for t in incs["198.51.100.7"]["techniques"]
    }  # firewall-drop sweep


def test_cef_escapes_and_sinks_deliver(tmp_path: Path) -> None:
    cef = to_cef(
        {"title": "a|b", "reason": "x=y", "severity": 7, "techniques": ["T1110"]}
    )
    assert "a\\|b" in cef and "msg=x\\=y" in cef and "cs1=T1110" in cef
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(3)
    s = json.loads(json.dumps(DEFAULTS))
    s["jsonl"]["path"] = str(tmp_path / "out.jsonl")
    s["syslog"].update(enabled=True, port=rx.getsockname()[1], protocol="udp")  # UDP chosen (default is TCP)
    res = send({"title": "t", "severity": 9, "techniques": []}, s)
    assert res == {"jsonl": "ok", "syslog": "ok"}
    assert b"CEF:0|SplitShield" in rx.recv(4096)
    rx.close()
    assert json.loads((tmp_path / "out.jsonl").read_text("utf-8"))["title"] == "t"


def test_rcl_rewards_move_thresholds_and_calibrate(tmp_path: Path) -> None:
    r = RCL(tmp_path / "s.json", tmp_path / "f.jsonl")
    fp = r.feedback(
        {"label": "DoS", "confidence": 0.8, "logits": [0.0, 3.0]}, correct=False
    )
    assert fp["threshold_after"] > fp["threshold_before"]  # false alarm -> stricter
    tp = r.feedback(
        {"label": "PortScan", "confidence": 0.9, "logits": [0.0, 3.0]}, correct=True
    )
    assert tp["threshold_after"] < tp["threshold_before"]  # confirmed -> more sensitive
    # overconfident model, mostly wrong -> temperature rises (confidence softened)
    for i in range(20):
        r.feedback(
            {"label": "Bot", "confidence": 0.99, "logits": [0.0, 6.0]},
            correct=i % 2 == 0,
        )
    assert r.temperature > 1.0
    assert (
        RCL(tmp_path / "s.json", tmp_path / "f.jsonl").thresholds["DoS"]
        == r.thresholds["DoS"]
    )
    assert np.isclose(r.calibrated(np.array([0.0, 0.0])).sum(), 1.0)


def test_firewall_never_blocks_local_addresses() -> None:
    from src.soc.model import firewall_block

    for ip in ("192.168.1.1", "10.0.0.1", "127.0.0.1", "169.254.0.1", "not-an-ip"):
        assert firewall_block(ip) is False


def test_store_cache_sees_appends_and_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.soc import engine

    monkeypatch.setattr(engine, "EVENTS", tmp_path / "events.jsonl")
    ev = parse_line("Jan 01 00:00:00 h sshd[1]: Failed password from 198.51.100.7")
    assert ev is not None
    engine._append(engine.EVENTS, [ev])
    assert len(engine.read_jsonl(engine.EVENTS)) == 1
    engine._append(engine.EVENTS, [ev])
    assert len(engine.read_jsonl(engine.EVENTS)) == 2  # cache invalidated by append
    assert engine.incidents()  # brute-force evidence -> an incident
    engine.EVENTS.unlink()
    assert engine.read_jsonl(engine.EVENTS) == [] and engine.incidents() == []


def test_timestamps_and_cef_are_sanitized() -> None:
    from src.soc.parsers import _ts

    assert _ts(1_700_000_000_000) == 1_700_000_000.0  # epoch ms -> seconds
    assert "\n" not in to_cef({"title": "a\nb", "reason": "c\r\nd", "severity": 5})


def test_valid_accounts_follows_the_brute_force() -> None:
    """A legitimate logon before the attack must not put T1078 before T1110."""
    from src.soc.correlate import correlate

    def ev(ts: float, kind: str, techs: list[str]) -> dict:
        return {"ts": ts, "src_ip": "192.0.2.10", "dst_ip": None, "dst_port": None,
                "source": "syslog", "kind": kind, "detail": "", "techniques": techs}  # fmt: skip

    evs = [ev(100, "successful_logon", [])]
    evs += [ev(200 + i, "failed_logon", ["T1110"]) for i in range(6)]
    evs += [ev(300, "successful_logon", [])]
    inc = correlate(evs, window_s=3600)[0]
    ids = [t["id"] for t in inc["techniques"]]
    assert ids.index("T1110") < ids.index("T1078")


def test_zeek_json_without_path_keeps_its_log_type() -> None:
    """2026-10-23: Zeek JSON records without _path were all read as conn.log (DCE-RPC,
    DNS and HTTP fields - and the Zeek Sigma rules for them - were lost)."""
    import json

    from src.soc.parsers import parse_text

    rows = [{"ts": "2022-07-20T06:51:37Z", "uid": "C1", "id.orig_h": "10.0.1.16", "id.orig_p": 50152,
             "id.resp_h": "10.0.1.14", "id.resp_p": 49666, "endpoint": "drsuapi", "operation": "DRSGetNCChanges"},
            {"ts": "2022-07-20T06:51:38Z", "uid": "C2", "id.orig_h": "10.0.1.16", "id.orig_p": 5353,
             "id.resp_h": "10.0.1.1", "id.resp_p": 53, "query": "evil.example", "qtype_name": "A"}]  # fmt: skip
    out = parse_text("\n".join(json.dumps(r) for r in rows), "zeek.json")
    evs = out[0] if isinstance(out, tuple) else out
    assert [e["kind"] for e in evs] == ["zeek_dce_rpc", "dns_query"]
    assert "drsuapi DRSGetNCChanges" in evs[0]["detail"]


def test_actions_inside_a_network_logon_session_are_lateral_movement() -> None:
    """2026-10-23: a process / scheduled task created inside a user's network logon (4624
    type 3) on this host is remote execution (T1021); machine accounts and local sessions
    are not."""
    from src.soc import parsers

    parsers.reset_sessions()
    logon = {"EventID": 4624, "LogonType": "3", "TargetUserName": "alice", "TargetLogonId": "0x1865b632"}
    machine = {"EventID": 4624, "LogonType": "3", "TargetUserName": "WS01$", "TargetLogonId": "0x99"}
    for ev in (logon, machine):
        parsers.parse_winsec(ev)
    remote = parsers.parse_winsec({"EventID": 4688, "SubjectLogonId": "0x3e7", "TargetLogonId": "0x1865b632",
                                   "Message": "A new process has been created."})  # fmt: skip
    task = parsers.parse_winsec({"EventID": 4698, "SubjectLogonId": "0x1865B632", "Message": "A scheduled task was created."})
    local = parsers.parse_winsec({"EventID": 4688, "SubjectLogonId": "0x3e7", "TargetLogonId": "0x0", "Message": "x"})
    by_machine = parsers.parse_winsec({"EventID": 4688, "SubjectLogonId": "0x99", "Message": "x"})
    assert remote and "T1021" in remote["techniques"] and remote["kind"] == "remote_process"
    assert task and "T1021" in task["techniques"]
    assert local and "T1021" not in local["techniques"] and by_machine and "T1021" not in by_machine["techniques"]
    parsers.reset_sessions()


def test_syslog_lines_keep_their_host() -> None:
    """2026-10-24 (Linux host test): the sending host of RFC 3164 / 5424 / ISO syslog lines."""
    from src.soc.parsers import parse_syslog, syslog_host

    assert syslog_host("Oct 24 10:31:21 web9 sshd[4000]: Failed password for root") == "web9"
    assert syslog_host("<165>1 2003-10-11T22:14:15.003Z mail.example.com app - ID47 - x") == "mail.example.com"
    assert syslog_host("2026-10-24T05:00:08+00:00 db01 systemd[1]: Started x") == "db01"
    assert syslog_host("no header here") is None
    ev = parse_syslog("Oct 24 10:31:21 web9 sudo[77]: alice : TTY=pts/0 ; COMMAND=/bin/bash")
    assert ev and ev["host"] == "web9"
