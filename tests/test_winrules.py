"""Windows endpoint rules (src/soc/winrules.py): behaviours map to techniques, routine
activity and other providers' look-alike event ids do not."""

from __future__ import annotations

import pytest

from src.knowledge.attack_details import TECHNIQUE_TACTICS
from src.soc.correlate import TECHNIQUES
from src.soc.parsers import parse_winsec

SYSMON = "Microsoft-Windows-Sysmon"


def _sys(eid: int, **fields: str) -> list[str]:
    msg = "RuleName: -\n" + "\n".join(f"{k}: {v}" for k, v in fields.items())
    ev = parse_winsec({"Id": eid, "ProviderName": SYSMON, "Message": msg})
    assert ev is not None
    return ev["techniques"]


@pytest.mark.parametrize(
    ("fields", "tech"),
    [
        ({"Image": r"C:\Windows\System32\cmd.exe", "CommandLine": "cmd.exe",
          "ParentImage": r"C:\Windows\System32\fodhelper.exe"}, "T1548"),
        ({"Image": r"C:\Windows\System32\cmd.exe", "CommandLine": "cmd.exe /c whoami",
          "ParentImage": r"C:\Windows\System32\wbem\WmiPrvSE.exe"}, "T1047"),
        ({"Image": r"C:\Windows\System32\whoami.exe", "CommandLine": "whoami /all",
          "ParentImage": r"C:\Windows\System32\cmd.exe"}, "T1033"),
        ({"Image": r"C:\Windows\System32\net.exe", "CommandLine": "net localgroup administrators",
          "ParentImage": r"C:\Windows\System32\cmd.exe"}, "T1069"),
        ({"Image": r"C:\Windows\System32\rundll32.exe",
          "CommandLine": "rundll32.exe advpack.dll,RegisterOCX calc.exe",
          "ParentImage": r"C:\Windows\explorer.exe"}, "T1218"),
        ({"Image": r"C:\Windows\System32\cmd.exe",
          "CommandLine": r"cmd.exe /c echo x > \\.\pipe\x",
          "ParentImage": r"C:\Windows\System32\services.exe"}, "T1134"),
        ({"Image": r"C:\Windows\System32\schtasks.exe",
          "CommandLine": "schtasks /create /tn x /tr calc.exe",
          "ParentImage": r"C:\Windows\System32\cmd.exe"}, "T1053"),
    ],
)  # fmt: skip
def test_process_behaviours(fields: dict[str, str], tech: str) -> None:
    assert tech in _sys(1, **fields)


def test_registry_and_other_sysmon_events() -> None:
    uac = r"HKU\S-1-5-21-1_Classes\ms-settings\shell\open\command\(Default)"
    assert _sys(13, TargetObject=uac, Details="cmd.exe") == ["T1548"]
    logging_off = r"HKLM\SOFTWARE\Policies\Microsoft\Windows\PowerShell\ScriptBlockLogging\EnableScriptBlockLogging"
    assert _sys(13, TargetObject=logging_off) == ["T1562"]
    assert _sys(20, Operation="Created") == ["T1546"]
    assert _sys(17, PipeName=r"\RoguePotato\pipe\epmapper") == ["T1134"]
    assert _sys(
        10, TargetImage=r"C:\x\notepad.exe", CallTrace="ntdll.dll+1|UNKNOWN(01B5)"
    ) == ["T1055"]


def test_rulename_labels_are_ignored() -> None:
    # the sample author's own label must not create a detection
    msg = "RuleName: technique_id=T1003,technique_name=mimikatz\nImage: C:\\Windows\\System32\\calc.exe\nCommandLine: calc.exe"
    ev = parse_winsec({"Id": 1, "ProviderName": SYSMON, "Message": msg})
    assert ev is not None and ev["techniques"] == []


def test_routine_activity_and_other_providers_stay_quiet() -> None:
    quiet = [
        {"Id": 4625, "ProviderName": "Microsoft-Windows-EventSystem",
         "Message": "The EventSystem sub system is suppressing duplicate event log entries"},
        {"Id": 161, "ProviderName": "Microsoft-Windows-WinRM",
         "Message": "The client cannot connect to the destination specified in the request."},
        {"Id": 1, "ProviderName": SYSMON, "Message":
         "Image: C:\\Windows\\System32\\notepad.exe\nCommandLine: notepad.exe a.txt\n"
         "ParentImage: C:\\Windows\\explorer.exe"},
    ]  # fmt: skip
    for obj in quiet:
        ev = parse_winsec(obj)
        assert ev is not None and ev["techniques"] == [], obj
    # without a provider (older exports, unit fixtures) Security ids still map
    ev = parse_winsec({"Id": 4625, "Message": "An account failed to log on."})
    assert ev is not None and ev["techniques"] == ["T1110"]


def test_security_rules() -> None:
    sec = "Microsoft-Windows-Security-Auditing"
    kerb = {"Id": 4769, "ProviderName": sec, "Message": "Ticket Encryption Type:\t0x17"}
    assert parse_winsec(kerb)["techniques"] == ["T1558"]  # type: ignore[index]
    dcsync = {"Id": 4662, "ProviderName": sec,
              "Message": "Properties: {1131f6aa-9c07-11d1-f79f-00c04fc2dcd2}"}  # fmt: skip
    assert parse_winsec(dcsync)["techniques"] == ["T1003"]  # type: ignore[index]


def test_every_new_technique_is_known_everywhere() -> None:
    assert set(TECHNIQUE_TACTICS) == set(TECHNIQUES)
    for t in ("T1548", "T1134", "T1546", "T1047", "T1033", "T1558", "T1572"):
        assert TECHNIQUES[t][1] in TECHNIQUE_TACTICS[t]
