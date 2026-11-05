"""Windows rules added 2026-10-16 and the two parser fixes found while scoring them:
(1) a Sysmon event rendered through another Sysmon version's template, (2) a message
title line ("Image loaded:") shadowing the event's own ImageLoaded field for Sigma."""

from __future__ import annotations

import json

from src.soc import parsers, sigma, winrules


def _sec(eid: int, msg: str, provider: str = "Microsoft-Windows-Security-Auditing"):  # noqa: ANN202
    return winrules.security(eid, msg, provider)


def test_new_security_rules_fire_on_the_behaviour() -> None:
    assert _sec(4826, "Kernel Debugging:\tNo\nTest Signing:\t\tYes")[1] == ["T1553"]
    assert _sec(4826, "Kernel Debugging:\tNo\nTest Signing:\t\tNo") is None
    assert _sec(4985, "Process Name:\t\tC:\\Users\\bob\\x.exe")[1] == ["T1055"]
    assert _sec(4985, "Process Name:\t\tC:\\Windows\\System32\\svchost.exe") is None
    anon = ("Logon Type:\t\t\t3\nAccount Name:\t\tANONYMOUS LOGON\n"
            "Source Network Address:\t127.0.0.1\n")  # fmt: skip
    assert _sec(4624, anon)[1] == ["T1134"]
    assert _sec(4624, anon.replace("127.0.0.1", "10.0.0.8")) is None
    efs = "InterfaceUuid: {c681d488-d850-11d0-8c52-00c04fd90f7e}"
    assert _sec(5, efs, "Microsoft-Windows-RPC")[1] == ["T1187"]
    assert (
        _sec(
            5,
            "InterfaceUuid: {647d4452-9f33-4a18-b2be-c5c0e920e94e}",
            "Microsoft-Windows-RPC",
        )
        is None
    )


def test_bits_rule_ignores_vendor_updaters() -> None:
    bits = "Microsoft-Windows-Bits-Client"
    bad = "BITS started the job that is associated with the https://a.uguu.se/x_Bin.7z URL."
    assert _sec(59, bad, bits)[1] == ["T1197"]
    for ok in ("https://dl.google.com/update/chrome.exe", "https://download.windowsupdate.com/x.cab",
               "https://example.org/wallpaper.jpg"):  # fmt: skip
        assert _sec(59, f"associated with the {ok} URL.", bits) is None


def test_new_registry_rules() -> None:
    lsa = winrules.registry(
        "TargetObject: HKLM\\SECURITY\\Policy\\Secrets\\$MACHINE.ACC\\CurrVal"
    )
    share = winrules.registry(
        "TargetObject: HKLM\\System\\CurrentControlSet\\Services\\LanmanServer\\Shares\\staging"
    )
    assert lsa is not None and lsa[1] == ["T1003"]
    assert share is not None and share[1] == ["T1021"]


def test_message_title_never_shadows_a_real_field() -> None:
    # 2026-10-16: "Image loaded:" became an empty "Imageloaded" field, which Sigma's
    # case-blind names read as ImageLoaded -> the fax-service side-loading rule failed
    row = {
        "Id": 7, "ProviderName": "Microsoft-Windows-Sysmon",
        "Channel": "Microsoft-Windows-Sysmon/Operational",
        "Message": "Image loaded:\nImage: C:\\Windows\\System32\\FXSSVC.exe\n"
                   "ImageLoaded: C:\\Windows\\System32\\Ualapi.dll",
        "EventData": {"Image": "C:\\Windows\\System32\\FXSSVC.exe",
                      "ImageLoaded": "C:\\Windows\\System32\\Ualapi.dll", "Signed": "false"},
    }  # fmt: skip
    fields = parsers._fields(row, str(row["Message"]), 7)
    assert "Imageloaded" not in fields
    if sigma.load()["stats"]["loaded"]:  # the Sigma rule pack is present
        ev = parsers.parse_text(json.dumps([row]), "x.json")[0]
        assert "T1574" in ev["techniques"]


def test_evtx_reader_rebuilds_sysmon_messages_from_named_fields() -> None:
    # the PowerShell reader must not trust a template-rendered Sysmon message
    assert "Microsoft-Windows-Sysmon" in parsers._EVTX_PS
    assert "rebuild from the event's own named fields" in parsers._EVTX_PS
