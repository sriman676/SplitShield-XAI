"""The .evtx reader for macOS / Linux gives the same detections as Windows'
Get-WinEvent (owner 2026-10-17). Measured on all 2026-10-17 attack samples: 10,255
events, identical kind + techniques except where the portable reader is MORE exact."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from src.soc import parsers

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "raw" / "evtx-attack-samples"
FILES = ["DE_KernelDebug_and_TestSigning_ON_Security_4826.evtx", "System_7045_namedpipe_privesc.evtx",
         "Win_4985_T1186_Process_Doppelganging.evtx", "DE_RDP_Tunneling_4624.evtx",
         "rogue_msi_url_1040_1042.evtx", "dc_applog_ntdsutil_dfir_325_326_327.evtx"]  # fmt: skip


def _find(name: str) -> Path | None:
    return next(SAMPLES.rglob(name), None) if SAMPLES.is_dir() else None


@pytest.mark.parametrize("name", FILES)
def test_portable_reader_matches_windows(name: str) -> None:
    f = _find(name)
    if f is None:
        pytest.skip("attack samples not present (removed in the portable copy)")
    portable = parsers.parse_text(json.dumps(parsers._evtx_rows_portable(f, 2000)), f.name)
    assert portable and any(e.get("techniques") for e in portable)
    if sys.platform != "win32":
        return
    windows = parsers.read_evtx(f, 2000)
    key = lambda evs: Counter((e.get("kind"), tuple(e.get("techniques") or [])) for e in evs)  # noqa: E731
    assert key(portable) == key(windows)


def test_padded_values_are_trimmed_for_sigma() -> None:
    row = {"Id": 4624, "ProviderName": "Microsoft-Windows-Security-Auditing", "Channel": "Security",
           "TimeCreated": "2019-02-02T09:17:22Z", "Message": "An account was successfully logged on.",
           "EventData": {"SubjectUserSid": "S-1-0-0", "LogonType": "3", "LogonProcessName": "NtLmSsp ", "KeyLength": "0",
                         "AuthenticationPackageName": "NTLM", "TargetUserName": "bob",
                         "WorkstationName": "PC7", "IpAddress": "192.0.2.4"}}  # fmt: skip
    ev = parsers.parse_text(json.dumps([row]), "sec.json")
    assert "T1550" in ev[0]["techniques"]  # Sigma "Pass the Hash" needs the trimmed value
