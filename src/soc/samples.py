"""Demo logs for the dashboard: one staged intrusion seen by several sensors.

Story (all within one hour, so the correlator joins it into incidents):
  203.0.113.45  scans the web server (Zeek + Suricata), tries SQL injection
                (Suricata), brute-forces RDP (Windows 4625 x8) and gets in (4624),
                and its flows are classified by the detector (flow CSV).
  192.0.2.10    brute-forces SSH and succeeds (Linux auth.log / syslog).
  198.51.100.7  sweeps ports and is dropped by Windows Firewall.
  host-local    a new account, an encoded PowerShell and a cleared audit log.
Addresses are from the RFC 5737 documentation ranges, never real hosts.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.paths import FIXTURES_DIR, PROJECT_ROOT

SAMPLES = PROJECT_ROOT / "data" / "samples"
ATTACKER, SSH_ATTACKER, SCANNER, SERVER = (
    "203.0.113.45",
    "192.0.2.10",
    "198.51.100.7",
    "10.0.0.5",
)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat()


def make_samples(out: Path = SAMPLES) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time() - 3000
    files: list[Path] = []

    # Flows (detector input): fixture rows with addresses + timestamps added.
    df = (
        pd.read_csv(FIXTURES_DIR / "fixture.csv")
        .dropna()
        .head(120)
        .reset_index(drop=True)
    )
    benign = df["Label"].astype(str).str.upper().eq("BENIGN")
    df.insert(0, "Timestamp", [_iso(t0 + 600 + i * 5) for i in range(len(df))])
    df.insert(
        1,
        "Src IP",
        [f"10.0.0.{20 + i % 30}" if b else ATTACKER for i, b in enumerate(benign)],
    )
    df.insert(2, "Dst IP", SERVER)
    p = out / "flows_sample.csv"
    df.drop(columns=["Label"]).to_csv(p, index=False)
    files.append(p)

    # Zeek conn.log: rejected connections to 30 ports (scan).
    lines = [
        "#separator \\x09",
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tconn_state",
    ]
    lines += [
        f"{t0 + i:.6f}\tC{i}\t{ATTACKER}\t{40000 + i}\t{SERVER}\t{20 + i}\ttcp\t-\tREJ"
        for i in range(30)
    ]
    p = out / "conn.log"
    p.write_text("\n".join(lines) + "\n", "utf-8")
    files.append(p)

    # Suricata eve.json: scan + SQL injection signatures.
    eve = [
        {
            "timestamp": _iso(t0 + 60),
            "event_type": "alert",
            "src_ip": ATTACKER,
            "dest_ip": SERVER,
            "dest_port": 80,
            "alert": {
                "signature": "ET SCAN Nmap Scripting Engine User-Agent Detected",
                "severity": 2,
            },
        },
        {
            "timestamp": _iso(t0 + 400),
            "event_type": "alert",
            "src_ip": ATTACKER,
            "dest_ip": SERVER,
            "dest_port": 80,
            "alert": {
                "signature": "ET WEB_SERVER Possible SQL Injection Attempt UNION SELECT",
                "severity": 1,
            },
        },
    ]
    p = out / "eve.json"
    p.write_text("\n".join(json.dumps(e) for e in eve) + "\n", "utf-8")
    files.append(p)

    # Windows Security events (Get-WinEvent | ConvertTo-Json shape).
    ev = [
        {
            "Id": 4625,
            "TimeCreated": _iso(t0 + 900 + i * 10),
            "Message": f"An account failed to log on.\nSource Network Address:\t{ATTACKER}",
        }
        for i in range(8)
    ]
    ev.append(
        {
            "Id": 4624,
            "TimeCreated": _iso(t0 + 1000),
            "Message": f"An account was successfully logged on.\nSource Network Address:\t{ATTACKER}",
        }
    )
    ev.append(
        {
            "Id": 4720,
            "TimeCreated": _iso(t0 + 1100),
            "Message": "A user account was created.",
        }
    )
    ev.append(
        {
            "Id": 4688,
            "TimeCreated": _iso(t0 + 1150),
            "Message": "A new process has been created.\nProcess Command Line: powershell.exe -enc SQBFAFgA",
        }
    )
    ev.append(
        {
            "Id": 1102,
            "TimeCreated": _iso(t0 + 1200),
            "Message": "The audit log was cleared.",
        }
    )
    p = out / "winsec.json"
    p.write_text(json.dumps(ev, indent=1), "utf-8")
    files.append(p)

    # Linux auth.log (generic syslog).
    stamp = lambda t: datetime.fromtimestamp(t).strftime("%b %d %H:%M:%S")  # noqa: E731
    auth = [
        f"{stamp(t0 + 1500 + i * 3)} web01 sshd[811]: Failed password for root from {SSH_ATTACKER} port 51{i:03d} ssh2"
        for i in range(10)
    ]
    auth.append(
        f"{stamp(t0 + 1560)} web01 sshd[811]: Accepted password for root from {SSH_ATTACKER} port 51999 ssh2"
    )
    p = out / "auth.log"
    p.write_text("\n".join(auth) + "\n", "utf-8")
    files.append(p)

    # Windows Firewall log: drops across 25 ports.
    fw = [
        "#Version: 1.5",
        "#Software: Microsoft Windows Firewall",
        "#Fields: date time action protocol src-ip dst-ip src-port dst-port size tcpflags tcpsyn tcpack tcpwin icmptype icmpcode info path",
    ]
    for i in range(25):
        t = datetime.fromtimestamp(t0 + 2000 + i)
        fw.append(
            f"{t:%Y-%m-%d} {t:%H:%M:%S} DROP TCP {SCANNER} {SERVER} {50000 + i} {1000 + i} 0 S 0 0 0 - - - RECEIVE"
        )
    p = out / "pfirewall.log"
    p.write_text("\n".join(fw) + "\n", "utf-8")
    files.append(p)
    return files
