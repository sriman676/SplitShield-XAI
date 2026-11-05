"""Watch this PC's Windows logs live (owner 2026-10-16).

    python -m src.soc.winlive            # keep watching (every 2 s)
    python -m src.soc.winlive --once     # one pass, then exit (prints the status)

Every few seconds the new records of Security, Sysmon and PowerShell are read with
Windows' own ``wevtutil`` (no PowerShell start per poll), parsed by the SOC's normal
parsers and stored through ``engine.record`` - so they get the same detection,
correlation, cross-checks, allow-list, place budget and SIEM forwarding as an
uploaded log. Read-only: nothing is ever blocked or changed on the PC.

Each log continues from the last record number it saw (artifacts/soc/winlive_state.json),
so nothing is read twice or missed between polls or restarts; the first time, it
starts from the newest ``FIRST_BACKFILL`` records. Reading Security and Sysmon needs
membership of Windows' built-in "Event Log Readers" group (owner decision: no
administrator rights for the SOC):

    net localgroup "Event Log Readers" %USERNAME% /add      (once, as administrator;
                                                            then sign out and in)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.soc import engine, sinks
from src.soc.parsers import WINDOWS_LOGS, parse_text
from utils.logging_config import get_logger

_log = get_logger(__name__)
NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"
FIRST_BACKFILL = 500
BATCH = 2000  # records per log per poll (a burst continues next poll)
EVERY_S = 2.0  # default; the Settings value "live.poll_s" wins (was 10 s)
FIX = 'net localgroup "Event Log Readers" %USERNAME% /add   (as administrator, then sign out and in)'


def state_path() -> Path:
    return Path(engine.EVENTS).with_name("winlive_state.json")


def load_state() -> dict[str, Any]:
    try:
        return json.loads(state_path().read_text("utf-8"))
    except (OSError, ValueError):
        return {"logs": {}}


def _save_state(st: dict[str, Any]) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), "utf-8")
    tmp.replace(p)


def _wevtutil(args: list[str]) -> tuple[int, str]:
    r = subprocess.run(
        ["wevtutil", *args], capture_output=True, timeout=120, creationflags=_NO_WINDOW
    )
    out = (
        r.stdout.decode("utf-16", errors="replace")
        if r.stdout[:2] == b"\xff\xfe"
        else (r.stdout.decode("utf-8", errors="replace"))
    )
    err = r.stderr.decode("utf-8", errors="replace") if r.returncode else ""
    return r.returncode, out if r.returncode == 0 else (err or out)


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def newest_record(log: str) -> int | None:
    """The newest record number of a log (None if it cannot be read)."""
    rc, out = _wevtutil(
        ["qe", log, "/c:1", "/rd:true", "/f:xml", "/e:Events", "/uni:true"]
    )
    if rc != 0:
        return None
    rows = _records(out)
    return rows[0][0] if rows else 0


def _records(xml_text: str) -> list[tuple[int, dict[str, Any]]]:
    """wevtutil RenderedXml -> [(record number, Get-WinEvent-like row)]: the same row
    shape parsers.read_evtx feeds to parse_text (Id, TimeCreated, ProviderName,
    Channel, Message, EventData)."""
    text = xml_text.strip().lstrip("﻿")
    if not text:
        return []
    root = ET.fromstring(text)
    out = []
    for ev in root.iter(f"{NS}Event"):
        sysn = ev.find(f"{NS}System")
        if sysn is None:
            continue
        rid = int((sysn.findtext(f"{NS}EventRecordID") or "0").strip())
        prov = sysn.find(f"{NS}Provider")
        tc = sysn.find(f"{NS}TimeCreated")
        data = {
            d.get("Name"): (d.text or "") for d in ev.iter(f"{NS}Data") if d.get("Name")
        }
        msg = ev.findtext(f"{NS}RenderingInfo/{NS}Message") or ""
        if prov is not None and prov.get("Name") == "Microsoft-Windows-Sysmon" and data:
            # the event's own named fields, never a template's positions (see read_evtx)
            title = msg.splitlines()[0] if msg else ""
            msg = "\n".join([title, *(f"{k}: {v}" for k, v in data.items())])
        if not msg:  # provider not installed here: rebuild like read_evtx does
            msg = "\n".join(f"{k}: {v}" for k, v in data.items())
        out.append(
            (
                rid,
                {
                    "Id": int((sysn.findtext(f"{NS}EventID") or "0").strip()),
                    "TimeCreated": tc.get("SystemTime") if tc is not None else None,
                    "ProviderName": prov.get("Name") if prov is not None else None,
                    "Channel": sysn.findtext(f"{NS}Channel"),
                    "Message": msg,
                    "EventData": data,
                },
            )
        )
    return out


def poll_log(
    name: str, log: str, st: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """New records of one log since its bookmark -> (parsed events, status)."""
    logs = st.setdefault("logs", {})
    mark = logs.get(name, {}).get("last")
    if mark is None:  # first time: start from the newest FIRST_BACKFILL records
        top = newest_record(log)
        if top is None:
            rc, err = _wevtutil(["qe", log, "/c:1"])
            return [], _status_of(err)
        mark = max(0, top - FIRST_BACKFILL)
    q = f"*[System[(EventRecordID>{int(mark)})]]"
    rc, out = _wevtutil(
        [
            "qe",
            log,
            f"/q:{q}",
            f"/c:{BATCH}",
            "/f:RenderedXml",
            "/e:Events",
            "/uni:true",
        ]
    )
    if rc != 0:
        return [], _status_of(out)
    rows = _records(out)
    if not rows:
        logs[name] = {"last": mark, "status": "ok", "checked": time.time()}
        return [], "ok"
    events = parse_text(json.dumps([r for _, r in rows]), f"{name}.json")
    logs[name] = {
        "last": max(r for r, _ in rows),
        "status": "ok",
        "checked": time.time(),
    }
    return events, f"{len(rows)} new records"


def _status_of(err: str) -> str:
    e = err.strip().splitlines()[0] if err.strip() else "unknown error"
    if "denied" in e.lower():
        return f"no permission - {FIX}"
    if "not found" in e.lower() or "could not be found" in e.lower():
        return "not installed on this PC"
    return e[:160]


def poll_once(send: bool = True) -> dict[str, Any]:
    """One pass over every watched log; stores what is new. Returns a summary."""
    st = load_state()
    summary: dict[str, Any] = {"ts": time.time(), "logs": {}, "events": 0, "alerts": 0}
    batch: list[dict[str, Any]] = []
    for name, log in WINDOWS_LOGS.items():
        try:
            evs, status = poll_log(name, log, st)
        except (OSError, subprocess.SubprocessError, ET.ParseError) as exc:
            evs, status = [], f"error: {exc}"[:160]
        if status != "ok" and not status.endswith("new records"):
            st.setdefault("logs", {}).setdefault(name, {})["status"] = status
        summary["logs"][name] = status
        batch += evs
    if batch:
        alerts = engine.record(batch, send)
        summary["events"], summary["alerts"] = len(batch), len(alerts)
    st["last_poll"] = summary
    _save_state(st)  # bookmarks only move on after the events were stored
    return summary


def run(every_s: float | None = None, send: bool = True) -> None:
    """Keep watching. One failing pass is logged, never fatal."""
    if every_s is None:
        every_s = sinks.live_timing("poll_s")
    _log.info("watching %s every %.0f s", ", ".join(WINDOWS_LOGS), every_s)
    while True:
        try:
            s = poll_once(send)
            if s["events"]:
                _log.info("%d new events, %d alerts", s["events"], s["alerts"])
        except Exception as exc:  # noqa: BLE001 - a long-running watcher must survive
            _log.warning("poll failed: %s", exc)
        time.sleep(every_s)


def main() -> int:
    from utils.logging_config import configure_logging

    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=EVERY_S)
    ap.add_argument(
        "--no-send", action="store_true", help="store only, no SIEM forwarding"
    )
    a = ap.parse_args()
    if a.once:
        print(json.dumps(poll_once(not a.no_send), indent=1))
        return 0
    run(a.every, not a.no_send)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
