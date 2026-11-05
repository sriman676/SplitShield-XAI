#!/usr/bin/env python3
"""SplitShield-XAI log agent for Linux, macOS and Windows (Python 3.8+, standard
library only - copy this one file to the machine).

It reads new log records and sends them in batches over HTTPS to the SplitShield-XAI
log receiver (src/soc/collector.py, port 8600):

  * Linux   - systemd journal (``journalctl -o json``) and/or log files such as
              /var/log/auth.log, /var/log/secure, /var/log/syslog (followed like
              ``tail -F``, rotation-safe)
  * macOS   - the unified log (``log show --style ndjson``) and/or log files
  * Windows - Security, Sysmon, PowerShell and System event logs (``wevtutil``)

Delivery is guaranteed: where each source stopped is saved only after the receiver
confirmed the batch, and batches that cannot be sent are kept in a local spool and
re-sent in order. The receiver's certificate is pinned by its SHA-256 fingerprint
(shown on the dashboard), so no certificate authority is needed and an impostor
server is refused.

    python3 splitshield_agent.py --setup      # asks server, key, fingerprint once
    python3 splitshield_agent.py              # run (Ctrl+C to stop)
    python3 splitshield_agent.py --once       # one pass (for cron / testing)

Config and state live next to this file (agent.json, agent_state.json, spool/) or in
the folder given with --home. Install as a service: scripts/agents/install_agent.sh
(Linux systemd / macOS launchd) or install_agent.ps1 (Windows Task Scheduler).
"""

from __future__ import annotations

import argparse
import getpass
import gzip
import hashlib
import json
import os
import platform
import socket
import ssl
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

VERSION = "1.0"
BATCH_LINES = 2000
EVERY_S = 2.0  # seconds between passes (was 10 s; --every changes it)
SPOOL_MAX_FILES = 10000
DEFAULT_FILES = {
    "Linux": ["/var/log/auth.log", "/var/log/secure", "/var/log/syslog", "/var/log/messages"],
    "Darwin": ["/var/log/system.log"],
    "Windows": [],
}
WIN_LOGS = {"security": "Security", "sysmon": "Microsoft-Windows-Sysmon/Operational",
            "powershell": "Microsoft-Windows-PowerShell/Operational", "system": "System"}
NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"


# --- config / state --------------------------------------------------------------------


class Agent:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.cfg_path = home / "agent.json"
        self.state_path = home / "agent_state.json"
        self.spool = home / "spool"
        self.cfg: dict[str, Any] = {}
        self.state: dict[str, Any] = {}

    def load(self) -> None:
        self.cfg = json.loads(self.cfg_path.read_text("utf-8"))
        try:
            self.state = json.loads(self.state_path.read_text("utf-8"))
        except (OSError, ValueError):
            self.state = {}

    def save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), "utf-8")
        os.replace(tmp, self.state_path)

    # --- sending ---------------------------------------------------------------------

    def _context(self) -> ssl.SSLContext:
        ca = self.cfg.get("ca_file")
        if ca:  # the organisation's own certificate: chain + host name checked normally
            return ssl.create_default_context(cafile=None if ca == "system" else str(ca))
        ctx = ssl.create_default_context()
        ctx.check_hostname = False  # identity = the pinned fingerprint below
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _post(self, path: str, payload: bytes | None) -> tuple[int, dict[str, Any]]:
        """HTTPS request with certificate pinning (checked before any data is sent)."""
        url = self.cfg["server"].rstrip("/")
        host_port = url.split("://", 1)[1].split("/", 1)[0]
        host, _, port = host_port.rpartition(":")
        host = host.strip("[]")
        with socket.create_connection((host, int(port or 8600)), timeout=30) as raw:
            with self._context().wrap_socket(raw, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True) or b""
                pin = str(self.cfg.get("fingerprint") or "").lower().replace(":", "")
                if not pin and not self.cfg.get("ca_file"):
                    raise ssl.SSLError("no pinned fingerprint and no ca_file: refusing an unverified server")
                if pin and hashlib.sha256(der).hexdigest() != pin:  # with ca_file: optional extra pin
                    raise ssl.SSLError("server certificate does not match the pinned fingerprint")
                method = "POST" if payload is not None else "GET"
                head = [f"{method} {path} HTTP/1.1", f"Host: {host_port}",
                        f"Authorization: Bearer {self.cfg['key']}",
                        f"User-Agent: splitshield-agent/{VERSION}", "Connection: close"]
                if payload is not None:
                    head += ["Content-Type: application/json", "Content-Encoding: gzip",
                             f"Content-Length: {len(payload)}"]
                tls.sendall(("\r\n".join(head) + "\r\n\r\n").encode() + (payload or b""))
                resp = b""
                while True:
                    chunk = tls.recv(65536)
                    if not chunk:
                        break
                    resp += chunk
        status_line, _, rest = resp.partition(b"\r\n")
        code = int(status_line.split()[1]) if status_line else 0
        body = rest.partition(b"\r\n\r\n")[2]
        try:
            return code, json.loads(body.decode("utf-8") or "{}")
        except ValueError:
            return code, {}

    def ping(self) -> tuple[int, dict[str, Any]]:
        return self._post("/agent/ping", None)

    def _spool_add(self, batch: dict[str, Any]) -> None:
        self.spool.mkdir(parents=True, exist_ok=True)
        files = sorted(self.spool.glob("*.json.gz"))
        if len(files) >= SPOOL_MAX_FILES:  # disk guard: the oldest is kept as .overflow
            files[0].rename(files[0].with_suffix(".overflow"))
        name = f"{time.time():.6f}-{os.getpid()}.json.gz"
        (self.spool / name).write_bytes(gzip.compress(json.dumps(batch).encode()))

    def _send_raw(self, payload: bytes) -> bool:
        try:
            code, out = self._post("/ingest", payload)
        except (OSError, ssl.SSLError, ValueError) as exc:
            log(f"receiver unreachable: {exc}")
            return False
        if code == 200:
            return True
        log(f"receiver answered {code}: {out.get('error', '')}")
        return False

    def flush_spool(self) -> bool:
        """Re-send spooled batches oldest first; False if the receiver is still down."""
        if not self.spool.is_dir():
            return True
        for f in sorted(self.spool.glob("*.json.gz")):
            if not self._send_raw(f.read_bytes()):
                return False
            f.unlink()
        return True

    def send(self, batch: dict[str, Any]) -> None:
        """Send now, or keep in the spool (never lost)."""
        if not self.flush_spool() or not self._send_raw(gzip.compress(json.dumps(batch).encode())):
            self._spool_add(batch)

    # --- sources ---------------------------------------------------------------------

    def run_once(self) -> int:
        n = 0
        host = socket.gethostname()
        for path in self.cfg.get("files", []):
            n += self._file(Path(path), host)
        if self.cfg.get("journald") and platform.system() == "Linux":
            n += self._journald(host)
        if self.cfg.get("macos_unified") and platform.system() == "Darwin":
            n += self._unified(host)
        for name in self.cfg.get("windows_logs", []):
            if platform.system() == "Windows":
                n += self._windows(name, host)
        self.flush_spool()
        return n

    def _emit(self, source: str, host: str, key: str, value: Any, *, lines: list[str] | None = None,
              events: list[dict[str, Any]] | None = None) -> None:
        batch: dict[str, Any] = {"source": source, "host": host, "agent_version": VERSION,
                                 "sent": time.time(),  # the receiver checks clock skew
                                 # the receiver drops a batch it already stored (its answer
                                 # was lost and the batch came back from the spool)
                                 "batch_id": uuid.uuid4().hex}
        if lines is not None:
            batch["lines"] = lines
        else:
            batch["events"] = events or []
        self.send(batch)  # sent or spooled -> safe to move the bookmark
        self.state[key] = value
        self.save_state()

    def _file(self, path: Path, host: str) -> int:
        """Follow a text log; rotation-safe (new inode or shrunk file restarts at 0)."""
        try:
            st = path.stat()
        except OSError:
            return 0
        key = f"file:{path}"
        mark = self.state.get(key)
        if mark is None:  # first run: from the end, unless backfill was asked for
            pos = 0 if self.cfg.get("backfill") else st.st_size
            self.state[key] = {"ino": st.st_ino, "pos": pos}
            self.save_state()
        else:
            pos = int(mark.get("pos", 0))
            if mark.get("ino") != st.st_ino or pos > st.st_size:
                pos = 0  # rotated or truncated
        sent = 0
        with open(path, "rb") as fh:
            fh.seek(pos)
            while True:
                chunk = fh.readlines(4 * 1024 * 1024)
                if not chunk:
                    break
                if not chunk[-1].endswith(b"\n"):  # half-written last line: next time
                    chunk = chunk[:-1]
                    if not chunk:
                        break
                for i in range(0, len(chunk), BATCH_LINES):
                    part = chunk[i:i + BATCH_LINES]
                    pos += sum(len(c) for c in part)
                    lines = [c.decode("utf-8", "replace").rstrip("\r\n") for c in part]
                    self._emit(path.name, host, key, {"ino": st.st_ino, "pos": pos}, lines=lines)
                    sent += len(part)
                fh.seek(pos)
        return sent

    def _journald(self, host: str) -> int:
        key = "journald"
        args = ["journalctl", "-o", "json", "--no-pager", "-n", str(BATCH_LINES)]
        cur = self.state.get(key)
        args += ["--after-cursor", cur] if cur else ["--since", "-5min"]
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=120).stdout
        except (OSError, subprocess.TimeoutExpired):
            return 0
        lines, last = [], cur
        for row in out.splitlines():
            try:
                e = json.loads(row)
            except ValueError:
                continue
            ts = float(e.get("__REALTIME_TIMESTAMP", time.time() * 1e6)) / 1e6
            ident = e.get("SYSLOG_IDENTIFIER") or e.get("_COMM") or "kernel"
            pid = e.get("_PID", "")
            msg = e.get("MESSAGE", "")
            if isinstance(msg, list):  # binary messages come as byte lists
                msg = bytes(msg).decode("utf-8", "replace")
            lines.append(f"{syslog_time(ts)} {e.get('_HOSTNAME', host)} {ident}"
                         f"{f'[{pid}]' if pid else ''}: {msg}")  # fmt: skip
            last = e.get("__CURSOR", last)
        if lines:
            self._emit("journald", host, key, last, lines=lines)
        return len(lines)

    def _unified(self, host: str) -> int:
        key = "macos_unified"
        since = float(self.state.get(key) or time.time() - 300)
        start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))
        pred = self.cfg.get("macos_predicate") or (
            'process == "sshd" OR process == "sudo" OR process == "loginwindow" OR '
            'process == "authd" OR subsystem == "com.apple.securityd" OR '
            'eventMessage CONTAINS[c] "authentication"')
        try:
            out = subprocess.run(["log", "show", "--style", "ndjson", "--start", start,
                                  "--predicate", pred], capture_output=True, text=True,
                                 timeout=300).stdout  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            return 0
        lines, last = [], since
        for row in out.splitlines():
            try:
                e = json.loads(row)
            except ValueError:
                continue
            try:
                ts = time.mktime(time.strptime(e["timestamp"][:19], "%Y-%m-%d %H:%M:%S"))
            except (KeyError, ValueError):
                continue
            if ts <= since:
                continue
            proc = Path(str(e.get("processImagePath", "") or "")).name or e.get("process", "log")
            lines.append(f"{syslog_time(ts)} {host} {proc}[{e.get('processID', 0)}]: "
                         f"{e.get('eventMessage', '')}")  # fmt: skip
            last = max(last, ts)
        if lines:
            for i in range(0, len(lines), BATCH_LINES):
                self._emit("unified", host, key, last, lines=lines[i:i + BATCH_LINES])
        return len(lines)

    def _windows(self, name: str, host: str) -> int:
        log_name = WIN_LOGS.get(name, name)
        key = f"win:{log_name}"
        mark = self.state.get(key)
        if mark is None:  # first run: start at the newest record
            out = _wevtutil(["qe", log_name, "/c:1", "/rd:true", "/f:xml"])
            rows = _win_records(f"<Events>{out}</Events>") if out else []
            self.state[key] = rows[0][0] if rows else 0
            self.save_state()
            return 0
        out = _wevtutil(["qe", log_name, f"/q:*[System[(EventRecordID>{int(mark)})]]",
                         f"/c:{BATCH_LINES}", "/f:RenderedXml", "/e:Events", "/uni:true"])
        rows = _win_records(out) if out else []
        if rows:
            self._emit(name, host, key, max(r for r, _ in rows), events=[e for _, e in rows])
        return len(rows)


# --- helpers ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def syslog_time(ts: float) -> str:
    t = time.localtime(ts)
    return f"{time.strftime('%b', t)} {t.tm_mday:2d} {time.strftime('%H:%M:%S', t)}"


def _wevtutil(args: list[str]) -> str:
    try:
        r = subprocess.run(["wevtutil", *args], capture_output=True, timeout=120,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return ""
    raw = r.stdout
    text = raw.decode("utf-16-le", "replace") if raw[:2] == b"\xff\xfe" or b"\x00<" in raw[:8] \
        else raw.decode("utf-8", "replace")  # fmt: skip
    return text if r.returncode == 0 else ""


def _win_records(xml_text: str) -> list[tuple[int, dict[str, Any]]]:
    """wevtutil XML -> rows shaped like Get-WinEvent (what the receiver's parsers read)."""
    text = xml_text.strip().lstrip("﻿")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    out = []
    for ev in root.iter(f"{NS}Event"):
        sysn = ev.find(f"{NS}System")
        if sysn is None:
            continue
        prov, tc = sysn.find(f"{NS}Provider"), sysn.find(f"{NS}TimeCreated")
        data = {d.get("Name"): (d.text or "") for d in ev.iter(f"{NS}Data") if d.get("Name")}
        msg = ev.findtext(f"{NS}RenderingInfo/{NS}Message") or ""
        if prov is not None and prov.get("Name") == "Microsoft-Windows-Sysmon" and data:
            title = msg.splitlines()[0] if msg else ""
            msg = "\n".join([title, *(f"{k}: {v}" for k, v in data.items())])
        if not msg:
            msg = "\n".join(f"{k}: {v}" for k, v in data.items())
        out.append((int((sysn.findtext(f"{NS}EventRecordID") or "0").strip()), {
            "Id": int((sysn.findtext(f"{NS}EventID") or "0").strip()),
            "TimeCreated": tc.get("SystemTime") if tc is not None else None,
            "ProviderName": prov.get("Name") if prov is not None else None,
            "Channel": sysn.findtext(f"{NS}Channel"), "Computer": sysn.findtext(f"{NS}Computer"),
            "Message": msg, "EventData": data}))  # fmt: skip
    return out


def setup(agent: Agent) -> int:
    """Interactive first-time setup (key typed hidden)."""
    system = platform.system()
    server = input("Receiver address (e.g. https://192.168.1.10:8600): ").strip()
    key = getpass.getpass("Agent key (from the dashboard, starts with ssa_): ").strip()
    fp = input("Certificate fingerprint (SHA-256, from the dashboard): ").strip().lower().replace(":", "")
    cfg: dict[str, Any] = {"server": server, "key": key, "fingerprint": fp,
                           "files": [f for f in DEFAULT_FILES.get(system, []) if Path(f).exists()],
                           "journald": system == "Linux", "macos_unified": system == "Darwin",
                           "windows_logs": list(WIN_LOGS) if system == "Windows" else [],
                           "backfill": False}  # fmt: skip
    agent.home.mkdir(parents=True, exist_ok=True)
    agent.cfg_path.write_text(json.dumps(cfg, indent=1), "utf-8")
    if os.name == "posix":
        os.chmod(agent.cfg_path, 0o600)  # the key is a secret
    agent.load()
    try:
        code, out = agent.ping()
    except (OSError, ssl.SSLError) as exc:
        print(f"Saved, but the receiver could not be reached: {exc}")
        return 1
    print("Connected as agent", out.get("agent") if code == 200 else f"- refused ({code})")
    return 0 if code == 200 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=EVERY_S)
    a = ap.parse_args(argv)
    agent = Agent(Path(a.home))
    if a.setup:
        return setup(agent)
    try:
        agent.load()
    except (OSError, ValueError):
        print("No agent.json yet - run with --setup first.")
        return 1
    while True:
        try:
            n = agent.run_once()
            if n:
                log(f"sent {n} records")
        except Exception as exc:  # noqa: BLE001 - an agent must keep running
            log(f"error: {exc}")
        if a.once:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
