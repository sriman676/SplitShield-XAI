"""This computer's own logs, live, on Windows, macOS and Linux (owner 2026-10-17).

* Windows - src/soc/winlive.py (Security, Sysmon, PowerShell logs via wevtutil).
* Linux   - the systemd journal and /var/log/auth.log, secure, syslog, messages.
* macOS   - the unified log (sign-ins, sudo, sshd, security) and /var/log/system.log.

Linux and macOS reuse the readers of the cross-platform agent
(scripts/agents/splitshield_agent.py) but store the records directly through the SOC
(same parsers, rules and models; no network hop). Where each source stopped is kept in
``oslive/agent_state.json`` next to the event store.

Permissions (read-only): Linux - membership of the ``adm`` and ``systemd-journal``
groups (``sudo usermod -aG adm,systemd-journal $USER``, then sign out and in);
macOS - Full Disk Access for the Python in .venv if log files are unreadable.

    python -m src.soc.oslive            # runs until stopped (the "live logs" job)
    python -m src.soc.oslive --once     # one pass, prints what was read
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from src.soc.osops import PLATFORM
from utils.paths import PROJECT_ROOT

EVERY_S = 2.0  # default; the Settings value "live.poll_s" wins (was 10 s)
_AGENT = PROJECT_ROOT / "scripts" / "agents" / "splitshield_agent.py"


def _agent_module() -> Any:
    spec = importlib.util.spec_from_file_location("splitshield_agent", _AGENT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def home() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).parent / "oslive"


def default_config(platform: str = PLATFORM) -> dict[str, Any]:
    files = {"linux": ["/var/log/auth.log", "/var/log/secure", "/var/log/syslog", "/var/log/messages"],
             "macos": ["/var/log/system.log"]}.get(platform, [])  # fmt: skip
    return {"files": [f for f in files if Path(f).exists()], "journald": platform == "linux",
            "macos_unified": platform == "macos", "windows_logs": [], "backfill": False}  # fmt: skip


def make_local_agent() -> Any:
    """The agent, but storing through the SOC instead of sending over HTTPS."""
    mod = _agent_module()

    class LocalAgent(mod.Agent):  # type: ignore[name-defined]
        def load(self) -> None:
            self.cfg = default_config()
            try:
                self.state = json.loads(self.state_path.read_text("utf-8"))
            except (OSError, ValueError):
                self.state = {}

        def send(self, batch: dict[str, Any]) -> None:
            from src.soc import collector

            collector.ingest_batch("local", batch)

        def flush_spool(self) -> bool:
            return True

    h = home()
    h.mkdir(parents=True, exist_ok=True)
    agent = LocalAgent(h)
    agent.load()
    return agent


def status() -> dict[str, str]:
    """Per source: readable or the reason it is not (shown on Getting started)."""
    import os

    cfg = default_config()
    out = {f: "ok" if os.access(f, os.R_OK) else "no read permission" for f in cfg["files"]}
    if cfg["journald"]:
        import shutil
        import subprocess

        if shutil.which("journalctl"):
            r = subprocess.run(["journalctl", "-n", "1", "-q", "--no-pager"], capture_output=True, text=True)
            out["journald"] = "ok" if r.returncode == 0 and r.stdout.strip() else "no read permission"
        else:
            out["journald"] = "journalctl not found"
    if cfg["macos_unified"]:
        out["unified log"] = "ok"
    return out


def run(once: bool = False, every_s: float | None = None) -> int:
    if PLATFORM == "windows":
        from src.soc import winlive

        if once:
            print(json.dumps(winlive.poll_once(), indent=1, default=str))
            return 0
        winlive.run()
        return 0
    if every_s is None:
        from src.soc import sinks

        every_s = sinks.live_timing("poll_s")
    agent = make_local_agent()
    while True:
        try:
            n = agent.run_once()
            if once:
                print(f"read {n} records", json.dumps(status(), indent=1))
                return 0
        except Exception as exc:  # noqa: BLE001 - a watcher must keep running
            print(f"oslive error: {exc}", flush=True)
            if once:
                return 1
        time.sleep(every_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    return run(a.once)


if __name__ == "__main__":
    raise SystemExit(main())
