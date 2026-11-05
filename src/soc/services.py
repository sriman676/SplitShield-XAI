"""The SOC's background jobs on Windows, macOS and Linux (owner 2026-10-17).

One table (JOBS) describes every job; ``install`` registers them with the platform's
own scheduler, all running as the signed-in user with normal rights:

  * Windows - Task Scheduler (schtasks / Register-ScheduledTask)
  * macOS   - launchd agents in ~/Library/LaunchAgents
  * Linux   - systemd user units (+ timers) in ~/.config/systemd/user; run
              ``loginctl enable-linger $USER`` once so they also run while signed out

    python -m src.soc.services install [--only NAME ...]
    python -m src.soc.services remove  [--only NAME ...]
    python -m src.soc.services status
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.soc.osops import PLATFORM, task_installed
from utils.paths import PROJECT_ROOT

# name -> what it runs (python -m ...), when, and whether it keeps running
JOBS: dict[str, dict[str, Any]] = {
    "SplitShield-XAI autopilot": {"module": "scripts.autopilot", "every_min": 60,
                                  "what": "hourly: drift, retrains, reports, health"},
    "SplitShield-XAI backup": {"module": "scripts.backup", "daily": "03:30",
                               "what": "daily backup of models, results and code"},
    "SplitShield-XAI update agent (daily)": {"module": "scripts.update_agent", "daily": "04:00",
                                             "what": "daily threat-intel / rule updates (only if switched on)"},
    "SplitShield-XAI live logs": {"module": "src.soc.oslive", "keep_running": True,
                                  "what": "reads this computer's own logs every 2 s"},
    "SplitShield-XAI log receiver": {"module": "src.soc.collector", "args": ["serve"], "keep_running": True,
                                     "optional": True, "what": "HTTPS receiver for agents + read-only API"},
}  # fmt: skip


def unit_name(name: str) -> str:
    """'SplitShield-XAI live logs' -> 'splitshield-live-logs' (launchd / systemd names)."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower().replace("splitshield-xai", "splitshield")).strip("-")
    return s


def _python(windowless: bool = True) -> str:
    exe = Path(sys.executable)
    if PLATFORM == "windows" and windowless and (exe.parent / "pythonw.exe").is_file():
        return str(exe.parent / "pythonw.exe")
    return str(exe)


def _windows_install(name: str, job: dict[str, Any]) -> str:
    args = " ".join(["-m", job["module"], *job.get("args", [])])
    if job.get("keep_running"):
        trig = "New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME"
    elif job.get("daily"):
        trig = f"New-ScheduledTaskTrigger -Daily -At {job['daily']}"
    else:
        trig = (f"New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) "
                f"-RepetitionInterval (New-TimeSpan -Minutes {int(job['every_min'])})")  # fmt: skip
    ps = (f"$a = New-ScheduledTaskAction -Execute '{_python()}' -Argument '{args}' "
          f"-WorkingDirectory '{PROJECT_ROOT}'; $t = {trig}; "
          "$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
          "-StartWhenAvailable -MultipleInstances IgnoreNew "
          + ("-ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1); "
             if job.get("keep_running") else "-ExecutionTimeLimit (New-TimeSpan -Hours 6); ")
          + "$p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited; "
          f"Register-ScheduledTask -TaskName '{name}' -Action $a -Trigger $t -Settings $s -Principal $p -Force | Out-Null"
          + (f"; Start-ScheduledTask -TaskName '{name}'" if job.get("keep_running") else ""))  # fmt: skip
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, timeout=120)  # fmt: skip
    return "ok" if r.returncode == 0 else (r.stderr or r.stdout).strip()[:300]


def _mac_plist(name: str, job: dict[str, Any]) -> str:
    label = f"ai.{unit_name(name)}"
    prog = "".join(f"<string>{x}</string>" for x in [_python(), "-m", job["module"], *job.get("args", [])])
    when = ""
    if job.get("keep_running"):
        when = "<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>"
    elif job.get("daily"):
        hh, mm = job["daily"].split(":")
        when = (f"<key>StartCalendarInterval</key><dict><key>Hour</key><integer>{int(hh)}</integer>"
                f"<key>Minute</key><integer>{int(mm)}</integer></dict>")  # fmt: skip
    else:
        when = f"<key>StartInterval</key><integer>{int(job['every_min']) * 60}</integer>"
    log = PROJECT_ROOT / "logs" / f"{unit_name(name)}.log"
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>'
            f"<key>Label</key><string>{label}</string><key>ProgramArguments</key><array>{prog}</array>"
            f"<key>WorkingDirectory</key><string>{PROJECT_ROOT}</string>{when}"
            f"<key>StandardOutPath</key><string>{log}</string><key>StandardErrorPath</key>"
            f"<string>{log}</string></dict></plist>\n")  # fmt: skip


def _systemd_units(name: str, job: dict[str, Any]) -> dict[str, str]:
    u = unit_name(name)
    cmd = " ".join([_python(), "-m", job["module"], *job.get("args", [])])
    svc = (f"[Unit]\nDescription={name}\n\n[Service]\nWorkingDirectory={PROJECT_ROOT}\nExecStart={cmd}\n"
           + ("Restart=always\nRestartSec=60\n\n[Install]\nWantedBy=default.target\n"
              if job.get("keep_running") else "Type=oneshot\n"))  # fmt: skip
    out = {f"{u}.service": svc}
    if not job.get("keep_running"):
        cal = f"*-*-* {job['daily']}:00" if job.get("daily") else f"*:0/{int(job['every_min'])}" \
            if int(job["every_min"]) < 60 else "hourly"  # fmt: skip
        out[f"{u}.timer"] = (f"[Unit]\nDescription={name} (schedule)\n\n[Timer]\nOnCalendar={cal}\n"
                             "Persistent=true\n\n[Install]\nWantedBy=timers.target\n")  # fmt: skip
    return out


def install(only: list[str] | None = None) -> dict[str, str]:
    from src.soc import sinks

    res: dict[str, str] = {}
    receiver_on = bool(sinks.load_settings().get("collector", {}).get("enabled"))
    for name, job in JOBS.items():
        if only and name not in only:
            continue
        if job.get("optional") and not receiver_on and not only:
            res[name] = "skipped (switch the receiver on in Agents & API, then install it)"
            continue
        try:
            if PLATFORM == "windows":
                res[name] = _windows_install(name, job)
            elif PLATFORM == "macos":
                d = Path.home() / "Library" / "LaunchAgents"
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"ai.{unit_name(name)}.plist"
                p.write_text(_mac_plist(name, job), "utf-8")
                subprocess.run(["launchctl", "unload", str(p)], capture_output=True)
                r = subprocess.run(["launchctl", "load", "-w", str(p)], capture_output=True, text=True)
                res[name] = "ok" if r.returncode == 0 else r.stderr.strip()[:300]
            else:
                d = Path.home() / ".config" / "systemd" / "user"
                d.mkdir(parents=True, exist_ok=True)
                units = _systemd_units(name, job)
                for fn, text in units.items():
                    (d / fn).write_text(text, "utf-8")
                subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
                target = next((f for f in units if f.endswith(".timer")), next(iter(units)))
                r = subprocess.run(["systemctl", "--user", "enable", "--now", target],
                                   capture_output=True, text=True)  # fmt: skip
                res[name] = "ok" if r.returncode == 0 else r.stderr.strip()[:300]
        except (OSError, subprocess.SubprocessError) as exc:
            res[name] = f"error: {exc}"
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    return res


def remove(only: list[str] | None = None) -> dict[str, str]:
    res: dict[str, str] = {}
    for name in JOBS:
        if only and name not in only:
            continue
        u = unit_name(name)
        if PLATFORM == "windows":
            r = subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], capture_output=True, text=True)
            res[name] = "removed" if r.returncode == 0 else "not installed"
        elif PLATFORM == "macos":
            p = Path.home() / "Library" / "LaunchAgents" / f"ai.{u}.plist"
            if p.is_file():
                subprocess.run(["launchctl", "unload", "-w", str(p)], capture_output=True)
                p.unlink()
            res[name] = "removed"
        else:
            d = Path.home() / ".config" / "systemd" / "user"
            for fn in (f"{u}.timer", f"{u}.service"):
                subprocess.run(["systemctl", "--user", "disable", "--now", fn], capture_output=True)
                (d / fn).unlink(missing_ok=True)
            res[name] = "removed"
    return res


def status() -> dict[str, bool]:
    return {name: task_installed(name) for name in JOBS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["install", "remove", "status"])
    ap.add_argument("--only", nargs="*", default=None, help="job names (default: all)")
    a = ap.parse_args()
    out: Any = install(a.only) if a.action == "install" else remove(a.only) if a.action == "remove" else status()
    for k, v in out.items():
        print(f"{k}: {v}")
    if PLATFORM == "linux" and a.action == "install":
        print("Linux: run `loginctl enable-linger $USER` once so the jobs also run while you are signed out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
