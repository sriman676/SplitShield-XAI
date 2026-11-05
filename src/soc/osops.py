"""Operating-system actions for Windows, macOS and Linux in one place (owner
2026-10-17: the whole SOC runs on all three).

* ``notify(text)``       desktop pop-up: msg.exe / osascript / notify-send (or wall)
* ``service_running(n)`` an OS service is running: sc / launchctl / systemctl
* ``task_installed(n)``  a SplitShield scheduled job exists: Task Scheduler /
                         launchd agent / systemd user unit (see src/soc/services.py)
* ``block_ip(ip)``       inbound block: Windows Firewall / pf table / nftables set
                         (Linux and macOS need root; without it the reason is returned)

Every function is best effort and never raises: an OS action must never stop detection.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PLATFORM = "windows" if sys.platform == "win32" else "macos" if sys.platform == "darwin" else "linux"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(cmd: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=_NO_WINDOW)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None


def notify(text: str, title: str = "SplitShield-XAI") -> None:
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("SPLITSHIELD_NO_NOTIFY"):
        return
    text = text[:240]
    try:
        if PLATFORM == "windows":
            subprocess.Popen(["msg", "*", "/TIME:3600", f"{title}: {text}"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)  # fmt: skip
        elif PLATFORM == "macos":
            safe = text.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.Popen(["osascript", "-e", f'display notification "{safe}" with title "{title}"'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        elif shutil.which("notify-send"):
            subprocess.Popen(["notify-send", title, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif shutil.which("wall"):
            subprocess.Popen(["wall", f"{title}: {text}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


# OS services the SOC reports on, by platform ("Sysmon" = the endpoint sensor)
SENSOR_SERVICE = {"windows": "Sysmon64", "linux": "auditd", "macos": "com.apple.auditd"}


def service_running(name: str) -> bool:
    if PLATFORM == "windows":
        r = _run(["sc", "query", name], 10)
        return bool(r and "RUNNING" in r.stdout)
    if PLATFORM == "macos":
        r = _run(["launchctl", "list"], 10)
        return bool(r and any(ln.split("\t")[-1] == name and not ln.startswith("-")
                              for ln in r.stdout.splitlines()))  # fmt: skip
    r = _run(["systemctl", "is-active", name], 10)
    return bool(r and r.stdout.strip() == "active")


def task_installed(name: str) -> bool:
    """A SplitShield job (display name, e.g. 'SplitShield-XAI backup')."""
    if PLATFORM == "windows":
        r = _run(["schtasks", "/query", "/tn", name], 15)
        return bool(r and r.returncode == 0)
    from src.soc import services

    unit = services.unit_name(name)
    if PLATFORM == "macos":
        return (Path.home() / "Library" / "LaunchAgents" / f"{unit}.plist").is_file()
    base = Path.home() / ".config" / "systemd" / "user"
    return (base / f"{unit}.service").is_file() or (base / f"{unit}.timer").is_file()


def block_ip(addr: str) -> tuple[bool, str]:
    """Add an inbound block for one (already validated, public) address."""
    if PLATFORM == "windows":
        name = f"name=SplitShield-block-{addr}"
        r = _run(["netsh", "advfirewall", "firewall", "show", "rule", name])
        if r and r.returncode == 0:
            return True, "rule existed"
        r = _run(["netsh", "advfirewall", "firewall", "add", "rule", name, "dir=in",
                  "action=block", f"remoteip={addr}"])  # fmt: skip
        return bool(r and r.returncode == 0), (r.stdout + r.stderr).strip()[:200] if r else "netsh not found"
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        return False, "needs root (run the SOC monitor as root to let it block)"
    if PLATFORM == "macos":
        # table <splitshield> must be referenced by a pf rule: install.sh --firewall adds it
        r = _run(["pfctl", "-t", "splitshield", "-T", "add", addr])
        return bool(r and r.returncode == 0), (r.stderr if r else "pfctl not found").strip()[:200]
    if shutil.which("nft"):
        for cmd in (["nft", "add", "table", "inet", "splitshield"],
                    ["nft", "add", "set", "inet", "splitshield", "blocked",
                     "{ type ipv4_addr; flags interval; }"],
                    ["nft", "add", "chain", "inet", "splitshield", "input",
                     "{ type filter hook input priority -10; policy accept; }"]):  # fmt: skip
            _run(cmd)  # idempotent setup ("exists" errors are fine)
        r = _run(["nft", "list", "chain", "inet", "splitshield", "input"])
        if r and "@blocked" not in r.stdout:
            _run(["nft", "add", "rule", "inet", "splitshield", "input", "ip", "saddr", "@blocked", "drop"])
        if ":" in addr:
            return False, "IPv6 blocking is not set up on this Linux host"
        r = _run(["nft", "add", "element", "inet", "splitshield", "blocked", f"{{ {addr} }}"])
        return bool(r and r.returncode == 0), (r.stderr if r else "").strip()[:200]
    if shutil.which("iptables"):
        r = _run(["iptables", "-I", "INPUT", "-s", addr, "-j", "DROP", "-m", "comment",
                  "--comment", "SplitShield-block"])  # fmt: skip
        return bool(r and r.returncode == 0), (r.stderr if r else "").strip()[:200]
    return False, "no firewall tool found (nft / iptables)"
