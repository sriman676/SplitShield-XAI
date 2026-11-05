"""Windows / macOS / Linux support (owner 2026-10-17): background jobs for each
platform's scheduler, OS actions, memory, and reading this computer's own logs."""

from __future__ import annotations

import plistlib
from pathlib import Path
from typing import Any

import pytest

from src.soc import engine, oslive, osops, services
from utils import sysinfo


def test_memory_is_read_on_this_os() -> None:
    total, avail = sysinfo.ram_gb()
    assert 0.5 < total < 4096 and 0 <= avail <= total


@pytest.mark.parametrize("name", list(services.JOBS))
def test_launchd_plist_is_valid(name: str) -> None:
    doc = plistlib.loads(services._mac_plist(name, services.JOBS[name]).encode())
    assert doc["Label"] == f"ai.{services.unit_name(name)}"
    assert doc["ProgramArguments"][1:3] == ["-m", services.JOBS[name]["module"]]
    job = services.JOBS[name]
    if job.get("keep_running"):
        assert doc["KeepAlive"] is True
    elif job.get("daily"):
        hh, mm = map(int, job["daily"].split(":"))
        assert doc["StartCalendarInterval"] == {"Hour": hh, "Minute": mm}
    else:
        assert doc["StartInterval"] == job["every_min"] * 60


@pytest.mark.parametrize("name", list(services.JOBS))
def test_systemd_units(name: str) -> None:
    units = services._systemd_units(name, services.JOBS[name])
    svc = units[f"{services.unit_name(name)}.service"]
    assert f"-m {services.JOBS[name]['module']}" in svc and "ExecStart=" in svc
    if services.JOBS[name].get("keep_running"):
        assert "Restart=always" in svc and len(units) == 1
    else:
        timer = units[f"{services.unit_name(name)}.timer"]
        assert "OnCalendar=" in timer and "Persistent=true" in timer


def test_unit_names_are_safe() -> None:
    names = {services.unit_name(n) for n in services.JOBS}
    assert len(names) == len(services.JOBS)
    assert all(n.replace("-", "").isalnum() and n.startswith("splitshield") for n in names)


def test_blocking_without_root_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(osops, "PLATFORM", "linux")
    monkeypatch.setattr(osops.os, "geteuid", lambda: 1000, raising=False)
    ok, why = osops.block_ip("203.0.113.5")
    assert not ok and "root" in why


def test_notify_is_silent_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[Any] = []
    monkeypatch.setattr(osops.subprocess, "Popen", lambda *a, **k: called.append(a))
    osops.notify("hello")
    assert called == []


def test_local_logs_on_linux_and_macos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "auth.log"
    log.write_text("", "utf-8")
    monkeypatch.setattr(oslive, "default_config", lambda platform=None: {
        "files": [str(log)], "journald": False, "macos_unified": False, "windows_logs": [],
        "backfill": False})  # fmt: skip
    agent = oslive.make_local_agent()
    assert agent.run_once() == 0  # first pass: bookmark at the end
    with open(log, "a", encoding="utf-8") as fh:
        for i in range(12):
            fh.write(f"Oct 17 02:00:{i:02d} mac1 sshd[{i}]: Failed password for admin from "
                     f"198.51.100.33 port {41000 + i} ssh2\n")  # fmt: skip
    before = engine.event_count()
    assert agent.run_once() == 12
    assert engine.event_count() == before + 12
    assert agent.run_once() == 0  # nothing twice
