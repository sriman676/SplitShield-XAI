"""Memory of this computer on Windows, Linux and macOS (standard library only).

``ram_gb()`` -> (total GB, available GB). Used for scheduling decisions only (how
many jobs or workers fit); it never changes what is computed. Unknown -> (16, 8).
"""

from __future__ import annotations

import re
import subprocess
import sys

GB = 2**30


def _windows() -> tuple[float, float]:
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong),  # noqa: RUF012 - ctypes layout
                    ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                    ("ae", ctypes.c_ulonglong)]  # fmt: skip

    m = MS()
    m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.total / GB, m.avail / GB


def _linux() -> tuple[float, float]:
    info: dict[str, int] = {}
    with open("/proc/meminfo", encoding="ascii") as fh:
        for ln in fh:
            k, _, v = ln.partition(":")
            info[k] = int(v.split()[0]) * 1024  # kB
    avail = info.get("MemAvailable", info.get("MemFree", 0) + info.get("Cached", 0))
    return info["MemTotal"] / GB, avail / GB


def _macos() -> tuple[float, float]:
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                               timeout=10).stdout.strip())  # fmt: skip
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    page = int(m.group(1)) if (m := re.search(r"page size of (\d+) bytes", vm)) else 4096
    pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in re.findall(r"^(Pages [^:]+):\s+(\d+\.?)$", vm, re.M)}
    free = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return total / GB, free * page / GB


def ram_gb() -> tuple[float, float]:
    try:
        if sys.platform == "win32":
            return _windows()
        if sys.platform == "darwin":
            return _macos()
        return _linux()
    except (OSError, ValueError, KeyError, AttributeError, subprocess.SubprocessError):
        return 16.0, 8.0


def free_ram_gb() -> float:
    return ram_gb()[1]
