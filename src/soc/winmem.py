"""Peak memory of the current process on Windows (owner 2026-10-16).

The API needs its argument types declared: without them ctypes passes the 64-bit
process handle as a 32-bit int, the call fails silently and reports 0 MB."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


class _PMC(ctypes.Structure):
    _fields_ = [  # noqa: RUF012 - ctypes layout (PROCESS_MEMORY_COUNTERS)
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_mb() -> float | None:
    """Peak working set of this process in MB (None where it cannot be read)."""
    c = _counters()
    return c.PeakWorkingSetSize / 2**20 if c else None


def current_mb() -> float | None:
    """Current working set of this process in MB (memory-growth checks, soak test)."""
    c = _counters()
    return c.WorkingSetSize / 2**20 if c else None


def _counters() -> _PMC | None:
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c
    except (AttributeError, OSError):
        return None
