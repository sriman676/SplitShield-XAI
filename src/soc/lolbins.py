"""Misused system programs (LOLBAS for Windows, GTFOBins for Linux; owner 2026-10-24).

A process event whose program and command line match a documented misuse - the
program name plus every distinctive switch of that documented command (and, for a
Linux shell escape, the shell it starts) - gets that command's ATT&CK technique.
Patterns: third_party/lolbins/lolbins.json (scripts/build_lolbins.py, GPL-3.0 data,
credited). Missing file -> no matches (the rest of detection is unchanged).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from utils.paths import PROJECT_ROOT

PATTERNS = PROJECT_ROOT / "third_party" / "lolbins" / "lolbins.json"
_TOKEN = re.compile(r"[^\s\"'`,;()]+")


@lru_cache(maxsize=1)
def _index() -> dict[tuple[str, str], list[dict[str, Any]]]:
    try:
        pats = json.loads(PATTERNS.read_text("utf-8"))["patterns"]
    except (OSError, ValueError, KeyError):
        return {}
    idx: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in pats:
        b = str(p["binary"]).lower()
        if p["os"] == "windows":
            b = b.removesuffix(".exe")
        idx.setdefault((p["os"], b), []).append(p)
    return idx


def _basename(image: str) -> str:
    return re.split(r"[\\/]", image.strip().strip("\"'"))[-1].lower()


def _win_tokens(cmd: str) -> set[str]:
    out = set()
    for t in _TOKEN.findall(cmd.lower()):
        if t[:1] in "-/":
            core = re.split(r"[:=]", t[1:], maxsplit=1)[0]
            out |= {"-" + core, "/" + core}  # Windows programs accept either prefix
        out.add(t)
    return out


def match(image: str, cmd: str, os_name: str) -> list[dict[str, Any]]:
    """Documented misuses this process start matches (empty for normal use)."""
    if not image and not cmd:
        return []
    name = _basename(image or cmd.split(" ", 1)[0])
    if os_name == "windows":
        name = name.removesuffix(".exe")
    pats = _index().get((os_name, name))
    if not pats:
        return []
    low = cmd.lower()
    toks = _win_tokens(low) if os_name == "windows" else {t.split("=", 1)[0] for t in _TOKEN.findall(low)}
    out = []
    for p in pats:
        if not all(s in toks for s in p["switches"]):
            continue
        if p.get("shell") and p["shell"] not in low:
            continue
        out.append(p)
    return out


def techniques(image: str, cmd: str, os_name: str) -> list[str]:
    seen: list[str] = []
    for p in match(image, cmd, os_name):
        if p["technique"] not in seen:
            seen.append(p["technique"])
    return seen
