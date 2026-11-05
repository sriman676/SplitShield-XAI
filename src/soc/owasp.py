"""OWASP Top 10 (owner 2026-10-23): web alerts get their OWASP category, and the list
itself is refreshed by the update agent once a month.

Source: github.com/OWASP/Top10 (CC BY-SA). The newest edition is found from its
``<year>/mkdocs.yml`` navigation; each category page gives its title and its "List of
Mapped CWEs". Alerts are labelled through CWE numbers (an SQL injection is CWE-89), so
the labels follow a new edition automatically (2021 A03 Injection, 2025 A05 Injection).

    python -m src.soc.owasp --build     # fetch the newest edition into src/knowledge
    python -m src.soc.owasp --show
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT

RAW = "https://raw.githubusercontent.com/OWASP/Top10/master"
KB = PROJECT_ROOT / "src" / "knowledge" / "owasp_top10.json"
FIRST_EDITION = 2021  # the CWE-mapped editions
USER_AGENT = "SplitShield-XAI-update-agent"

# attack class -> CWE (CRS tags "attack-sqli", Sigma / own rule wording)
ATTACK_CWE: dict[str, int] = {
    "sqli": 89, "sql injection": 89, "xss": 79, "cross-site scripting": 79,
    "rce": 78, "command injection": 78, "os command": 78, "code injection": 94,
    "php injection": 94, "lfi": 22, "path traversal": 22, "directory traversal": 22,
    "rfi": 98, "remote file inclusion": 98, "xxe": 611, "ssrf": 918,
    "deserialization": 502, "java": 502, "session fixation": 384, "ldap injection": 90,
    "header injection": 113, "crlf": 93, "protocol": 444, "request smuggling": 444,
    "multipart": 434, "file upload": 434, "webshell": 434, "web shell": 434, "brute force": 307, "credential stuffing": 307,
    "generic": 20, "nodejs": 94, "template injection": 1336,
}  # fmt: skip


def _get(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https host
        return str(r.read(2_000_000).decode("utf-8", "replace"))


_NAV = re.compile(r"-\s*(A\d\d)\s+([^:]+):\s*'(en/A\d\d_\d{4}-[^']+\.md)'")


def latest_edition(get: Callable[[str], str] = _get, now_year: int | None = None) -> tuple[int, list[tuple[str, str, str]]]:
    """(year, [(id, name, page)]) of the newest edition that has ten categories."""
    year = now_year or time.gmtime().tm_year
    for y in range(year, FIRST_EDITION - 1, -1):
        try:
            nav = _NAV.findall(get(f"{RAW}/{y}/mkdocs.yml"))
        except OSError:
            continue
        seen = list(dict.fromkeys(nav))
        if len({a for a, _, _ in seen}) == 10:
            return y, seen
    raise ValueError("no OWASP Top 10 edition with ten categories found")


def parse_page(text: str) -> list[int]:
    """CWE numbers of a category page's "List of Mapped CWEs"."""
    part = text.split("List of Mapped CWEs", 1)[-1] if "List of Mapped CWEs" in text else ""
    return sorted({int(n) for n in re.findall(r"CWE-(\d+)", part)})


def build(get: Callable[[str], str] = _get, now_year: int | None = None) -> dict[str, Any]:
    year, nav = latest_edition(get, now_year)
    cats = []
    for aid, name, page in nav:
        cats.append({"id": f"{aid}:{year}", "name": name.strip(), "cwes": parse_page(get(f"{RAW}/{year}/docs/{page}")),
                     "url": f"https://owasp.org/Top10/{year}/{page.removeprefix('en/').removesuffix('.md')}/"})  # fmt: skip
    return {"source": "OWASP Top 10 (github.com/OWASP/Top10, CC BY-SA 4.0)", "edition": year, "categories": cats}


def validate(kb: dict[str, Any]) -> list[str]:
    """Problems that make a fetched list unusable (an empty list means it is fine)."""
    cats = kb.get("categories", [])
    out = []
    if len(cats) != 10:
        out.append(f"{len(cats)} categories, expected 10")
    if sum(1 for c in cats if c.get("cwes")) < 8:
        out.append("fewer than 8 categories list CWEs")
    if not any(89 in c.get("cwes", []) for c in cats):
        out.append("no category holds SQL injection (CWE-89)")
    return out


def save(kb: dict[str, Any], path: Path = KB) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kb, indent=1), "utf-8")
    load.cache_clear()


@lru_cache(maxsize=1)
def load(path: Path = KB) -> dict[str, Any]:
    try:
        return dict(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError):
        return {"edition": None, "categories": []}


def category_for_cwe(cwe: int) -> dict[str, Any] | None:
    for c in load().get("categories", []):
        if cwe in c.get("cwes", []):
            return dict(c)
    return None


def category_for(text: str) -> dict[str, Any] | None:
    """OWASP category of an alert, from its attack class ("attack-sqli") or wording."""
    low = text.lower().replace("attack-", "")
    for key in sorted(ATTACK_CWE, key=len, reverse=True):  # most specific wording first
        if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", low):
            return category_for_cwe(ATTACK_CWE[key])
    return None


def label(ev: dict[str, Any]) -> dict[str, Any]:
    """Adds ``owasp`` ("A05:2025 Injection") to a web alert that maps to a category."""
    if ev.get("owasp") or not ev.get("techniques"):
        return ev
    if ev.get("kind") not in ("http_request", "web_attack") and "T1190" not in ev.get("techniques", []):
        return ev
    c = category_for(" ".join(str(ev.get(k, "")) for k in ("rule", "title", "kind", "detail", "crs_tags")))
    if c:
        ev["owasp"] = f"{c['id']} {c['name']}"
    return ev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.build:
        kb = build()
        bad = validate(kb)
        if bad:
            raise SystemExit("not saved: " + "; ".join(bad))
        save(kb)
    kb = load()
    print(f"OWASP Top 10 {kb.get('edition')}: " + ", ".join(f"{c['id']} {c['name']} ({len(c['cwes'])} CWEs)"
                                                          for c in kb.get("categories", [])))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
