"""Cross-model checks at ingestion (owner 2026-10-15): one model's input becomes
another model's question.

For every batch of ingested events:
  * URLs in the events      -> phishing model (batch) + URL / domain threat intel;
  * source IPs              -> IP threat intel (botnet C2 lists);
  * file SHA256 (Sysmon)    -> hash threat intel (ThreatFox / MalwareBazaar);
  * executable files that exist on this PC -> malware model via LIEF features
    (``pefeatures.py``), reported as LOW confidence unless a hash-intel match or
    another model confirms it (owner decision: scanning accuracy on real files is not
    verifiable without live malware, which is never downloaded).

Each hit becomes an ordinary event (source phishing / intel / malware) carrying the
entities of the event it came from, so the risk board links it to the same user,
host or IP. Never raises: a broken check is logged and skipped.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

from utils.logging_config import get_logger

_log = get_logger(__name__)
Event = dict[str, Any]
MAX_URLS = 5000  # per batch (phishing model is batch-fast)
MAX_FILES = 50  # files scanned per batch (feature extraction reads the file)
SCAN_EXT = (".exe", ".dll", ".scr", ".sys", ".com", ".cpl")
_DERIVED = ("phishing", "intel", "malware")


def _carry(ev: Event, **kw: Any) -> Event:
    from src.soc.parsers import _event

    out = _event(ts=ev.get("ts"), src_ip=ev.get("src_ip"), **kw)
    for k in ("user", "host", "file", "sha256"):
        if ev.get(k):
            out[k] = ev[k]
    return out


_URL_CACHE: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()  # (url, model version) -> result
URL_CACHE_MAX = 50_000


def _check_cached(urls: list[str]) -> list[dict[str, Any]]:
    """phishing.check_urls with a bounded per-process memory of results: the same URL
    in many log lines (or slices of one file) is scored and explained once."""
    from src.mlareas import phishing

    ver = int(phishing.load().get("version", 0))
    todo = [u for u in urls if (u, ver) not in _URL_CACHE]
    for r in phishing.check_urls(todo) if todo else []:
        _URL_CACHE[(r["url"], ver)] = r
        if len(_URL_CACHE) > URL_CACHE_MAX:
            _URL_CACHE.popitem(last=False)
    out = []
    for u in urls:
        hit = _URL_CACHE.get((u, ver))
        if hit is not None:
            _URL_CACHE.move_to_end((u, ver))
            out.append(hit)
    return out


def _urls(events: list[Event]) -> list[Event]:
    from src.soc.riskboard import urls_in

    first: dict[str, Event] = {}
    for ev in events:
        for u in urls_in(ev):
            first.setdefault(u, ev)
    if not first:
        return []
    urls = list(first)[:MAX_URLS]
    out = []
    from src.soc import custom

    base = custom.validated("phishing")
    thr = custom.threshold("phishing", base) if base is not None else None
    for r in _check_cached(urls):
        p = float(r.get("phishing_probability") or 0)
        # the user's threshold decides when set (stricter or looser, Customise page)
        if (r["verdict"] == "phishing" and thr is None) or (thr is not None and p >= thr):
            out.append(_carry(first[r["url"]], source="phishing", kind="phishing_url",
                              url=r["url"], techniques=["T1566.002"], confidence="model",
                              detail=f"phishing model: {r['url'][:120]} "
                                     f"(p={r['phishing_probability']:.2f})"))  # fmt: skip
    return out


def _intel(events: list[Event]) -> list[Event]:
    from src.soc import intel
    from src.soc.riskboard import urls_in

    if not intel.status():
        return []  # no feeds downloaded (update agent off): nothing to compare with
    idx = intel.index()
    out = []
    seen: set[tuple[str, str]] = set()
    for ev in events:
        checks: list[tuple[str, str, str, list[Any]]] = []
        if ev.get("src_ip"):
            checks.append(("ip", ev["src_ip"], "T1071", idx.lookup_ip(ev["src_ip"])))
        for u in urls_in(ev):
            checks.append(("url", u, "T1566.002", idx.lookup_url(u)))
        if ev.get("sha256"):
            checks.append(
                ("sha256", ev["sha256"], "T1204.002", idx.lookup_sha256(ev["sha256"]))
            )
        for kind, val, tech, hits in checks:
            if hits and (kind, val) not in seen:
                seen.add((kind, val))
                src = ", ".join(sorted({h["source"] for h in hits}))
                out.append(_carry(ev, source="intel", kind=f"intel_{kind}", techniques=[tech],
                                  confidence="intel", url=val if kind == "url" else None,
                                  detail=f"threat intel ({src}): {kind} {val[:100]} - "
                                         f"{hits[0].get('threat', '')[:60]}"))  # fmt: skip
    return out


def _files(events: list[Event]) -> list[Event]:
    from src.mlareas import malware
    from src.soc import pefeatures

    if not pefeatures.available():
        return []
    paths: dict[str, Event] = {}
    for ev in events:
        f = str(ev.get("file") or "")
        if f.lower().endswith(SCAN_EXT) and os.path.isfile(f):
            paths.setdefault(f, ev)
    out = []
    for path, ev in list(paths.items())[:MAX_FILES]:
        try:
            res = malware.check_file_features(pefeatures.extract(Path(path)))
        except Exception as exc:  # noqa: BLE001 - unreadable / locked / not a PE
            _log.debug("scan skipped %s: %s", path, exc)
            continue
        if res["verdict"] in ("malicious", "suspicious"):
            out.append(_carry(ev, source="malware", kind="malware_file_scan",
                              techniques=["T1204.002"], confidence="low",
                              detail=f"malware model ({res['verdict']}, LOW confidence "
                                     f"unless confirmed): {path[:120]} "
                                     f"(p={res['malware_probability']:.2f})"))  # fmt: skip
    return out


def run(events: list[Event]) -> list[Event]:
    """Extra events from all cross-checks (never raises)."""
    base = [e for e in events if str(e.get("source", "")) not in _DERIVED]
    if not base:
        return []
    out: list[Event] = []
    for name, fn in (("urls", _urls), ("intel", _intel), ("files", _files)):
        try:
            out += fn(base)
        except Exception as exc:  # noqa: BLE001 - a cross-check must never stop ingestion
            _log.warning("cross-check %s failed: %s", name, exc)
    return out
