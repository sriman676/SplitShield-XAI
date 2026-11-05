"""Free threat-intel feeds: parsing, storage and lookups (owner 2026-10-15).

Feeds (all free, no sign-up, plain HTTP GET; nothing is ever sent to them):
  urlhaus   - abuse.ch URLhaus, URLs online now (malware downloads)
  feodo     - abuse.ch Feodo Tracker, botnet C2 IP blocklist
  threatfox - abuse.ch ThreatFox, recent IOCs (ip:port, domain, url)
  openphish - OpenPhish community phishing URL feed

Each feed is stored as ``artifacts/intel/<feed>.json`` with its source URL and the
date it was fetched; ``src/soc/updater.py`` refreshes them daily (when enabled) and
keeps the previous copy for rollback.

Lookups return one dict per match: source, type, value, threat, first_seen, fetched.
Private / reserved IPs never match, and a URL on a big shared host (github.com,
drive.google.com, ...) only matches as that exact URL, never as the whole domain.
"""

from __future__ import annotations

import csv
import io
import ipaddress
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from utils.paths import PROJECT_ROOT

INTEL_DIR = PROJECT_ROOT / "artifacts" / "intel"

Item = dict[str, Any]  # {"type": ip|domain|url|sha256, "value", "threat", "first_seen"}

FEEDS: dict[str, dict[str, Any]] = {
    "urlhaus": {
        "url": "https://urlhaus.abuse.ch/downloads/csv_online/",
        "name": "abuse.ch URLhaus (online URLs)",
        "min_items": 100,
    },
    "feodo": {
        "url": "https://feodotracker.abuse.ch/downloads/ipblocklist.csv",
        "name": "abuse.ch Feodo Tracker (botnet C2 IPs)",
        "min_items": 1,
    },
    "threatfox": {
        "url": "https://threatfox.abuse.ch/export/csv/recent/",
        "name": "abuse.ch ThreatFox (recent IOCs)",
        "min_items": 10,
    },
    "malwarebazaar": {
        "url": "https://bazaar.abuse.ch/export/txt/sha256/recent/",
        "name": "abuse.ch MalwareBazaar (recent malware SHA256)",
        "min_items": 10,
    },
    "openphish": {
        "url": "https://openphish.com/feed.txt",
        "name": "OpenPhish community feed",
        "min_items": 10,
    },
}

# hosts where anyone can publish a file: a bad URL there says nothing about the host
SHARED_HOSTS = frozenset(
    """github.com raw.githubusercontent.com githubusercontent.com gist.github.com
    objects.githubusercontent.com gitlab.com bitbucket.org google.com drive.google.com
    docs.google.com storage.googleapis.com googleapis.com sites.google.com
    dropbox.com dl.dropboxusercontent.com onedrive.live.com 1drv.ms sharepoint.com
    discord.com discordapp.com cdn.discordapp.com media.discordapp.net
    microsoft.com live.com office.com windows.net blob.core.windows.net
    amazonaws.com s3.amazonaws.com cloudfront.net pastebin.com paste.ee
    mediafire.com mega.nz wetransfer.com telegram.org t.me archive.org
    vercel.app netlify.app pages.dev workers.dev webflow.io weebly.com wixsite.com
    firebaseapp.com web.app herokuapp.com glitch.me blogspot.com wordpress.com
    typedream.app edgeone.dev ipfs.io cloudflare-ipfs.com""".split()
)


# --- normalisation ------------------------------------------------------------------


def norm_ip(v: str) -> str | None:
    """Canonical public IP text, else None (private / reserved / not an IP)."""
    try:
        ip = ipaddress.ip_address(str(v).strip().strip("[]"))
    except ValueError:
        return None
    if not ip.is_global:
        return None
    return str(ip)


def norm_domain(v: str) -> str | None:
    d = str(v).strip().lower().rstrip(".")
    if d.startswith("*."):
        d = d[2:]
    if not re.fullmatch(r"[a-z0-9_-]+(\.[a-z0-9_-]+)+", d) or norm_ip(d):
        return None
    return d


_HOST_RX = re.compile(r"[a-z0-9_-]+(\.[a-z0-9_-]+)+|[0-9a-f:]+:[0-9a-f:]*")


def norm_url(v: str) -> str | None:
    """Lower-case scheme and host, no fragment, no trailing '/' on an empty path."""
    s = str(v).strip()
    if "://" not in s:
        s = "http://" + s
    try:
        p = urlsplit(s)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        return None
    if p.scheme.lower() not in ("http", "https") or not _HOST_RX.fullmatch(host):
        return None
    netloc = host if port is None else f"{host}:{port}"
    path = p.path if p.path not in ("", "/") else ""
    return f"{p.scheme.lower()}://{netloc}{path}" + (f"?{p.query}" if p.query else "")


def url_host(u: str) -> str:
    try:
        return (urlsplit(u).hostname or "").lower()
    except ValueError:
        return ""


def _shared(host: str) -> bool:
    parts = host.split(".")
    return any(".".join(parts[i:]) in SHARED_HOSTS for i in range(len(parts) - 1))


# --- feed parsers -------------------------------------------------------------------


def _csv_rows(text: str, header_in_comment: bool) -> list[dict[str, str]]:
    """abuse.ch CSV: '#' comment lines; the header is either the last comment line
    or the first data line."""
    lines = text.splitlines()
    header: list[str] | None = None
    data: list[str] = []
    for ln in lines:
        if ln.startswith("#"):
            body = ln.lstrip("#").strip()
            if header_in_comment and body.count(",") >= 2:
                header = next(csv.reader([body], skipinitialspace=True))
            continue
        if ln.strip():
            data.append(ln)
    if header is None:
        if not data:
            return []
        header, data = next(csv.reader([data[0]], skipinitialspace=True)), data[1:]
    reader = csv.reader(io.StringIO("\n".join(data)), skipinitialspace=True)
    return [dict(zip(header, row, strict=False)) for row in reader if row]


def parse_urlhaus(text: str) -> list[Item]:
    out: list[Item] = []
    for r in _csv_rows(text, header_in_comment=True):
        u = norm_url(r.get("url", ""))
        if u:
            tags = r.get("tags", "")
            threat = r.get("threat", "") + (
                f" ({tags})" if tags not in ("", "None") else ""
            )
            out.append(_item("url", u, threat, r.get("dateadded", "")))
    return out


def parse_feodo(text: str) -> list[Item]:
    out: list[Item] = []
    for r in _csv_rows(text, header_in_comment=False):
        ip = norm_ip(r.get("dst_ip", ""))
        if ip:
            threat = f"botnet C2 {r.get('malware', '')} port {r.get('dst_port', '')}"
            out.append(_item("ip", ip, threat.strip(), r.get("first_seen_utc", "")))
    return out


def parse_threatfox(text: str) -> list[Item]:
    out: list[Item] = []
    for r in _csv_rows(text, header_in_comment=True):
        kind, val = r.get("ioc_type", ""), r.get("ioc_value", "")
        threat = f"{r.get('threat_type', '')} {r.get('malware_printable', '')}".strip()
        seen = r.get("first_seen_utc", "")
        if kind == "ip:port":
            ip = norm_ip(val.rsplit(":", 1)[0])
            if ip:
                out.append(_item("ip", ip, threat, seen))
        elif kind == "domain":
            d = norm_domain(val)
            if d:
                out.append(_item("domain", d, threat, seen))
        elif kind == "url":
            u = norm_url(val)
            if u:
                out.append(_item("url", u, threat, seen))
        elif kind == "sha256_hash":
            h = norm_sha256(val)
            if h:
                out.append(_item("sha256", h, threat, seen))
    return out


_SHA256 = re.compile(r"[0-9a-f]{64}")


def norm_sha256(v: str) -> str | None:
    h = str(v).strip().lower()
    return h if _SHA256.fullmatch(h) else None


def parse_malwarebazaar(text: str) -> list[Item]:
    """One SHA256 per line; '#' lines are comments."""
    out: list[Item] = []
    for line in text.splitlines():
        h = norm_sha256(line.split("#", 1)[0])
        if h:
            out.append(_item("sha256", h, "malware sample (MalwareBazaar)", ""))
    return out


def parse_openphish(text: str) -> list[Item]:
    out: list[Item] = []
    for ln in text.splitlines():
        u = norm_url(ln) if ln.strip() and not ln.startswith("#") else None
        if u:
            out.append(_item("url", u, "phishing", ""))
    return out


PARSERS: dict[str, Callable[[str], list[Item]]] = {
    "urlhaus": parse_urlhaus,
    "feodo": parse_feodo,
    "threatfox": parse_threatfox,
    "malwarebazaar": parse_malwarebazaar,
    "openphish": parse_openphish,
}


def _item(kind: str, value: str, threat: str, first_seen: str) -> Item:
    return {"type": kind, "value": value, "threat": threat, "first_seen": first_seen}


def build_feed(name: str, raw: bytes, fetched: str) -> dict[str, Any]:
    """Raw download -> the stored feed document (deduplicated items)."""
    items = PARSERS[name](raw.decode("utf-8", errors="replace"))
    seen: set[tuple[str, str]] = set()
    uniq = []
    for it in items:
        key = (it["type"], it["value"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return {
        "source": name,
        "name": FEEDS[name]["name"],
        "url": FEEDS[name]["url"],
        "fetched": fetched,
        "count": len(uniq),
        "items": uniq,
    }


# --- storage + lookups --------------------------------------------------------------


def write_feed(doc: dict[str, Any], folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{doc['source']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), "utf-8")
    tmp.replace(path)
    return path


def read_feeds(folder: Path) -> list[dict[str, Any]]:
    docs = []
    for name in FEEDS:
        p = folder / f"{name}.json"
        try:
            doc = json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("items"), list):
            docs.append(doc)
    return docs


class Index:
    """In-memory lookup tables built from stored feed documents."""

    def __init__(self, docs: Iterable[dict[str, Any]]) -> None:
        self.ip: dict[str, list[Item]] = {}
        self.domain: dict[str, list[Item]] = {}
        self.url: dict[str, list[Item]] = {}
        self.sha256: dict[str, list[Item]] = {}
        for doc in docs:
            meta = {"source": doc.get("source", "?"), "fetched": doc.get("fetched", "")}
            for it in doc.get("items", []):
                m = {**it, **meta}
                kind, val = it.get("type"), str(it.get("value", ""))
                if kind == "sha256":
                    self.sha256.setdefault(val, []).append(m)
                elif kind == "ip":
                    self.ip.setdefault(val, []).append(m)
                elif kind == "domain":
                    self.domain.setdefault(val, []).append(m)
                elif kind == "url":
                    self.url.setdefault(val, []).append(m)
                    host = url_host(val)
                    if norm_ip(host):
                        self.ip.setdefault(host, []).append(m)
                    elif host and not _shared(host):
                        self.domain.setdefault(host, []).append(m)

    def lookup_ip(self, ip: str) -> list[Item]:
        n = norm_ip(ip)
        return list(self.ip.get(n, [])) if n else []

    def lookup_domain(self, d: str) -> list[Item]:
        n = norm_domain(d)
        if not n:
            return []
        parts = n.split(".")
        out: list[Item] = []
        for i in range(len(parts) - 1):  # the domain itself, then its parents
            out += self.domain.get(".".join(parts[i:]), [])
        return out

    def lookup_url(self, u: str) -> list[Item]:
        n = norm_url(u)
        if not n:
            return []
        out = list(self.url.get(n, []))
        host = url_host(n)
        out += self.lookup_ip(host) if norm_ip(host) else self.lookup_domain(host)
        uniq: dict[tuple[str, str, str], Item] = {}
        for m in out:
            uniq.setdefault((m["source"], m["type"], m["value"]), m)
        return list(uniq.values())

    def lookup_sha256(self, h: str) -> list[Item]:
        n = norm_sha256(h)
        return list(self.sha256.get(n, [])) if n else []

    def lookup_any(self, indicator: str) -> list[Item]:
        s = indicator.strip()
        if norm_sha256(s):
            return self.lookup_sha256(s)
        if norm_ip(s):
            return self.lookup_ip(s)
        if "/" in s or "://" in s:
            return self.lookup_url(s)
        return self.lookup_domain(s)


_CACHE: dict[str, Any] = {"key": None, "index": None}


def index(folder: Path | None = None) -> Index:
    """The index of the stored feeds, rebuilt when a feed file changes."""
    folder = folder or INTEL_DIR
    key = tuple(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(folder.glob("*.json"))
        if p.stem in FEEDS
    ) if folder.is_dir() else ()  # fmt: skip
    if _CACHE["key"] != (str(folder), key) or _CACHE["index"] is None:
        _CACHE["index"] = Index(read_feeds(folder)) if key else Index([])
        _CACHE["key"] = (str(folder), key)
    idx: Index = _CACHE["index"]
    return idx


def lookup_ip(ip: str, folder: Path | None = None) -> list[Item]:
    """Matches for an IP address (Feodo, ThreatFox, URLhaus URLs on that IP)."""
    return index(folder).lookup_ip(ip)


def lookup_domain(d: str, folder: Path | None = None) -> list[Item]:
    """Matches for a domain or any parent domain of it."""
    return index(folder).lookup_domain(d)


def lookup_url(u: str, folder: Path | None = None) -> list[Item]:
    """Matches for a URL: the exact URL, or its host (IP / non-shared domain)."""
    return index(folder).lookup_url(u)


def status(folder: Path | None = None) -> dict[str, Any]:
    """Per-feed count + fetch date, for the dashboard."""
    return {
        d["source"]: {"count": d.get("count", 0), "fetched": d.get("fetched", "")}
        for d in read_feeds(folder or INTEL_DIR)
    }


# --- indicators in log text (for the benign false-alarm check) -----------------------

_URL_RX = re.compile(r"\bhttps?://[^\s\"'<>()\[\]]+", re.I)
_IP_RX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DOMAIN_RX = re.compile(
    r"\b(?:[a-z0-9-]{1,63}\.)+(?:com|net|org|io|info|biz|ru|cn|top|xyz)\b", re.I
)


def indicators(text: str) -> set[tuple[str, str]]:
    """(kind, value) pairs found in free text."""
    out: set[tuple[str, str]] = set()
    for m in _URL_RX.findall(text):
        if u := norm_url(m.rstrip(".,;")):
            out.add(("url", u))
    for m in _IP_RX.findall(text):
        if ip := norm_ip(m):
            out.add(("ip", ip))
    for m in _DOMAIN_RX.findall(text):
        if d := norm_domain(m):
            out.add(("domain", d))
    return out


def hits(idx: Index, inds: Iterable[tuple[str, str]]) -> int:
    """How many indicators match at least one feed entry."""
    look = {"ip": idx.lookup_ip, "domain": idx.lookup_domain, "url": idx.lookup_url}
    return sum(1 for kind, val in inds if look[kind](val))


def lookup_sha256(h: str, folder: Path | None = None) -> list[Item]:
    """Matches for a file's SHA256 (ThreatFox / MalwareBazaar)."""
    return index(folder).lookup_sha256(h)
