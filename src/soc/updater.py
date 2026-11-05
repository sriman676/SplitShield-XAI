"""Internet update agent (owner decisions 2026-10-15).

OFF by default. When switched on (``set_enabled(True)``, stored in
``artifacts/soc/update_agent.json``) a daily scheduled task runs ``run_once()``,
which may update ONLY:

  1. SigmaHQ rules - the latest release's ``sigma_all_rules.zip`` (owner 2026-10-24: the
     full bundle incl. emerging threats; was core+), tag kept
  2. MITRE ATT&CK Enterprise - ``src/knowledge/attack_enterprise.json``, rebuilt from
     the latest mitre/cti release with ``build_attack_kb`` (same format as before)
  3. Free threat-intel feeds - URLhaus, Feodo Tracker, ThreatFox, OpenPhish
     (``src/soc/intel.py``, stored under ``artifacts/intel/``)

Never: Python packages, uploads, or anything but plain HTTP GETs (no account, no
e-mail, no local data leaves the machine).

Safety gate, per update:
  * download into ``artifacts/updates/staging/`` and validate (parses, counts sane);
  * false-alarm check on a fixed benign sample (``artifacts/intel/benign_sample.jsonl``):
    alerts (Sigma) or intel hits (feeds) BEFORE vs AFTER - must not rise by more than
    0.1 percentage points;
  * the project's tests (``pytest -q -p no:warnings``) on a sandbox copy of the
    project with the staged files in place;
  * only then apply; the replaced version is kept in ``artifacts/updates/previous/``
    (``rollback(source)`` restores it). Every rejection is recorded with its reason.

Sigma's fair-test exclusions (``src/soc/sigma.py``) are applied by the rule loader
itself, so they hold for new rules too; validation records how many were excluded.
Network errors never raise: they are logged and the source is retried next run.
"""

from __future__ import annotations

import calendar
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.soc import intel
from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)

SOURCES = ("sigma", "attack", "owasp_top10", "owasp_crs", *intel.FEEDS)
# owner 2026-10-23: detection content is checked once a month, threat-intel feeds daily
# (bad IPs / URLs go stale within days)
CONTENT_SOURCES = ("sigma", "attack", "owasp_top10", "owasp_crs")
CRS_REPO = "coreruleset/coreruleset"
CRS_MIN_RULES = 60  # usable paranoia-level-1 rules on access logs (v4.30: 106)
CONTENT_EVERY_DAYS = 30
MAX_RISE_PP = 0.1  # benign alert-rate rise allowed, percentage points
SIGMA_MIN_FILES = 500
ATTACK_MIN_TECHNIQUES = 400
HISTORY_KEEP = 90
LOCK_STALE_S = 4 * 3600
TEST_TIMEOUT_S = 1800
USER_AGENT = "SplitShield-XAI-update-agent"
_GH = "https://api." + "github.com/repos"
SIGMA_API = f"{_GH}/SigmaHQ/sigma/releases/latest"
SIGMA_ASSET = "sigma_all_rules.zip"  # owner 2026-10-24 (was sigma_core+.zip)
ATTACK_API = f"{_GH}/mitre/cti/releases/latest"
ATTACK_ASSET = "enterprise-attack.json"
ATTACK_SOURCE = (
    "MITRE ATT&CK Enterprise (github.com/mitre/cti), (c) The MITRE Corporation"
)

# well-known benign indicators: a feed that flags one of these is a false alarm
BENIGN_INDICATORS: tuple[tuple[str, str], ...] = (
    ("ip", "8.8.8.8"), ("ip", "8.8.4.4"), ("ip", "1.1.1.1"), ("ip", "1.0.0.1"),
    ("ip", "9.9.9.9"), ("ip", "208.67.222.222"),
    ("domain", "google.com"), ("domain", "www.google.com"), ("domain", "microsoft.com"),
    ("domain", "windowsupdate.com"), ("domain", "update.microsoft.com"),
    ("domain", "github.com"), ("domain", "raw.githubusercontent.com"),
    ("domain", "pypi.org"), ("domain", "python.org"), ("domain", "wikipedia.org"),
    ("domain", "amazon.com"), ("domain", "apple.com"), ("domain", "cloudflare.com"),
    ("domain", "mozilla.org"), ("domain", "ubuntu.com"), ("domain", "debian.org"),
    ("domain", "office.com"), ("domain", "live.com"), ("domain", "youtube.com"),
    ("domain", "linkedin.com"), ("domain", "zoom.us"), ("domain", "slack.com"),
    ("url", "https://www.google.com/"), ("url", "https://github.com/"),
    ("url", "https://pypi.org/simple/"), ("url", "https://www.microsoft.com/"),
    ("url", "https://attack.mitre.org/"), ("url", "https://en.wikipedia.org/"),
)  # fmt: skip


# --- layout + settings --------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """``root``: the project whose current files are compared and tested;
    ``out``: where updates are applied (== root, except for a dry run)."""

    root: Path = PROJECT_ROOT
    out: Path = PROJECT_ROOT

    def sigma_dir(self, base: Path | None = None) -> Path:
        return (base or self.out) / "third_party" / "sigma"

    def attack_json(self, base: Path | None = None) -> Path:
        return (base or self.out) / "src" / "knowledge" / "attack_enterprise.json"

    def owasp_json(self, base: Path | None = None) -> Path:
        return (base or self.out) / "src" / "knowledge" / "owasp_top10.json"

    def crs_dir(self, base: Path | None = None) -> Path:
        """OWASP CRS; staging and the previous copy live inside it too, because Windows
        Defender flags CRS files and only this folder is excluded (docs/INSTALL.md)."""
        return (base or self.out) / "third_party" / "coreruleset"

    def intel_dir(self, base: Path | None = None) -> Path:
        return (base or self.out) / "artifacts" / "intel"

    @property
    def work(self) -> Path:
        return self.out / "artifacts" / "updates"

    @property
    def staging(self) -> Path:
        return self.work / "staging"

    @property
    def previous(self) -> Path:
        return self.work / "previous"

    @property
    def state(self) -> Path:
        return self.out / "artifacts" / "soc" / "update_agent.json"

    @property
    def benign(self) -> Path:
        return self.out / "artifacts" / "intel" / "benign_sample.jsonl"


def _default_state() -> dict[str, Any]:
    return {"enabled": True, "last_run": None, "versions": {}, "history": []}


def load_state(path: Path | None = None) -> dict[str, Any]:
    p = path or Layout().state
    try:
        st = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return _default_state()
    return {**_default_state(), **st} if isinstance(st, dict) else _default_state()


def save_state(st: dict[str, Any], path: Path | None = None) -> None:
    p = path or Layout().state
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str), "utf-8")
    tmp.replace(p)


def get_enabled(path: Path | None = None) -> bool:
    """Whether the update agent is switched on (default ON since 2026-10-25, owner decision)."""
    return bool(load_state(path).get("enabled", True))


def set_enabled(on: bool, path: Path | None = None) -> None:
    """Switch the daily update agent on/off (the dashboard toggle)."""
    st = load_state(path)
    st["enabled"] = bool(on)
    st.setdefault("history", []).append(
        {"ts": _now(), "action": "enabled" if on else "disabled"}
    )
    st["history"] = st["history"][-HISTORY_KEEP:]
    save_state(st, path)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- downloads ----------------------------------------------------------------------


def http_get(url: str, timeout: float = 120.0, max_bytes: int = 400_000_000) -> bytes:
    """Plain GET (only a User-Agent header; no cookies, no credentials)."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (https only)
        data: bytes = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"download larger than {max_bytes} bytes")
    return data


FETCH: Callable[[str], bytes] = http_get  # tests replace this
NET_ERRORS = (urllib.error.URLError, TimeoutError, OSError, ValueError)


def _latest_release(api: str, asset: str) -> tuple[str, str]:
    rel = json.loads(FETCH(api))
    url = next(
        (
            a["browser_download_url"]
            for a in rel.get("assets", [])
            if a["name"] == asset
        ),
        None,
    )
    if not url:
        raise ValueError(f"release {rel.get('tag_name')} has no asset {asset}")
    return str(rel["tag_name"]), str(url)


# --- candidates ---------------------------------------------------------------------


@dataclass
class Candidate:
    source: str
    status: str = "pending"  # skipped | error | rejected | passed | applied | dry_run
    old_version: str = ""
    new_version: str = ""
    staged: Path | None = None
    checks: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "checks": self.checks,
            "reason": self.reason,
        }


# --- Sigma --------------------------------------------------------------------------


def current_sigma_tag(lay: Layout, st: dict[str, Any]) -> str:
    if tag := st.get("versions", {}).get("sigma"):
        return str(tag)
    zips = sorted(lay.sigma_dir(lay.root).glob("sigma_*_r*.zip"), key=lambda p: p.name.rsplit("_", 1)[-1])
    m = re.search(r"_(r[\d-]+)\.zip$", zips[-1].name) if zips else None
    return m.group(1) if m else ""


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> int:
    dest = dest.resolve()
    n = 0
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if dest not in target.parents and target != dest:
            raise ValueError(f"unsafe path in archive: {info.filename}")
        if not (info.is_dir() or info.filename.endswith((".yml", ".yaml", ".txt"))):
            continue  # rules + version.txt only
        zf.extract(info, dest)
        n += 1
    return n


def place_emerging(extracted: Path) -> None:
    """The full bundle keeps emerging-threat rules next to rules/; the engine reads one
    root (rules/rules, with windows/ inside), so they move in as rules/emerging-threats."""
    src, dst = extracted / "rules-emerging-threats", extracted / "rules" / "emerging-threats"
    if src.is_dir() and (extracted / "rules").is_dir() and not dst.exists():
        shutil.move(str(src), str(dst))


def _sigma_stats(rules_root: Path) -> dict[str, Any]:
    from src.soc import sigma

    stats = dict(sigma.load(str(rules_root))["stats"])
    stats.pop("by_product", None)
    return stats


def stage_sigma(lay: Layout, st: dict[str, Any], force: bool) -> Candidate:
    c = Candidate("sigma", old_version=current_sigma_tag(lay, st))
    tag, url = _latest_release(SIGMA_API, SIGMA_ASSET)
    c.new_version = tag
    if tag == c.old_version and not force:
        c.status, c.reason = "skipped", "already the latest release"
        return c
    stage = lay.staging / "sigma"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "rules").mkdir(parents=True)
    raw = FETCH(url)
    (stage / f"sigma_all_rules_{tag}.zip").write_bytes(raw)
    with zipfile.ZipFile(stage / f"sigma_all_rules_{tag}.zip") as zf:
        _safe_extract(zf, stage / "rules")
    place_emerging(stage / "rules")
    c.staged = stage
    # validation: parses, counts sane against the current rule set
    new_root = stage / "rules" / "rules"
    if not new_root.is_dir():
        c.status, c.reason = "rejected", "archive has no rules/ folder"
        return c
    new = _sigma_stats(new_root)
    old_root = lay.sigma_dir(lay.root) / "rules" / "rules"
    old = _sigma_stats(old_root) if old_root.is_dir() else {"loaded": 0, "files": 0}
    c.checks["validate"] = {"new": new, "old": old}
    problems = []
    if new["files"] < SIGMA_MIN_FILES:
        problems.append(f"only {new['files']} rule files")
    if new["loaded"] < 0.8 * old.get("loaded", 0):
        problems.append(f"usable rules fell {old.get('loaded')} -> {new['loaded']}")
    if old.get("excluded_fair_test", 0) and not new.get("excluded_fair_test"):
        problems.append("fair-test exclusions no longer apply")
    if problems:
        c.status, c.reason = "rejected", "; ".join(problems)
    return c


@contextlib.contextmanager
def sigma_rules_at(rules_root: Path) -> Iterator[None]:
    """Make src/soc/sigma.py use another rules folder (for before/after checks)."""
    from src.soc import sigma

    saved = (sigma.RULES_ROOT, sigma.RULES_DIR, sigma.load)
    orig_load = sigma.load

    def _load(rules_dir: str = str(rules_root)) -> dict[str, Any]:
        return orig_load(rules_dir)

    sigma.RULES_ROOT, sigma.RULES_DIR = rules_root, rules_root / "windows"
    sigma.load = _load  # type: ignore[assignment]
    try:
        yield
    finally:
        sigma.RULES_ROOT, sigma.RULES_DIR, sigma.load = saved


def benign_alerts(lines: list[str], rules_root: Path) -> tuple[int, int]:
    """(alerts, events) on the benign sample with the given Sigma rules."""
    from src.soc.parsers import parse_line

    alerts = n = 0
    with sigma_rules_at(rules_root):
        for ln in lines:
            try:
                ev = parse_line(ln)
            except Exception:  # noqa: BLE001 - a parser bug must not stop the gate
                continue
            if ev is None:
                continue
            n += 1
            alerts += bool(ev.get("techniques"))
    return alerts, n


def _rate_check(before: int, after: int, n: int) -> dict[str, Any]:
    rb = 100.0 * before / max(n, 1)
    ra = 100.0 * after / max(n, 1)
    return {
        "events": n,
        "before": before,
        "after": after,
        "rate_before_pct": round(rb, 4),
        "rate_after_pct": round(ra, 4),
        "rise_pp": round(ra - rb, 4),
        "ok": ra - rb <= MAX_RISE_PP,
    }


def benign_check_sigma(lay: Layout, c: Candidate, lines: list[str]) -> None:
    assert c.staged is not None
    old_root = lay.sigma_dir(lay.root) / "rules" / "rules"
    before, n = benign_alerts(lines, old_root)
    after, _ = benign_alerts(lines, c.staged / "rules" / "rules")
    chk = _rate_check(before, after, n)
    c.checks["false_alarm"] = chk
    if not chk["ok"]:
        c.status = "rejected"
        c.reason = f"benign alert rate rose {chk['rise_pp']} pp (> {MAX_RISE_PP})"


# --- ATT&CK -------------------------------------------------------------------------


def _clean(text: str) -> str:
    text = re.sub(r"\(Citation:[^)]*\)", "", text or "")
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"<code>|</code>", "`", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _ext_id(o: dict[str, Any]) -> str | None:
    return next(
        (
            r["external_id"]
            for r in o.get("external_references", [])
            if r.get("source_name") == "mitre-attack"
        ),
        None,
    )


def build_attack_kb(bundle: dict[str, Any], version: str = "") -> dict[str, Any]:
    """MITRE enterprise-attack STIX bundle -> the knowledge JSON format of
    ``src/knowledge/attack_enterprise.json`` (name, tactics, description, detection,
    detection_notes from x-mitre-detection-strategy -> x-mitre-analytic,
    mitigations, platforms, url)."""
    objs: list[dict[str, Any]] = bundle["objects"]
    by_id = {o["id"]: o for o in objs}
    tech: dict[str, str] = {}
    for o in objs:
        if (
            o.get("type") != "attack-pattern"
            or o.get("revoked")
            or o.get("x_mitre_deprecated")
        ):
            continue
        tid = _ext_id(o)
        if tid:
            tech[o["id"]] = tid
    mitig: dict[str, list[str]] = {}
    detect: dict[str, list[str]] = {}
    for o in objs:
        if (
            o.get("type") != "relationship"
            or o.get("revoked")
            or o.get("x_mitre_deprecated")
        ):
            continue
        tgt = tech.get(o.get("target_ref", ""))
        src = by_id.get(o.get("source_ref", ""), {})
        if not tgt:
            continue
        rel = o.get("relationship_type")
        if rel == "mitigates" and src.get("type") == "course-of-action":
            note = _clean(o.get("description", "")) or _clean(
                src.get("description", "")
            )
            mitig.setdefault(tgt, []).append(f"{src.get('name')}: {note[:300]}")
        if rel == "detects":
            if src.get("type") == "x-mitre-detection-strategy":
                for ref in src.get("x_mitre_analytic_refs", [])[:4]:
                    an = by_id.get(ref, {})
                    plats = "/".join(an.get("x_mitre_platforms", []))
                    logs = ", ".join(
                        sorted(
                            {
                                x.get("name", "")
                                for x in an.get("x_mitre_log_source_references", [])
                            }
                        )
                    )[:120]
                    desc = _clean(an.get("description", ""))
                    if desc:
                        detect.setdefault(tgt, []).append(
                            f"[{plats}] {desc[:350]}"
                            + (f" (logs: {logs})" if logs else "")
                        )
            else:
                note = _clean(o.get("description", ""))
                if note:
                    detect.setdefault(tgt, []).append(note[:300])
    out: dict[str, dict[str, Any]] = {}
    for sid, tid in tech.items():
        o = by_id[sid]
        out[tid] = {
            "name": o["name"],
            "tactics": [
                p["phase_name"]
                for p in o.get("kill_chain_phases", [])
                if p.get("kill_chain_name") == "mitre-attack"
            ],
            "description": _clean(o.get("description", ""))[:2500],
            "detection": _clean(o.get("x_mitre_detection", ""))[:1500],
            "detection_notes": detect.get(tid, [])[:6],
            "mitigations": mitig.get(tid, [])[:8],
            "platforms": o.get("x_mitre_platforms", []),
            "url": f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/",
        }
    ver = version or next(
        (
            str(o.get("x_mitre_version") or "")
            for o in objs
            if o.get("type") == "x-mitre-collection"
        ),
        "",
    )
    return {
        "source": ATTACK_SOURCE,
        "version": ver,
        "tactics": attack_tactics(objs),
        "techniques": dict(sorted(out.items())),
    }


def attack_tactics(objs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The Enterprise matrix's tactics in ATT&CK's own order: shortname, name, TA id
    (2026-10-23: the incident builder takes its tactic list from here, so a new ATT&CK
    tactic is known after the next monthly update instead of waiting for a code change)."""
    tac = {o["id"]: o for o in objs if o.get("type") == "x-mitre-tactic"
           and not o.get("revoked") and not o.get("x_mitre_deprecated")}  # fmt: skip
    for m in objs:
        if m.get("type") == "x-mitre-matrix" and not m.get("revoked"):
            return [{"shortname": tac[r].get("x_mitre_shortname", ""), "name": tac[r].get("name", ""),
                     "id": _ext_id(tac[r]) or ""} for r in m.get("tactic_refs", []) if r in tac]  # fmt: skip
    return []


def new_tactics(kb: dict[str, Any]) -> list[str]:
    """Tactics in the knowledge base that the incident builder has no severity weight for."""
    from src.soc.correlate import TACTIC_WEIGHT

    return [t["shortname"] for t in kb.get("tactics", []) if t.get("shortname") not in TACTIC_WEIGHT]


def write_attack_kb(kb: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(kb, fh, indent=0)
    tmp.replace(path)


def _kb_summary(kb: dict[str, Any]) -> dict[str, Any]:
    t = kb.get("techniques", {})
    n = max(len(t), 1)
    return {
        "techniques": len(t),
        "with_detection": round(
            sum(1 for v in t.values() if v.get("detection") or v.get("detection_notes"))
            / n,
            3,
        ),
        "with_mitigations": round(
            sum(1 for v in t.values() if v.get("mitigations")) / n, 3
        ),
    }


KB_KEYS = ("name", "tactics", "description", "detection", "detection_notes",
           "mitigations", "platforms", "url")  # fmt: skip


def stage_attack(lay: Layout, st: dict[str, Any], force: bool) -> Candidate:
    path = lay.attack_json(lay.root)
    try:
        old_kb = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        old_kb = {"version": "", "techniques": {}}
    c = Candidate(
        "attack",
        old_version=str(
            st.get("versions", {}).get("attack") or old_kb.get("version", "")
        ),
    )
    tag, url = _latest_release(ATTACK_API, ATTACK_ASSET)
    version = tag.split("-v")[-1].lstrip("v")
    c.new_version = version
    if version == c.old_version and not force:
        c.status, c.reason = "skipped", "already the latest release"
        return c
    bundle = json.loads(FETCH(url))
    kb = build_attack_kb(bundle, version)
    stage = lay.staging / "attack"
    shutil.rmtree(stage, ignore_errors=True)
    write_attack_kb(kb, stage / "attack_enterprise.json")
    c.staged = stage
    new, old = _kb_summary(kb), _kb_summary(old_kb)
    c.checks["validate"] = {"new": new, "old": old}
    problems = []
    if new["techniques"] < ATTACK_MIN_TECHNIQUES:
        problems.append(f"only {new['techniques']} techniques")
    if new["techniques"] < 0.9 * old["techniques"]:
        problems.append(f"techniques fell {old['techniques']} -> {new['techniques']}")
    if new["with_detection"] < 0.9 * old["with_detection"]:
        problems.append("detection guidance coverage fell")
    if new["with_mitigations"] < 0.9 * old["with_mitigations"]:
        problems.append("mitigation coverage fell")
    bad = [
        t for t, v in kb["techniques"].items() if set(KB_KEYS) - set(v) or not v["name"]
    ]
    if bad:
        problems.append(f"{len(bad)} malformed entries")
    if not kb.get("tactics"):
        problems.append("no tactic list (Enterprise matrix missing)")
    c.checks["new_tactics"] = new_tactics(kb)
    if problems:
        c.status, c.reason = "rejected", "; ".join(problems)
    # the knowledge base explains alerts; it does not raise them
    c.checks["false_alarm"] = {"ok": True, "note": "knowledge base raises no alerts"}
    return c


# --- OWASP Top 10 -------------------------------------------------------------------


def stage_owasp_top10(lay: Layout, st: dict[str, Any], force: bool) -> Candidate:
    """The newest OWASP Top 10 edition (src/soc/owasp.py) -> staging, validated."""
    from src.soc import owasp

    path = lay.owasp_json(lay.root)
    try:
        old = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        old = {"edition": None, "categories": []}
    c = Candidate("owasp_top10", old_version=str(st.get("versions", {}).get("owasp_top10") or old.get("edition") or ""))
    kb = owasp.build(lambda url: FETCH(url).decode("utf-8", "replace"))
    c.new_version = str(kb["edition"])
    same = kb["categories"] == old.get("categories")
    if same and not force:
        c.status, c.reason = "skipped", "already the newest edition"
        return c
    stage = lay.staging / "owasp_top10"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "owasp_top10.json").write_text(json.dumps(kb, indent=1), "utf-8")
    c.staged = stage
    problems = owasp.validate(kb)
    c.checks["validate"] = {"edition": kb["edition"], "categories": len(kb["categories"]), "problems": problems}
    if problems:
        c.status, c.reason = "rejected", "; ".join(problems)
    c.checks["false_alarm"] = {"ok": True, "note": "the list labels alerts; it does not raise them"}
    return c


# --- OWASP Core Rule Set -------------------------------------------------------------


def crs_latest_tag() -> str:
    """The latest CRS release tag, from the release page's redirect (no API call)."""
    req = urllib.request.Request(f"https://github.com/{CRS_REPO}/releases/latest",
                                 headers={"User-Agent": USER_AGENT})  # fmt: skip
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https URL
        return str(r.geturl()).rstrip("/").rsplit("/", 1)[-1]


CRS_TAG: Callable[[], str] = crs_latest_tag  # tests replace this


def stage_owasp_crs(lay: Layout, st: dict[str, Any], force: bool) -> Candidate:
    """Request-side CRS rules (+ their .data lists) of the latest release, file by file."""
    from src.soc import crs

    cur = lay.crs_dir(lay.root)
    old = (cur / "VERSION").read_text("utf-8").strip() if (cur / "VERSION").is_file() else ""
    c = Candidate("owasp_crs", old_version=str(st.get("versions", {}).get("owasp_crs") or old))
    tag = CRS_TAG()
    c.new_version = tag
    if tag == c.old_version and not force:
        c.status, c.reason = "skipped", "already the latest release"
        return c
    tree = json.loads(FETCH(f"{_GH}/{CRS_REPO}/git/trees/{tag}?recursive=1"))["tree"]
    want = [t["path"] for t in tree if t.get("type") == "blob" and t["path"].startswith("rules/")
            and (t["path"].endswith(".data") or re.match(r"rules/REQUEST-9(1[1-9]|[2-4]\d)-.*\.conf$", t["path"]))]  # fmt: skip
    stage = lay.crs_dir() / f".staging-{tag}"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "rules").mkdir(parents=True, exist_ok=True)
    for p in want:
        (stage / p).write_bytes(FETCH(f"https://raw.githubusercontent.com/{CRS_REPO}/{tag}/{p}"))
    (stage / "VERSION").write_text(tag, "utf-8")
    c.staged = stage
    present = [p for p in want if (stage / p).is_file()]
    rules, skipped = crs.parse_rules(stage / "rules")
    c.checks["validate"] = {"files": len(want), "present": len(present), "usable_rules": len(rules), "skipped": skipped}
    problems = []
    missing = [p for p in want if p not in present and p.endswith(".conf")]
    if missing:
        problems.append(f"{len(missing)} rule files missing (removed by antivirus? exclude {lay.crs_dir()})")
    if len(rules) < CRS_MIN_RULES:
        problems.append(f"only {len(rules)} usable rules")
    if problems:
        c.status, c.reason = "rejected", "; ".join(problems)
    return c


def benign_check_crs(lay: Layout, c: Candidate, lines: list[str]) -> None:
    """Web lines of the fixed benign sample: CRS attacks before vs after the update."""
    from src.soc import crs, logformats

    assert c.staged is not None
    reqs = []
    for ln in lines:
        m = logformats._ACCESS.match(ln)
        if m:
            reqs.append(crs.request(m["method"], m["target"], m["proto"] or "", m["ua"] or "", m["ref"] or ""))
    old_dir = lay.crs_dir(lay.root) / "rules"
    before = sum(crs.check(r, old_dir)["attack"] for r in reqs) if crs.available(old_dir) else 0
    after = sum(crs.check(r, c.staged / "rules")["attack"] for r in reqs)
    chk = _rate_check(before, after, max(1, len(reqs)))
    c.checks["false_alarm"] = chk | {"web_lines": len(reqs)}
    if not chk["ok"]:
        c.status = "rejected"
        c.reason = f"benign alert rate rose {chk['rise_pp']} pp (> {MAX_RISE_PP})"


# --- threat intel -------------------------------------------------------------------


def stage_feed(lay: Layout, name: str) -> Candidate:
    old = next(
        (d for d in intel.read_feeds(lay.intel_dir(lay.root)) if d["source"] == name),
        None,
    )
    c = Candidate(name, old_version=str(old.get("fetched", "")) if old else "")
    try:
        raw = FETCH(intel.FEEDS[name]["url"])
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            c.status, c.reason = "skipped", f"feed needs a key (HTTP {e.code})"
            return c
        raise
    doc = intel.build_feed(name, raw, _now())
    c.new_version = doc["fetched"]
    stage = lay.staging / "intel"
    intel.write_feed(doc, stage)
    c.staged = stage / f"{name}.json"
    old_n = int(old.get("count", 0)) if old else 0
    c.checks["validate"] = {"new": doc["count"], "old": old_n}
    problems = []
    if doc["count"] < intel.FEEDS[name]["min_items"]:
        problems.append(f"only {doc['count']} indicators")
    if old_n >= 100 and doc["count"] < 0.1 * old_n:
        problems.append(f"indicators fell {old_n} -> {doc['count']}")
    if doc["count"] > 2_000_000:
        problems.append("implausibly large feed")
    if problems:
        c.status, c.reason = "rejected", "; ".join(problems)
    return c


def benign_indicators(lines: list[str]) -> set[tuple[str, str]]:
    inds = set(BENIGN_INDICATORS)
    for ln in lines:
        inds |= intel.indicators(ln)
    return inds


def benign_check_feed(lay: Layout, c: Candidate, inds: set[tuple[str, str]]) -> None:
    assert c.staged is not None
    current = intel.read_feeds(lay.intel_dir(lay.root))
    new_doc = json.loads(c.staged.read_text("utf-8"))
    after_docs = [d for d in current if d["source"] != c.source] + [new_doc]
    before = intel.hits(intel.Index(current), inds)
    after = intel.hits(intel.Index(after_docs), inds)
    chk = _rate_check(before, after, len(inds))
    chk["indicators"] = chk.pop("events")
    c.checks["false_alarm"] = chk
    if not chk["ok"]:
        c.status = "rejected"
        c.reason = (
            f"benign indicator hit rate rose {chk['rise_pp']} pp (> {MAX_RISE_PP})"
        )


# --- benign sample ------------------------------------------------------------------

_LOGHUB_BAD = re.compile(
    r"authentication failure|failed password|invalid user|break-in|illegal user|"
    r"refused|denied|segfault|error",
    re.I,
)


def _spread(lines: list[str], k: int) -> list[str]:
    if len(lines) <= k:
        return lines
    step = len(lines) / k
    return [lines[int(i * step)] for i in range(k)]


def _windows_events(max_each: int = 300) -> list[str]:
    if os.name != "nt":
        return []
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$e=@();foreach($l in 'System','Application'){"
        f"$e+=Get-WinEvent -LogName $l -MaxEvents {max_each}}};"
        "$e|Select-Object Id,ProviderName,LogName,"
        "@{n='TimeCreated';e={$_.TimeCreated.ToString('o')}},Message"
        "|ConvertTo-Json -Compress -Depth 2"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=180,
            encoding="utf-8",
            errors="replace",
        )
        arr = json.loads(r.stdout or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    arr = arr if isinstance(arr, list) else [arr]
    return [json.dumps(o) for o in arr if isinstance(o, dict) and o.get("Id")]


def build_benign_sample(lay: Layout, path: Path | None = None) -> dict[str, int]:
    """Fixed benign sample for the false-alarm check: normal Loghub Apache + Linux
    lines (error / auth-failure lines left out) and this PC's own System and
    Application events. Built once; the same sample is reused for every update."""
    path = path or lay.benign
    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    loghub = lay.root / "data" / "raw" / "loghub"
    for name, k in (("Apache", 250), ("Linux", 250)):
        p = loghub / name / f"{name}.log"
        try:
            lines = p.read_text("utf-8", errors="replace").splitlines()
        except OSError:
            continue
        keep = _spread(
            [ln for ln in lines if ln.strip() and not _LOGHUB_BAD.search(ln)], k
        )
        rows += [{"source": f"loghub/{name}", "line": ln} for ln in keep]
        counts[f"loghub/{name}"] = len(keep)
    win = _windows_events()
    rows += [{"source": "windows/System+Application", "line": ln} for ln in win]
    counts["windows"] = len(win)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    return counts


def read_benign(path: Path) -> list[str]:
    out = []
    try:
        for ln in path.read_text("utf-8").splitlines():
            with contextlib.suppress(ValueError):
                out.append(str(json.loads(ln)["line"]))
    except OSError:
        return []
    return out


# --- tests on a sandbox copy --------------------------------------------------------

_COPY = ("src", "utils", "scripts", "tests", "config", "third_party", "security",
         "deploy_siem", ".streamlit")  # fmt: skip
_LINK = ("data", "models", "runs", "results", "logs", "datasets", "tools")
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".mypy_cache", ".pytest_cache", "coreruleset")


def _link_dir(link: Path, target: Path) -> bool:
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                timeout=30,
            )
            return r.returncode == 0
        link.symlink_to(target, target_is_directory=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _unlink_dir(link: Path) -> None:
    """Remove a junction / symlink without touching what it points to."""
    with contextlib.suppress(OSError):
        if link.is_symlink() or getattr(os.path, "isjunction", lambda _: False)(link):
            os.unlink(link) if link.is_symlink() else os.rmdir(link)


def build_sandbox(lay: Layout, cands: list[Candidate], dest: Path) -> list[Path]:
    """Copy the project's code into ``dest`` (big data folders linked, not copied)
    and put the staged files in place. Returns the links made."""
    root = lay.root
    links: list[Path] = []
    for name in _COPY:
        if (root / name).is_dir():
            shutil.copytree(root / name, dest / name, ignore=_IGNORE)
    for p in root.iterdir():
        if p.is_file() and p.suffix in (".py", ".toml", ".cfg", ".ini", ".txt"):
            shutil.copy2(p, dest / p.name)
    for name in _LINK:
        if (root / name).is_dir() and _link_dir(dest / name, root / name):
            links.append(dest / name)
    art = root / "artifacts"
    if art.is_dir():
        (dest / "artifacts").mkdir(exist_ok=True)
        for p in art.iterdir():
            if p.name in ("intel", "updates"):
                continue
            if p.is_dir():
                if _link_dir(dest / "artifacts" / p.name, p):
                    links.append(dest / "artifacts" / p.name)
            else:
                shutil.copy2(p, dest / "artifacts" / p.name)
        if lay.intel_dir(root).is_dir():
            shutil.copytree(lay.intel_dir(root), dest / "artifacts" / "intel")
    for c in cands:
        if c.staged is None:
            continue
        if c.source == "sigma":
            rules = lay.sigma_dir(dest) / "rules"
            shutil.rmtree(rules, ignore_errors=True)
            shutil.copytree(c.staged / "rules", rules)
        elif c.source == "attack":
            shutil.copy2(c.staged / "attack_enterprise.json", lay.attack_json(dest))
        elif c.source == "owasp_top10":
            shutil.copy2(c.staged / "owasp_top10.json", lay.owasp_json(dest))
        else:
            (dest / "artifacts" / "intel").mkdir(parents=True, exist_ok=True)
            shutil.copy2(c.staged, dest / "artifacts" / "intel" / c.staged.name)
    return links


def run_tests(lay: Layout, cands: list[Candidate]) -> dict[str, Any]:
    """The project's tests on a sandbox copy with ``cands`` staged in place."""
    tmp = Path(tempfile.mkdtemp(prefix="ug_"))  # short name: Windows MAX_PATH
    links: list[Path] = []
    t0 = time.time()
    try:
        links = build_sandbox(lay, cands, tmp)
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:warnings",
                "-p",
                "no:cacheprovider",
            ],  # fmt: skip
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_S,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            encoding="utf-8",
            errors="replace",
        )
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "summary": " | ".join(tail)[-400:],
            "seconds": round(time.time() - t0, 1),
        }
    except (OSError, subprocess.SubprocessError, shutil.Error) as e:
        return {"ok": False, "summary": f"could not run tests: {e}", "seconds": 0}
    finally:
        for link in links:
            _unlink_dir(link)
        if not any(p.exists() for p in links):
            shutil.rmtree(tmp, ignore_errors=True)


# --- apply / rollback ---------------------------------------------------------------


def _replace_dir(src: Path, dst: Path, backup: Path) -> None:
    if dst.exists():
        shutil.rmtree(backup, ignore_errors=True)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(backup))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _replace_file(src: Path, dst: Path, backup: Path) -> None:
    if dst.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, backup)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".new")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def apply(lay: Layout, c: Candidate) -> None:
    """Put a gated candidate live, keeping the replaced version for rollback."""
    assert c.staged is not None
    prev = lay.previous / c.source
    if c.source == "sigma":
        sd = lay.sigma_dir()
        _replace_dir(c.staged / "rules", sd / "rules", prev / "rules")
        for z in sd.glob("sigma_*_r*.zip"):
            prev.mkdir(parents=True, exist_ok=True)
            shutil.move(str(z), str(prev / z.name))
        for z in c.staged.glob("sigma_*_r*.zip"):
            shutil.copy2(z, sd / z.name)
        prev.mkdir(parents=True, exist_ok=True)
        (prev / "VERSION").write_text(c.old_version, "utf-8")
    elif c.source == "attack":
        _replace_file(
            c.staged / "attack_enterprise.json",
            lay.attack_json(),
            prev / "attack_enterprise.json",
        )
        prev.mkdir(parents=True, exist_ok=True)
        (prev / "VERSION").write_text(c.old_version, "utf-8")
    elif c.source == "owasp_crs":
        cd = lay.crs_dir()
        _replace_dir(c.staged / "rules", cd / "rules", cd / ".previous" / "rules")
        (cd / ".previous").mkdir(parents=True, exist_ok=True)
        (cd / ".previous" / "VERSION").write_text(c.old_version, "utf-8")
        shutil.copy2(c.staged / "VERSION", cd / "VERSION")
        shutil.rmtree(c.staged, ignore_errors=True)
        from src.soc import crs

        crs._rules.cache_clear()
        prev.mkdir(parents=True, exist_ok=True)
        (prev / "VERSION").write_text(c.old_version, "utf-8")
    elif c.source == "owasp_top10":
        _replace_file(c.staged / "owasp_top10.json", lay.owasp_json(), prev / "owasp_top10.json")
        prev.mkdir(parents=True, exist_ok=True)
        (prev / "VERSION").write_text(c.old_version, "utf-8")
    else:
        _replace_file(c.staged, lay.intel_dir() / c.staged.name, prev / c.staged.name)
    c.status = "applied"


def rollback(source: str, lay: Layout | None = None) -> dict[str, Any]:
    """Restore the version an update replaced (one step back)."""
    lay = lay or Layout()
    prev = lay.previous / source
    if not prev.is_dir():
        return {"ok": False, "reason": f"no previous version of {source}"}
    st = load_state(lay.state)
    old_ver = (
        (prev / "VERSION").read_text("utf-8") if (prev / "VERSION").is_file() else ""
    )
    if source == "sigma" and (prev / "rules").is_dir():
        sd = lay.sigma_dir()
        shutil.rmtree(sd / "rules", ignore_errors=True)
        shutil.copytree(prev / "rules", sd / "rules")
        for z in sd.glob("sigma_*_r*.zip"):
            z.unlink()
        for z in prev.glob("sigma_*_r*.zip"):
            shutil.copy2(z, sd / z.name)
    elif source == "attack" and (prev / "attack_enterprise.json").is_file():
        shutil.copy2(prev / "attack_enterprise.json", lay.attack_json())
    elif source == "owasp_crs" and (lay.crs_dir() / ".previous" / "rules").is_dir():
        cd = lay.crs_dir()
        shutil.rmtree(cd / "rules", ignore_errors=True)
        shutil.copytree(cd / ".previous" / "rules", cd / "rules")
        if (cd / ".previous" / "VERSION").is_file():
            shutil.copy2(cd / ".previous" / "VERSION", cd / "VERSION")
    elif source == "owasp_top10" and (prev / "owasp_top10.json").is_file():
        shutil.copy2(prev / "owasp_top10.json", lay.owasp_json())
    elif source in intel.FEEDS and (prev / f"{source}.json").is_file():
        shutil.copy2(prev / f"{source}.json", lay.intel_dir() / f"{source}.json")
    else:
        return {"ok": False, "reason": f"previous copy of {source} is incomplete"}
    st.setdefault("versions", {})[source] = old_ver
    st["history"] = (
        st.get("history", []) + [{"ts": _now(), "action": "rollback", "source": source}]
    )[-HISTORY_KEEP:]
    save_state(st, lay.state)
    return {"ok": True, "source": source, "restored": old_ver}


# --- the agent ----------------------------------------------------------------------


@contextlib.contextmanager
def _lock(lay: Layout) -> Iterator[bool]:
    p = lay.work / ".lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if p.exists() and time.time() - p.stat().st_mtime < LOCK_STALE_S:
            yield False
            return
        p.write_text(str(os.getpid()), "utf-8")
    except OSError:
        yield False
        return
    try:
        yield True
    finally:
        with contextlib.suppress(OSError):
            p.unlink()


def is_due(st: dict[str, Any], source: str, now: float | None = None) -> bool:
    """Feeds: every daily run. Detection content: when it was last checked at least
    CONTENT_EVERY_DAYS ago (a failed download is retried the next day)."""
    if source not in CONTENT_SOURCES:
        return True
    last = str(st.get("checked", {}).get(source) or "")
    try:
        t = calendar.timegm(time.strptime(last[:19], "%Y-%m-%dT%H:%M:%S"))  # stored in UTC
    except ValueError:
        return True
    return (time.time() if now is None else now) - t >= CONTENT_EVERY_DAYS * 86400


def _stage(lay: Layout, st: dict[str, Any], src: str, force: bool) -> Candidate:
    try:
        if src == "sigma":
            return stage_sigma(lay, st, force)
        if src == "attack":
            return stage_attack(lay, st, force)
        if src == "owasp_top10":
            return stage_owasp_top10(lay, st, force)
        if src == "owasp_crs":
            return stage_owasp_crs(lay, st, force)
        return stage_feed(lay, src)
    except NET_ERRORS as e:  # offline, timeout, HTTP error: retry next run
        _log.warning("update %s: download failed: %s", src, e)
        return Candidate(src, status="error", reason=f"download failed: {e}"[:300])
    except Exception as e:  # noqa: BLE001 - a bad download must never crash the agent
        _log.warning("update %s: staging failed: %r", src, e)
        return Candidate(src, status="error", reason=f"staging failed: {e!r}"[:300])


def run_once(
    force: bool = False,
    dry_run: bool = False,
    lay: Layout | None = None,
    sources: tuple[str, ...] = SOURCES,
    run_tests_gate: bool = True,
) -> dict[str, Any]:
    """One update cycle -> report dict. ``force``: run even when switched off or
    already run today, and rebuild sources that are already current. ``dry_run``:
    stage, validate and gate as usual but apply into ``lay.out`` (a scratch copy)."""
    lay = lay or Layout()
    st = load_state(lay.state)
    report: dict[str, Any] = {"started": _now(), "dry_run": dry_run, "sources": {}}
    if not force and not st.get("enabled"):
        return report | {"status": "disabled"}
    today = _now()[:10]
    if not force and str(st.get("last_run") or "")[:10] == today:
        return report | {"status": "already ran today"}
    with _lock(lay) as got:
        if not got:
            return report | {"status": "another update is running"}
        try:
            report |= _cycle(lay, st, force, dry_run, sources, run_tests_gate)
        except Exception as e:  # noqa: BLE001 - never crash the scheduled task
            _log.exception("update agent failed")
            report |= {"status": "error", "error": repr(e)[:300]}
    report["finished"] = _now()
    st = load_state(lay.state)  # the toggle may have changed meanwhile
    st["last_run"] = report["finished"]
    st["history"] = (
        st.get("history", [])
        + [
            {
                "ts": report["finished"],
                "action": "dry_run" if dry_run else "run",
                "status": report.get("status"),
                "sources": {
                    k: {kk: v[kk] for kk in ("status", "new_version", "reason")}
                    for k, v in report["sources"].items()
                },
            }
        ]
    )[-HISTORY_KEEP:]
    for k, v in report["sources"].items():
        if v["status"] in ("applied", "dry_run"):
            st.setdefault("versions", {})[k] = v["new_version"]
        if v["status"] != "error":  # checked (applied, current or rejected): next check in a month
            st.setdefault("checked", {})[k] = report["finished"]
    save_state(st, lay.state)
    return report


def _cycle(
    lay: Layout,
    st: dict[str, Any],
    force: bool,
    dry_run: bool,
    sources: tuple[str, ...],
    run_tests_gate: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    bpath = lay.benign
    root_sample = lay.root / "artifacts" / "intel" / "benign_sample.jsonl"
    if not bpath.is_file() and root_sample.is_file():
        bpath = root_sample  # dry run: reuse the project's fixed sample
    if not bpath.is_file():
        out["benign_sample_built"] = build_benign_sample(lay, bpath)
    lines = read_benign(bpath)
    out["benign_sample_events"] = len(lines)
    inds = benign_indicators(lines)
    due = [s for s in sources if force or is_due(st, s)]
    out["not_due"] = [s for s in sources if s not in due]
    cands = [_stage(lay, st, s, force) for s in due]
    for c in cands:
        if c.status != "pending":
            continue
        try:
            if c.source == "sigma":
                benign_check_sigma(lay, c, lines)
            elif c.source == "owasp_crs":
                benign_check_crs(lay, c, lines)
            elif c.source in intel.FEEDS:
                benign_check_feed(lay, c, inds)
        except Exception as e:  # noqa: BLE001
            c.status, c.reason = "rejected", f"false-alarm check failed: {e!r}"[:300]
    ready = [c for c in cands if c.status == "pending"]
    if ready and run_tests_gate:
        res = run_tests(lay, ready)
        out["tests"] = res
        if not res["ok"] and len(ready) > 1:  # find which update breaks the tests
            for c in ready:
                one = run_tests(lay, [c])
                c.checks["tests"] = one
                if not one["ok"]:
                    c.status, c.reason = "rejected", "tests fail: " + one["summary"]
        elif not res["ok"]:
            ready[0].checks["tests"] = res
            ready[0].status, ready[0].reason = (
                "rejected",
                "tests fail: " + res["summary"],
            )
        else:
            for c in ready:
                c.checks["tests"] = {"ok": True}
    elif ready:
        out["tests"] = {"ok": None, "summary": "skipped (run_tests_gate=False)"}
    for c in cands:
        if c.status != "pending":
            continue
        try:
            apply(lay, c)
            if dry_run:
                c.status = "dry_run"
        except OSError as e:
            c.status, c.reason = "error", f"apply failed: {e}"[:300]
            with contextlib.suppress(Exception):
                rollback(c.source, lay)
    for c in cands:
        if c.source == "attack" and c.status == "applied" and c.checks.get("new_tactics"):
            _notify_new_tactics(c.checks["new_tactics"])
    out["sources"] = {c.source: c.report() for c in cands}
    out["status"] = "ok"
    return out


def _notify_new_tactics(names: list[str]) -> None:
    """Owner decision 2026-10-23: a new ATT&CK tactic gets the neutral severity weight 5
    automatically, and the owner is asked to review it."""
    from src.soc import osops

    msg = (f"ATT&CK update added tactic(s) {', '.join(names)}. Incidents give them severity weight 5 "
           "until reviewed: set a weight in src/soc/correlate.py (TACTIC_WEIGHT).")  # fmt: skip
    _log.warning(msg)
    with contextlib.suppress(Exception):
        osops.notify(msg)
