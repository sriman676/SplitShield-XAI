"""Shared risk board: the models coordinate through it (owner 2026-10-15).

Every detector posts its findings about an *entity* (IP, user, host, URL, domain,
file) to one board. The board also links entities that appear together in an event
(a logon ties a user to an IP and a host; a process ties a file to a host). An
incident's severity is then raised when OTHER model families have findings about the
same entity, or about an entity one link away, within ``window_s``:

  boost = min(cap, weight x number of other families)

Families come from config/models.yaml (``family``) and the log source (network,
endpoint, web, auth, ids, cloud, phishing, malware, insider, ics, intel). The board
never lowers a severity and never hides an alert. ``weight`` and ``cap`` were chosen
on the AIT development servers and checked on the held-out servers
(results/extra/riskboard_validation.json, ``scripts/validate_riskboard.py``); the place
profile (src/soc/places.py) may scale the weight only within the validated range.

Hubs (a DNS server or proxy every user talks to) would link everything to everything,
so an entity with more than ``max_degree`` links is not followed.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from utils.paths import PROJECT_ROOT

Event = dict[str, Any]
VALIDATED = PROJECT_ROOT / "results" / "extra" / "riskboard_validation.json"
DEFAULT = {"weight": 1, "cap": 2, "window_s": 24 * 3600.0, "max_degree": 50, "follow_s": 0.0, "hops": 1,
           "link_degree": 0, "link_failed": True}

# log source -> family (substring match, first hit wins)
_SOURCE_FAMILY = (
    (("phishing",), "phishing"),
    (("malware",), "malware"),
    (("insider",), "insider"),
    (("ics",), "ics"),
    (("intel",), "intel"),
    (("detector", "netflow", "zeek", "firewall", "flow"), "network"),
    (("suricata", "snort", "ids"), "ids"),
    (
        ("winsec", "sysmon", "powershell", "windows", "evtx", "osquery", "auditd"),
        "endpoint",
    ),
    (("apache", "nginx", "iis", "http", "web", "access"), "web"),
    (("cloudtrail", "azure", "gcp", "okta", "o365", "k8s", "aws"), "cloud"),
    (("exim", "postfix", "mail", "smtp"), "mail"),
    (("ssh", "auth", "syslog", "secure", "linux"), "auth"),
)
_URL = re.compile(r"\bhttps?://[^\s\"'<>()\[\]|]+", re.I)


def family_of(ev: Event) -> str:
    src = str(ev.get("source", "")).lower()
    if src.startswith("unknown"):
        return "unknown"
    for keys, fam in _SOURCE_FAMILY:
        if any(k in src for k in keys):
            return fam
    return "other"


def is_failed_logon(ev: Event) -> bool:
    """A failed authentication (it names an account and an address, but they belong to
    each other only if the logon succeeded)."""
    kind = str(ev.get("kind", "")).lower()
    return "fail" in kind or "invalid_user" in kind or "T1110" in (ev.get("techniques") or [])


def urls_in(ev: Event) -> list[str]:
    out = [str(ev["url"])] if ev.get("url") else []
    detail = str(ev.get("detail", ""))
    # most events carry no URL: skip the pattern unless "://" is there at all
    return out + (_URL.findall(detail)[:3] if "://" in detail else [])


def _local(ip: str) -> bool:
    return ip.startswith(("127.", "0.")) or ip in ("::1", "::", "0.0.0.0")  # noqa: S104 - not a bind


def entities_of(ev: Event) -> list[str]:
    """Typed entity keys an event is about. The destination IP is left out on
    purpose: the victim is not the actor, and servers would become hubs."""
    keys = []
    if ev.get("src_ip") and not _local(str(ev["src_ip"])):
        keys.append(f"ip:{ev['src_ip']}")
    if ev.get("user"):
        keys.append(f"user:{str(ev['user']).lower()}")
    if ev.get("host"):
        keys.append(f"host:{str(ev['host']).lower()}")
    if ev.get("sha256"):
        keys.append(f"file:{ev['sha256']}")
    elif ev.get("file"):
        keys.append(f"file:{str(ev['file']).lower()}")
    for u in urls_in(ev):
        keys.append(f"url:{u.lower()}")
        host = (
            re.sub(r"^https?://", "", u, flags=re.I).split("/")[0].split(":")[0].lower()
        )
        if host:
            keys.append(f"domain:{host}")
    return keys


@dataclass
class Finding:
    family: str
    ts: float
    techniques: list[str]
    what: str


@dataclass
class Board:
    weight: float = 1.0
    cap: float = 2.0
    window_s: float = 24 * 3600.0
    max_degree: int = 50
    hops: int = 1  # 1: also entities one link away; 0: the incident's own entities only
    findings: dict[str, list[Finding]] = field(
        default_factory=lambda: defaultdict(list)
    )
    links: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # victim follow (owner 2026-10-24): findings by the machine that logged them
    # (``observer``). An incident whose actor attacked that machine counts the
    # machine's findings from other families from the incident's start until
    # ``follow_s`` after its end (0 = off). Servers are hubs, so the link graph
    # never reaches them; this time-bounded attacker -> victim link does.
    follow_s: float = 0.0
    observed: dict[str, list[Finding]] = field(default_factory=lambda: defaultdict(list))
    # safer links (owner 2026-10-24): an entity one link away counts only when it has at
    # most ``link_degree`` links itself (0 = the hub limit only), and failed logons never
    # link (an attacker trying an account does not tie that account to the attacker)
    link_degree: int = 0
    link_failed: bool = True

    def add(self, ev: Event) -> None:
        keys = entities_of(ev)
        if self.link_failed or not is_failed_logon(ev):
            for a in keys:
                for b in keys:
                    if a != b:
                        self.links[a].add(b)
        if ev.get("techniques"):
            f = Finding(family_of(ev), float(ev.get("ts", 0.0)),
                        list(ev["techniques"]), str(ev.get("detail", ""))[:120])  # fmt: skip
            for k in keys:
                self.findings[k].append(f)
            obs = str(ev.get("observer") or ev.get("host") or "").lower()
            if obs:
                self.observed[obs].append(f)

    def add_all(self, events: list[Event]) -> Board:
        for ev in events:
            self.add(ev)
        return self

    def _linkable(self, key: str) -> bool:
        return not self.link_degree or len(self.links.get(key, ())) <= self.link_degree

    def hub(self, key: str) -> bool:
        return len(self.links.get(key, ())) > self.max_degree

    def related(
        self, keys: list[str], start: float, end: float, primary: str | None = None
    ) -> dict[str, list[tuple[str, Finding]]]:
        """family -> [(entity, finding)] on these entities or one link away, in the
        window. Hubs (more than ``max_degree`` links: a proxy, a DNS server, a shared
        host) are skipped - their findings would spread to everything that touches
        them - unless the incident is about the hub itself (``primary``)."""
        usable = {k for k in keys if k == primary or not self.hub(k)}
        near = set(usable)
        for k in usable if self.hops else ():
            if not self.hub(k):
                near |= {n for n in self.links.get(k, set()) if not self.hub(n) and self._linkable(n)}
        out: dict[str, list[tuple[str, Finding]]] = defaultdict(list)
        lo, hi = start - self.window_s, end + self.window_s
        for k in sorted(near):  # sorted: the same reasons in the same order every run
            for f in self.findings.get(k, []):
                if lo <= f.ts <= hi:
                    out[f.family].append((k, f))
        return out

    def boost(
        self,
        keys: list[str],
        own: set[str],
        start: float,
        end: float,
        primary: str | None = None,
        victims: tuple[str, ...] | list[str] = (),
    ) -> tuple[float, list[str]]:
        """(severity points to add, plain-English reasons) for one incident."""
        rel = self.related(keys, start, end, primary)
        if self.follow_s > 0:
            for v in victims:
                for f in self.observed.get(str(v).lower(), []):
                    if start <= f.ts <= end + self.follow_s:
                        rel[f.family].append((f"victim:{v}", f))
        other = sorted(set(rel) - own - {"unknown", "other"})
        pts = min(self.cap, self.weight * len(other))
        reasons = []
        for fam in other:
            ent, f = rel[fam][0]
            reasons.append(f"{fam} finding on {ent.split(':', 1)[0]} {ent.split(':', 1)[1][:60]}: "
                           f"{', '.join(f.techniques[:3]) or 'flagged'} ({f.what[:80]})")  # fmt: skip
        return pts, reasons


def settings() -> dict[str, Any]:
    """The validated settings with the users' changes (Customise -> Engine settings) on top."""
    s = validated_settings()
    try:
        from src.soc import rulectl

        user = rulectl._read(rulectl.SETTINGS, {})
    except Exception:  # noqa: BLE001 - never stops correlation
        user = {}
    for key, name, cast in (("riskboard_weight", "weight", float), ("riskboard_cap", "cap", float),
                            ("riskboard_hops", "hops", int), ("riskboard_follow_s", "follow_s", float)):  # fmt: skip
        if key in user:
            s[name] = cast(user[key])
    return s


def validated_settings() -> dict[str, Any]:
    """Validated weight/cap (or the safe defaults when no validation ran)."""
    try:
        doc = json.loads(VALIDATED.read_text("utf-8"))
        chosen = doc.get("chosen", {})
    except (OSError, ValueError):
        chosen = {}
    return DEFAULT | {k: chosen[k] for k in ("weight", "cap", "follow_s", "hops", "link_degree", "link_failed")
                      if k in chosen}


def entity_key(correlate_entity: str) -> str:
    """correlate.entity_of() form ("1.2.3.4" / "user:x" / "host:x") -> board key."""
    kind, sep, val = correlate_entity.partition(":")
    if sep and kind in ("user", "host"):
        return f"{kind}:{val.lower()}"
    return f"ip:{correlate_entity}"


def board_for(events: list[Event], scale: float = 1.0) -> Board:
    """A board with the validated settings; ``scale`` (place profile) multiplies the
    weight but is clamped to [0, 1]: a profile may soften, never exceed, validation."""
    s = settings()
    return Board(weight=float(s["weight"]) * max(0.0, min(1.0, scale)), cap=float(s["cap"]),
                 window_s=float(s["window_s"]), max_degree=int(s["max_degree"]),
                 follow_s=float(s["follow_s"]), hops=int(s["hops"]), link_degree=int(s["link_degree"]),
                 link_failed=bool(s["link_failed"])).add_all(events)  # fmt: skip


# --- switching the boost on from the owner's reviews (owner 2026-10-15) ------------------

MIN_REVIEWED = 20  # reviewed incidents with linked findings needed before deciding
HIGH = 6


def learn_from_reviews(min_reviewed: int = MIN_REVIEWED) -> dict[str, Any]:
    """Per-site tuning from reviews (owner 2026-10-24). Once ``min_reviewed`` incidents
    with linked findings have reviewed (Real / False alarm) alerts, every (weight, hops)
    setting is compared on exactly those incidents with the board off: a setting may be
    PROPOSED only if precision at "high or worse" does not drop there; the one with the
    best recall wins. A user accepts it on Customise -> Proposals & history."""
    from src.soc import engine
    from src.soc.correlate import correlate, entity_of
    from src.soc.rcl import RCL

    evs = engine.read_jsonl(engine.EVENTS)
    alerts = {a.get("id"): a for a in engine.read_jsonl(engine.ALERTS)}
    reviews = [(alerts[r["alert_id"]], bool(r["correct"])) for r in RCL().rows()
               if r.get("alert_id") in alerts]  # fmt: skip
    rows: dict[str, Any] = {}
    n_reviewed = 0
    for w, hops in ((0.0, 0), (1.0, 0), (1.0, 1), (2.0, 0), (2.0, 1)):
        incs = correlate(evs, board=Board(weight=w, cap=2.0, hops=hops).add_all(evs))
        labelled = []
        for inc in incs:
            if not inc.get("linked_findings"):
                continue
            hits = [ok for a, ok in reviews if entity_of(a) == inc["src_ip"]
                    and inc["start"] <= float(a.get("ts") or 0) <= inc["end"]]  # fmt: skip
            if hits:
                labelled.append((inc["severity"] >= HIGH, any(hits)))
        n_reviewed = len(labelled)
        high = sum(h for h, _ in labelled)
        tp = sum(h and real for h, real in labelled)
        real = sum(r for _, r in labelled)
        rows[f"w{w:g}_h{hops}"] = {"weight": w, "hops": hops, "high": high, "true_high": tp,
                                   "precision": tp / max(1, high), "recall": tp / max(1, real)}  # fmt: skip
    doc: dict[str, Any] = {}
    try:
        doc = json.loads(VALIDATED.read_text("utf-8"))
    except (OSError, ValueError):
        doc = {}
    out: dict[str, Any] = {"reviewed_incidents_with_links": n_reviewed, "needed": min_reviewed,
                           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
    if n_reviewed < min_reviewed:
        out["status"] = f"waiting for reviews ({n_reviewed}/{min_reviewed})"
    else:
        base = rows["w0_h0"]["precision"]
        ok = [r for r in rows.values() if r["precision"] >= base - 1e-12]
        best = max(ok, key=lambda r: (r["recall"], -r["weight"], -r["hops"]))
        out |= {
            "status": "decided",
            "candidates": rows,
            "chosen_weight": best["weight"],
        }
        now = settings()
        out["proposed"] = {"weight": best["weight"], "hops": best["hops"]}
        from src.soc import rulectl
        from src.soc.controller import notify

        why = (f"{n_reviewed} reviewed incidents with linked findings: weight {best['weight']:g}, hops "
               f"{best['hops']} keeps their precision and has the best recall")  # fmt: skip
        for key, name in (("riskboard_weight", "weight"), ("riskboard_hops", "hops")):
            if float(best[name]) != float(now[name]):  # proposed; a user accepts it
                rulectl.add_proposal(f"engine:{key}", {"value": best[name]}, f"{why} (now {now[name]})")
                notify(f"risk board: your reviews suggest {name} {best[name]} - see Customise -> Proposals")
    doc["review_based"] = out
    VALIDATED.parent.mkdir(parents=True, exist_ok=True)
    VALIDATED.write_text(json.dumps(doc, indent=1), "utf-8")
    return out
