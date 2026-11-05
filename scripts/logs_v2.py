"""Log detection on all of Splunk attack_data (owner 2026-10-24): dev / held-out halves.

The owner released the used blind sets (2026-10-24), so every attack_data dataset with
an ATT&CK label (data/raw/splunk-attack-data/attack_data_repo.zip, read in place) is
split by dataset FOLDER into a DEV half (rules may be written from it) and a HELD-OUT
half (scored only):

  * folders already in data/attack_data_dev keep their earlier split (rules were
    written from that dev half on 2026-10-15, so its held-out half stays clean);
  * every other folder is assigned by a seeded hash of its path.

Per file: detected (any technique), right tactic, right technique, by sourcetype, plus
the alert rate on benign logs (scripts.attackdata_followup.benign_lines). The new blind
log sets (AIT-LDS v2, EVTX-to-MITRE-Attack in D:/SplitShield-data/blind2) are never read.

    python -m scripts.logs_v2 score --stage before|after [--out DIR] [--split dev|heldout|both]
    python -m scripts.logs_v2 misses [--sourcetype auditd] [--limit 20]   (dev half only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT, RAW_DIR, RESULTS_DIR

REPO_ZIP = RAW_DIR / "splunk-attack-data" / "attack_data_repo.zip"
OLD_DEV = PROJECT_ROOT / "data" / "attack_data_dev" / "manifest.json"
PREFIX = "attack_data-master/"
MAX_BYTES = 20 << 20
SEED = "splitshield-logs-v2-2026-10-24"


def split_of(folder: str, old: dict[str, str]) -> str:
    if folder in old:
        return old[folder]
    h = hashlib.sha256(f"{SEED}:{folder}".encode()).digest()[0]
    return "dev" if h % 2 == 0 else "heldout"


@lru_cache(maxsize=1)
def pool() -> list[dict[str, Any]]:
    """Every labelled attack_data log file in the repo zip, with its split."""
    import yaml

    old: dict[str, str] = {}
    if OLD_DEV.is_file():
        for r in json.loads(OLD_DEV.read_text("utf-8")):
            old[str(r["folder"])] = str(r["split"])
    z = zipfile.ZipFile(REPO_ZIP)
    sizes = {i.filename: i.file_size for i in z.infolist()}
    out = []
    for n in z.namelist():
        if not n.endswith((".yml", ".yaml")) or "/datasets/" not in n:
            continue
        try:
            d = yaml.safe_load(z.read(n))
        except Exception:  # noqa: BLE001 - a broken yml is skipped
            continue
        if not isinstance(d, dict):
            continue
        techs = sorted({str(t) for t in d.get("mitre_technique") or [] if str(t).startswith("T")})
        if not techs:
            continue
        for ds in d.get("datasets") or []:
            if not isinstance(ds, dict) or not ds.get("sourcetype"):
                continue
            p = str(ds.get("path") or "").lstrip("/")
            member = PREFIX + p
            if member not in sizes or sizes[member] < 200:
                continue
            folder = p.rsplit("/", 1)[0]
            out.append({"path": p, "member": member, "folder": folder, "sourcetype": str(ds["sourcetype"]),
                        "techniques": techs, "bytes": sizes[member], "split": split_of(folder, old)})  # fmt: skip
    return sorted(out, key=lambda r: r["path"])


def _read(z: zipfile.ZipFile, member: str) -> str:
    with z.open(member) as fh:
        return fh.read(MAX_BYTES).decode("utf-8", errors="replace")


def score_rows(rows: list[dict[str, Any]], keep: bool = False) -> dict[str, Any]:
    from src.knowledge.attack_details import tactics_of
    from src.soc import parsers
    from src.soc.parsers import parse_text

    z = zipfile.ZipFile(REPO_ZIP)
    by: dict[str, Counter[str]] = {}
    per: list[dict[str, Any]] = []
    for m in rows:
        truth_t = {t.split(".")[0] for t in m["techniques"]}
        truth_tac = {x for t in truth_t for x in tactics_of(t)}
        if hasattr(parsers, "reset_sessions"):
            parsers.reset_sessions()
        text = _read(z, m["member"])
        evs = parse_text(text, Path(m["path"]).name)
        techs = {t.split(".")[0] for e in evs for t in e.get("techniques", [])}
        tacs = {x for t in techs for x in tactics_of(t)}
        r = {"detected": bool(techs), "right_tactic": bool(tacs & truth_tac), "right_technique": bool(techs & truth_t)}
        # noise: share of this recording's events that raised an alert (most events in
        # an attack recording are normal background activity)
        noise = sum(1 for e in evs if e.get("techniques")) / max(1, len(evs))
        c = by.setdefault(m["sourcetype"], Counter())
        c["files"] += 1
        for k, v in r.items():
            c[k] += v
        c["_noise_milli"] += round(1000 * noise)
        if keep:
            per.append({"path": m["path"], "sourcetype": m["sourcetype"], "truth": sorted(truth_t),
                        "found": sorted(techs), "events": len(evs), "noise": round(noise, 4), **r})  # fmt: skip
    tot: Counter[str] = Counter()
    for c in by.values():
        tot.update(c)
    n = max(1, tot["files"])
    rates = {k: round(tot[k] / n, 4) for k in ("detected", "right_tactic", "right_technique")}
    rates["mean_event_alert_share"] = round(tot["_noise_milli"] / 1000 / n, 4)
    res: dict[str, Any] = {"by_sourcetype": {k: dict(v) for k, v in sorted(by.items())}, "total": dict(tot),
                           "rates": rates}  # fmt: skip
    if keep:
        res["rows"] = per
    return res


def score(stage: str, split: str, out: Path, rules: Path | None = None,
          levels: tuple[str, ...] | None = None) -> dict[str, Any]:  # fmt: skip
    import contextlib

    from src.soc import sigma
    from src.soc.updater import sigma_rules_at

    ctx = sigma_rules_at(rules) if rules else contextlib.nullcontext()
    saved = sigma.LEVELS
    if levels:
        sigma.LEVELS = levels
    sigma.load.cache_clear()
    try:
        with ctx:
            return _score(stage, split, out, rules, levels)
    finally:
        sigma.LEVELS = saved


def _score(stage: str, split: str, out: Path, rules: Path | None, levels: tuple[str, ...] | None) -> dict[str, Any]:
    from scripts.attackdata_followup import _benign_rate

    t0 = time.time()
    rows = pool()
    res: dict[str, Any] = {"label": "all labelled Splunk attack_data (owner released 2026-10-24); "
                           "dev half for writing rules, held-out half scored only",
                           "stage": stage, "when": time.strftime("%Y-%m-%d %H:%M"),
                           "files": Counter(r["split"] for r in rows),
                           "rules": str(rules or "deployed bundle"), "levels": list(levels or [])}  # fmt: skip
    for s in ("dev", "heldout"):
        if split in (s, "both"):
            res[s] = score_rows([r for r in rows if r["split"] == s], keep=True)
    res["benign"] = _benign_rate()
    res["seconds"] = round(time.time() - t0)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "logs_v2.json"
    doc = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    doc[stage] = res
    path.write_text(json.dumps(doc, indent=1), "utf-8")
    return res


def misses(sourcetype: str | None, limit: int) -> list[dict[str, Any]]:
    """Dev-half files that were missed (or got the wrong tactic), for writing rules."""
    rows = [r for r in pool() if r["split"] == "dev" and (sourcetype is None or r["sourcetype"] == sourcetype)]
    res = score_rows(rows, keep=True)
    return [r for r in res["rows"] if not r["right_tactic"]][:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--stage", required=True)
    s.add_argument("--split", default="both", choices=("dev", "heldout", "both"))
    s.add_argument("--out", type=Path, default=RESULTS_DIR / "extra")
    s.add_argument("--rules", type=Path, default=None, help="another Sigma rules folder")
    s.add_argument("--levels", nargs="+", default=None, help="Sigma levels to load, e.g. low medium high critical")
    m = sub.add_parser("misses")
    m.add_argument("--sourcetype", default=None)
    m.add_argument("--limit", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "score":
        res = score(a.stage, a.split, a.out, a.rules, tuple(a.levels) if a.levels else None)
        print(json.dumps({k: res[k]["rates"] for k in ("dev", "heldout") if k in res}, indent=1))
        print(json.dumps({k: res[k]["by_sourcetype"] for k in ("dev", "heldout") if k in res}, indent=0)[:4000])
    else:
        for r in misses(a.sourcetype, a.limit):
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
