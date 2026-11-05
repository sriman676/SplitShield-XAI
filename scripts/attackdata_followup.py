"""Follow-up on NON-blind Splunk attack_data files (owner 2026-10-15).

The blind test (results/extra/blind_tests.json, key "logs") was weak for Linux, web
and identity logs. New rules may only be written from data the blind sample never
used, so this script:

  download  picks every attack_data dataset for the weak sourcetypes whose file AND
            dataset folder are absent from the blind manifest, splits them by dataset
            folder (seeded) into a DEV half (rules are written from it) and a
            HELD-OUT half (scored only), and downloads them to data/attack_data_dev/;
  score     scores both halves (alerted / right tactic / right technique, per
            sourcetype) plus the alert rate on benign logs, as stage "before" or
            "after" the new rules; writes results/extra/attackdata_followup.{json,md}.

This is a labelled follow-up on non-blind files; the registered blind result is
unchanged and the blind sample is never re-scored.

    python -m scripts.attackdata_followup download
    python -m scripts.attackdata_followup score --stage before|after
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from utils.paths import PROJECT_ROOT, RAW_DIR, RESULTS_DIR

DEV = PROJECT_ROOT / "data" / "attack_data_dev"
BLIND = PROJECT_ROOT / "data" / "blind" / "splunk" / "manifest.json"
REPO_ZIP = RAW_DIR / "splunk-attack-data" / "attack_data_repo.zip"
OUT = RESULTS_DIR / "extra"
TARGET = (
    "auditd", "linux_secure", "sysmon:linux", "linux_messages_syslog",
    "nginx:plus:access", "iis", "OktaIM2:log", "azure:monitor:aad",
    "o365:management:activity",
)  # fmt: skip
MAX_BYTES = 20 << 20
SEED = 2026


def _blind_sets() -> tuple[set[str], set[str]]:
    rows = json.loads(BLIND.read_text("utf-8"))
    paths = {r["path"] for r in rows}
    dirs = {r["path"].rsplit("/", 1)[0] for r in rows}
    dirs |= {
        r["yml"].split("attack_data-master/", 1)[-1].rsplit("/", 1)[0] for r in rows
    }
    return paths, dirs


def pool() -> list[dict[str, Any]]:
    """Every target-sourcetype dataset outside the blind sample (files + folders)."""
    import yaml

    b_paths, b_dirs = _blind_sets()
    out = []
    z = zipfile.ZipFile(REPO_ZIP)
    for n in z.namelist():
        if not n.endswith((".yml", ".yaml")) or "/datasets/" not in n:
            continue
        try:
            d = yaml.safe_load(z.read(n))
        except Exception:  # noqa: BLE001 - a broken yml is skipped
            continue
        if not isinstance(d, dict):
            continue
        techs = [
            str(t) for t in d.get("mitre_technique") or [] if str(t).startswith("T")
        ]
        for ds in d.get("datasets") or []:
            if not isinstance(ds, dict) or ds.get("sourcetype") not in TARGET:
                continue
            p = str(ds.get("path") or "").lstrip("/")
            folder = p.rsplit("/", 1)[0]
            if not p or p in b_paths or folder in b_dirs:
                continue
            out.append({"path": p, "folder": folder, "sourcetype": ds["sourcetype"],
                        "techniques": techs})  # fmt: skip
    return out


def download() -> dict[str, Any]:
    items = pool()
    folders = sorted({i["folder"] for i in items})
    random.Random(SEED).shuffle(folders)
    held = set(folders[: len(folders) // 2])
    DEV.mkdir(parents=True, exist_ok=True)
    ua = {"User-Agent": "splitshield-followup"}

    def fetch(it: dict[str, Any]) -> dict[str, Any]:
        it["split"] = "heldout" if it["folder"] in held else "dev"
        dest = DEV / it["split"] / it["path"].replace("/", "__")
        dest.parent.mkdir(parents=True, exist_ok=True)
        it["file"] = str(dest.relative_to(DEV))
        if dest.exists() and dest.stat().st_size > 200:
            return it
        url = (
            "https://media.githubusercontent.com/media/splunk/attack_data/master/"
            + it["path"]
        )
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=ua), timeout=120
            ) as r:
                if int(r.headers.get("Content-Length") or 0) > MAX_BYTES:
                    it["skipped"] = "too large"
                    return it
                dest.write_bytes(r.read())
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            it["error"] = str(exc)[:120]
        return it

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch, items))
    (DEV / "manifest.json").write_text(json.dumps(res, indent=1), "utf-8")
    ok = Counter(
        (r["split"], r["sourcetype"])
        for r in res
        if "error" not in r and "skipped" not in r
    )
    return {
        "files": len(res),
        "downloaded": {f"{a}/{b}": n for (a, b), n in sorted(ok.items())},
    }


# --- scoring ------------------------------------------------------------------------


def _score_files(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from src.knowledge.attack_details import tactics_of
    from src.soc.parsers import parse_text

    by: dict[str, Counter[str]] = {}
    for m in rows:
        f = DEV / m["file"]
        if "error" in m or "skipped" in m or not f.is_file():
            continue
        truth_t = {t.split(".")[0] for t in m["techniques"]}
        truth_tac = {x for t in truth_t for x in tactics_of(t)}
        try:
            evs = parse_text(f.read_text("utf-8", errors="replace"), f.name)
        except OSError:
            continue
        techs = {t.split(".")[0] for e in evs for t in e.get("techniques", [])}
        tacs = {x for t in techs for x in tactics_of(t)}
        c = by.setdefault(m["sourcetype"], Counter())
        c["files"] += 1
        c["detected"] += bool(techs)
        c["right_tactic"] += bool(tacs & truth_tac)
        c["right_technique"] += bool(techs & truth_t)
    tot: Counter[str] = Counter()
    for c in by.values():
        tot.update(c)
    return {
        "by_sourcetype": {k: dict(v) for k, v in sorted(by.items())},
        "total": dict(tot),
    }


def benign_lines() -> dict[str, list[str]]:
    """Benign logs on this PC: Loghub Apache (error log) and Loghub Linux with the
    authentication-failure / brute-force lines removed (Loghub SSH is genuine
    internet brute force, so it is not benign and is not used)."""
    out = {}
    ap = RAW_DIR / "loghub" / "Apache" / "Apache.log"
    if ap.is_file():
        out["loghub_apache"] = ap.read_text("utf-8", errors="replace").splitlines()
    lx = RAW_DIR / "loghub" / "Linux" / "Linux.log"
    if lx.is_file():
        bad = ("authentication failure", "Failed password", "Invalid user", "check pass",
               "BREAK-IN", "illegal user")  # fmt: skip
        out["loghub_linux_clean"] = [ln for ln in lx.read_text("utf-8", errors="replace").splitlines()
                                     if not any(b in ln for b in bad)]  # fmt: skip
    return out


def _benign_rate() -> dict[str, Any]:
    from src.soc.parsers import parse_line

    res = {}
    for name, lines in benign_lines().items():
        n = hit = 0
        rules: Counter[str] = Counter()
        for ln in lines:
            ev = parse_line(ln)
            if ev is None:
                continue
            n += 1
            if ev.get("techniques"):
                hit += 1
                rules[str(ev.get("kind"))] += 1
        res[name] = {"events": n, "alerts": hit, "rate": hit / max(1, n),
                     "top": rules.most_common(5)}  # fmt: skip
    return res


def score(stage: str) -> dict[str, Any]:
    rows = json.loads((DEV / "manifest.json").read_text("utf-8"))
    t0 = time.time()
    doc = {
        "label": "follow-up on non-blind attack_data files; the registered blind result is unchanged",
        "dev": _score_files([r for r in rows if r["split"] == "dev"]),
        "heldout": _score_files([r for r in rows if r["split"] == "heldout"]),
        "benign": _benign_rate(),
        "seconds": round(time.time() - t0),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    store = OUT / "attackdata_followup.json"
    allr = json.loads(store.read_text("utf-8")) if store.is_file() else {}
    allr[stage] = doc
    store.write_text(json.dumps(allr, indent=1), "utf-8")
    _write_md(allr)
    return doc


def _pct(c: dict[str, int], k: str) -> str:
    return f"{c.get(k, 0)}/{c.get('files', 0)}"


def _write_md(allr: dict[str, Any]) -> None:
    lines = ["# 🔁 Follow-up: Linux / identity rules on non-blind Splunk attack_data",
             "",
             "_Follow-up on non-blind attack_data files; the registered blind result is "
             "unchanged and the blind sample is never re-scored._ Rules were written from the "
             "DEV half only; the HELD-OUT half (other dataset folders) is scored only.",
             ""]  # fmt: skip
    for split in ("dev", "heldout"):
        lines += [f"## {split}", "", "| sourcetype | stage | files | alerted | right tactic | right technique |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        sts = sorted({s for st in allr.values() for s in st[split]["by_sourcetype"]})
        for s in [*sts, "TOTAL"]:
            for stage in ("before", "after"):
                if stage not in allr:
                    continue
                d = allr[stage][split]
                c = d["total"] if s == "TOTAL" else d["by_sourcetype"].get(s, {})
                lines.append(f"| {s} | {stage} | {c.get('files', 0)} | {_pct(c, 'detected')} | "
                             f"{_pct(c, 'right_tactic')} | {_pct(c, 'right_technique')} |")  # fmt: skip
        lines.append("")
    lines += [
        "## Alerts on benign logs",
        "",
        "| log | stage | events | alerts | rate |",
        "|---|---|---|---|---|",
    ]
    for stage in ("before", "after"):
        for k, v in (allr.get(stage) or {}).get("benign", {}).items():
            lines.append(
                f"| {k} | {stage} | {v['events']} | {v['alerts']} | {v['rate']:.4%} |"
            )
    (OUT / "attackdata_followup.md").write_text("\n".join(lines) + "\n", "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("download", "score"))
    ap.add_argument("--stage", default="after", choices=("before", "after"))
    a = ap.parse_args()
    res = download() if a.cmd == "download" else score(a.stage)
    print(json.dumps(res, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
