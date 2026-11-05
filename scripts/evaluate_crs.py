"""Should OWASP CRS be switched on? (owner 2026-10-23, rule: held-out precision AND recall
must not drop.)

Web access logs of the AIT-LDS mail servers (labelled attack steps): the current rules
(Sigma + own) alone vs. current rules OR CRS (paranoia level 1, inbound threshold 5 -
CRS defaults, nothing tuned). Development servers are reported; the held-out servers
decide. Per server and access log: attack steps caught (a step is caught when one of its
lines alerts), attack lines alerted, false alarms on sampled normal lines, and an
estimated precision (alerted attack lines / (alerted attack lines + false-alarm rate x
all normal lines)).

    python -m scripts.evaluate_crs [--rules D:/SplitShield-data/crs/v4.30.0/rules]
Results: results/extra/crs_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.extra_logs import AIT_HOSTS, NORMAL_SAMPLE, _scan_labels
from src.soc import crs, logformats
from utils.paths import RAW_DIR, RESULTS_DIR

OUT = RESULTS_DIR / "extra" / "crs_evaluation.json"


PARANOIA = [crs.PARANOIA]  # --paranoia (owner 2026-10-24: is PL2 worth it?)


def _crs_hit(line: str, rules: Path) -> bool:
    m = logformats._ACCESS.match(line)
    if not m:
        return False
    req = crs.request(m["method"], m["target"], m["proto"] or "", m["ua"] or "", m["ref"] or "")
    return bool(crs.check(req, rules, paranoia=PARANOIA[0])["attack"])


def run(which: str, rules: Path) -> dict[str, Any]:
    from src.soc.parsers import parse_line

    base = RAW_DIR / "ait-lds-v1.1"
    tot: dict[str, dict[str, Any]] = {k: {"steps": set(), "caught": set(), "att": 0, "att_hit": 0, "norm": 0, "norm_hit": 0, "norm_total": 0}
           for k in ("rules", "rules+crs")}  # fmt: skip
    files = []
    for host in AIT_HOSTS[which]:
        for lab in sorted((base / "labels" / host).rglob("*")):
            rel = lab.relative_to(base / "labels" / host).as_posix()
            if lab.is_dir() or "access" not in rel or "apache" not in rel:
                continue
            data = base / "data" / host / rel
            if not data.is_file():
                continue
            t0 = time.time()
            attacks, last, n_lines = _scan_labels(lab)
            span = min(n_lines, (last + 200_000) if last else 2_000_000)
            every = max(1, span // NORMAL_SAMPLE)
            n_norm_total = span - sum(1 for i in attacks if i < span)
            with open(data, encoding="utf-8", errors="replace") as df:
                for i, line in enumerate(df):
                    if i >= span:
                        break
                    step = attacks.get(i)
                    if step == "" or (step is None and i % every):
                        continue
                    ev = parse_line(line)
                    base_hit = bool(ev and ev["techniques"])
                    both_hit = base_hit or _crs_hit(line, rules)
                    for k, hit in (("rules", base_hit), ("rules+crs", both_hit)):
                        t = tot[k]
                        if step is None:
                            t["norm"] += 1
                            t["norm_hit"] += hit
                        else:
                            t["steps"].add((host, step))
                            t["att"] += 1
                            t["att_hit"] += hit
                            if hit:
                                t["caught"].add((host, step))
            for t in tot.values():
                t["norm_total"] += n_norm_total
            files.append({"host": host, "log": rel, "seconds": round(time.time() - t0)})
            print(f"{which} {host} {rel} ({round(time.time() - t0)} s)", flush=True)
    out = {}
    for k, t in tot.items():
        fp_rate = t["norm_hit"] / max(1, t["norm"])
        est_fp = fp_rate * t["norm_total"]
        out[k] = {"steps": len(t["steps"]), "steps_caught": len(t["caught"]),
                  "step_recall": round(len(t["caught"]) / max(1, len(t["steps"])), 4),
                  "attack_lines": t["att"], "attack_lines_alerted": t["att_hit"],
                  "line_recall": round(t["att_hit"] / max(1, t["att"]), 4),
                  "normal_sampled": t["norm"], "normal_alerted": t["norm_hit"],
                  "false_alarm_rate": round(fp_rate, 6),
                  "precision_est": round(t["att_hit"] / max(1e-9, t["att_hit"] + est_fp), 4)}  # fmt: skip
    return {"hosts": AIT_HOSTS[which], "files": files, "result": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", type=Path, default=crs.RULES_DIR)
    ap.add_argument("--paranoia", type=int, default=crs.PARANOIA)
    a = ap.parse_args()
    PARANOIA[0] = a.paranoia
    out_path = OUT if a.paranoia == crs.PARANOIA else OUT.with_name(f"crs_evaluation_pl{a.paranoia}.json")
    rules, skipped = crs.parse_rules(a.rules, a.paranoia)
    doc: dict[str, Any] = {"what": f"current rules vs current rules + OWASP CRS (PL{a.paranoia}, threshold 5) on AIT web access logs",
                           "crs_rules_used": len(rules), "crs_rules_skipped": skipped,
                           "rule": "switch CRS on only if held-out precision AND recall (steps and lines) do not drop"}  # fmt: skip
    for which in ("dev", "heldout"):
        doc[which] = run(which, a.rules)
    h = doc["heldout"]["result"]
    keep = (h["rules+crs"]["precision_est"] >= h["rules"]["precision_est"]
            and h["rules+crs"]["step_recall"] >= h["rules"]["step_recall"]
            and h["rules+crs"]["line_recall"] >= h["rules"]["line_recall"])  # fmt: skip
    doc["decision"] = "switch on" if keep else "keep off (held-out precision or recall would drop)"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1), "utf-8")
    print(json.dumps({k: doc[k]["result"] for k in ("dev", "heldout")} | {"decision": doc["decision"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
