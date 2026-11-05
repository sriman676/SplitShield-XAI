"""Choose and check the risk-board escalation weights on AIT-LDS (owner 2026-10-15).

Multi-source, labelled data: every AIT server has web, auth, mail, syslog and IDS
logs with each line labelled attack-step or normal. Per server, attack lines and an
even sample of normal lines are parsed and correlated twice - board off and on. An
incident is POSITIVE when an attack line of the same entity falls inside it.

At the alerting level "high or worse" (severity >= 6):
  precision = positive incidents / all high+ incidents
  recall    = positive incidents that reach high+ / all positive incidents
Weights (weight x other families, capped) are chosen on the DEV servers: the
highest recall whose precision is not below board-off. The choice is then checked on
the HELD-OUT servers; if precision drops there, the weight is set to 0 (the board then
only shows linked findings, never changes severity). Owner rule: kept only if
precision does not drop.

    python -m scripts.validate_riskboard
Writes results/extra/riskboard_validation.json (read by src/soc/riskboard.py).
"""

from __future__ import annotations

import json
import time
from typing import Any

from scripts import extra_logs as el
from src.soc.correlate import correlate, entity_of
from src.soc.parsers import parse_line
from src.soc.riskboard import Board
from utils.paths import RESULTS_DIR

HIGH = 6
# (weight, cap, follow_s): follow_s > 0 = victim follow (owner 2026-10-24)
FOLLOW = (0.0, 1800.0, 3600.0, 4 * 3600.0)
# (weight, cap, follow_s, hops, link_degree, link_failed); safer links 2026-10-24
GRID = [(w, c, f, h, d, lf) for w in (0.0, 1.0, 2.0) for c in (1.0, 2.0) for f in (0.0, 1800.0) for h in (0, 1)
        for d in (0, 3, 10) for lf in (True, False)
        if (w or (c == 1.0 and f == 0.0 and h == 1 and d == 0 and lf)) and (h or (d == 0 and lf))]  # fmt: skip


AIT_YEAR = 2020  # AIT-LDS v1.1 was recorded Feb-Mar 2020 (its dated web logs say so)


def server_events(host: str) -> list[dict[str, Any]]:
    """Parsed events (attack lines + sampled normal lines) of one AIT server. Year-less
    syslog lines are dated in the recording's year, as a live agent would date them."""
    from src.soc.parsers import assume_year

    with assume_year(AIT_YEAR):
        return _server_events(host)


def _server_events(host: str) -> list[dict[str, Any]]:
    base = el.RAW_DIR / "ait-lds-v1.1"
    out = []
    for lab in sorted((base / "labels" / host).rglob("*")):
        if lab.is_dir() or lab.suffix == ".zip":
            continue
        data = base / "data" / host / lab.relative_to(base / "labels" / host)
        if not data.is_file():
            continue
        attacks, last, n_lines = el._scan_labels(lab)
        span = min(n_lines, (last + 200_000) if last else 2_000_000)
        every = max(1, span // el.NORMAL_SAMPLE)
        with open(data, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= span:
                    break
                step = attacks.get(i)
                if step == "" or (step is None and i % every):
                    continue
                ev = parse_line(line)
                if ev is None:
                    continue
                ev.setdefault("observer", host)  # what an agent on this server records
                ev["_attack"] = step is not None
                out.append(ev)
    out.sort(key=lambda e: e["ts"])
    return out


def score(events: list[dict[str, Any]], board: Board | None) -> dict[str, Any]:
    incs = correlate(events, board=board)
    att: dict[str, list[float]] = {}
    for e in events:
        if e["_attack"]:
            att.setdefault(entity_of(e), []).append(e["ts"])

    def positive(inc: dict[str, Any]) -> bool:
        return any(inc["start"] <= t <= inc["end"] for t in att.get(inc["src_ip"], []))

    pos = [i for i in incs if positive(i)]
    high = [i for i in incs if i["severity"] >= HIGH]
    tp = sum(positive(i) for i in high)
    # entity-level truth (2026-10-24): AIT's attacker is a dedicated machine, so every
    # incident of an entity with labelled attack lines is the attacker's; reported beside
    # the window rule above (an attack line inside the incident's own time window)
    actor = set(att)
    pos_e = [i for i in incs if i["src_ip"] in actor]
    tp_e = sum(i["src_ip"] in actor for i in high)
    return {"incidents": len(incs), "positive": len(pos), "high": len(high), "high_positive": tp,
            "precision": tp / max(1, len(high)), "recall": tp / max(1, len(pos)),
            "entity_positive": len(pos_e), "entity_high_positive": tp_e,
            "entity_precision": tp_e / max(1, len(high)), "entity_recall": tp_e / max(1, len(pos_e)),
            "boosted": sum(1 for i in incs if i.get("board_boost"))}  # fmt: skip


def evaluate(
    hosts: list[str], cache: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for w, c, fo, hp, ld, lf in GRID:
        tot: dict[str, float] = {
            "high": 0,
            "high_positive": 0,
            "positive": 0,
            "boosted": 0,
            "entity_positive": 0,
            "entity_high_positive": 0,
        }
        for h in hosts:
            r = score(cache[h], Board(weight=w, cap=c, follow_s=fo, hops=hp, link_degree=ld,
                                      link_failed=lf).add_all(cache[h]) if w else None)  # fmt: skip
            for k in tot:
                tot[k] += r[k]
        tot["precision"] = tot["high_positive"] / max(1, tot["high"])
        tot["recall"] = tot["high_positive"] / max(1, tot["positive"])
        tot["entity_precision"] = tot["entity_high_positive"] / max(1, tot["high"])
        tot["entity_recall"] = tot["entity_high_positive"] / max(1, tot["entity_positive"])
        rows[f"w{w:g}_c{c:g}_f{fo:g}_h{hp}_d{ld}_l{int(lf)}"] = {"weight": w, "cap": c, "follow_s": fo, "hops": hp,
                                                              "link_degree": ld, "link_failed": lf} | tot  # fmt: skip
    return rows


def main() -> None:
    t0 = time.time()
    cache = {h: server_events(h) for hs in el.AIT_HOSTS.values() for h in hs}
    dev = evaluate(el.AIT_HOSTS["dev"], cache)
    off = dev["w0_c1_f0_h1_d0_l1"]
    # decided on ENTITY truth (owner 2026-10-24: AIT attackers are dedicated machines); the
    # window-rule numbers are reported beside it
    ok = [r for r in dev.values() if r["entity_precision"] >= off["entity_precision"] - 1e-12]
    best = max(ok, key=lambda r: (r["entity_recall"], -r["weight"], -r["cap"], -r["follow_s"], -r["hops"],
                                  r["link_failed"], -r["link_degree"]))  # fmt: skip
    held = evaluate(el.AIT_HOSTS["heldout"], cache)
    key = (f"w{best['weight']:g}_c{best['cap']:g}_f{best['follow_s']:g}_h{best['hops']}"
           f"_d{best['link_degree']}_l{int(best['link_failed'])}")
    h_on, h_off = held[key], held["w0_c1_f0_h1_d0_l1"]
    kept = h_on["entity_precision"] >= h_off["entity_precision"] - 1e-12
    chosen = {"weight": best["weight"] if kept else 0.0, "cap": best["cap"],
              "follow_s": best["follow_s"] if kept else 0.0, "hops": best["hops"],
              "link_degree": best["link_degree"], "link_failed": best["link_failed"]}
    out = {
        "what": "risk-board escalation chosen on AIT dev servers, checked on held-out",
        "dev_hosts": el.AIT_HOSTS["dev"], "heldout_hosts": el.AIT_HOSTS["heldout"],
        "alert_level": f"severity >= {HIGH}",
        "events": {h: len(v) for h, v in cache.items()},
        "dev": dev, "heldout": held,
        "chosen_on_dev": {k: best[k] for k in ("weight", "cap", "follow_s", "hops", "link_degree", "link_failed")},
        "heldout_check": {"board_off": h_off, "board_on": h_on, "precision_kept": kept},
        "chosen": chosen,
        "seconds": round(time.time() - t0),
    }  # fmt: skip
    (RESULTS_DIR / "extra").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "extra" / "riskboard_validation.json").write_text(
        json.dumps(out, indent=1), "utf-8"
    )
    print(
        json.dumps(
            {k: out[k] for k in ("chosen_on_dev", "heldout_check", "chosen")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
