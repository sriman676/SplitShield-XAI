"""Self-evolving detection loop (owner 2026-10-15: "find alerts better and faster
with every iteration").

Each iteration proposes ONE change (a challenger), measures it on a fixed labelled
benchmark and keeps it only if it is better than the current setup (champion):

  * detection   - attack events caught must not drop,
  * false alarms - alerts on normal events must not rise,
  * speed        - events scored per second,

and at least one of them must improve. Every decision is logged
(``artifacts/evolve/history.jsonl``) so the dashboard can show the curve.

Challengers:
  1. suppress a noisy rule: one that fires on normal benchmark events (or was
     marked "false alarm" by analysts) but never on an attack;
  2. a parse cache for repeated log lines (identical lines are scored once).

The benchmark (``artifacts/evolve/benchmark.jsonl``) is a compact labelled sample
of held-out attack and normal log lines; it is kept when the big datasets are
removed, so the loop also works from a pendrive copy.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT

DIR = PROJECT_ROOT / "artifacts" / "evolve"
BENCH = DIR / "benchmark.jsonl"
STATE = DIR / "state.json"
HISTORY = DIR / "history.jsonl"


# --- settings applied by the parsers --------------------------------------------------


def state() -> dict[str, Any]:
    try:
        return json.loads(STATE.read_text("utf-8"))
    except (OSError, ValueError):
        return {"suppressed": [], "cache": 0, "iteration": 0}


def suppressed() -> set[str]:
    return set(state().get("suppressed", []))


def _save_state(st: dict[str, Any]) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1), "utf-8")


# --- benchmark --------------------------------------------------------------------------


def build_benchmark(per_source: int = 1500, seed: int = 0) -> int:
    """Labelled sample from the held-out AIT servers (attack lines per step + normal
    lines) and Loghub (normal server traffic). Built once, before datasets go."""
    from utils.paths import RAW_DIR

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    ait = RAW_DIR / "ait-lds-v1.1"
    for host in ("mail.onion.com", "mail.spiral.com"):
        for lab in sorted((ait / "labels" / host).rglob("*")):
            if lab.is_dir() or lab.suffix == ".zip" or "audit" in lab.parts:
                continue
            data = ait / "data" / host / lab.relative_to(ait / "labels" / host)
            if not data.is_file():
                continue
            attack: list[dict[str, Any]] = []
            normal: list[dict[str, Any]] = []
            with (
                open(lab, encoding="utf-8", errors="replace") as lf,
                open(data, encoding="utf-8", errors="replace") as df,
            ):
                for lab_line, line in zip(lf, df, strict=False):
                    parts = lab_line.strip().split(",")
                    if parts == ["0", "0"]:
                        if rng.random() < 0.02:
                            normal.append(
                                {"line": line.rstrip("\n"), "attack": False, "step": ""}
                            )
                    elif len(parts) > 1 and parts[1] != "0":
                        attack.append(
                            {
                                "line": line.rstrip("\n"),
                                "attack": True,
                                "step": parts[1],
                            }
                        )
            rng.shuffle(attack)
            rng.shuffle(normal)
            rows += attack[: per_source // 3] + normal[:per_source]
    loghub = RAW_DIR / "loghub"
    # Apache only: the Loghub SSH / Linux logs contain real internet attacks, so
    # they are not clean normal traffic
    for f in sorted(loghub.rglob("Apache.log")):
        lines = f.read_text("utf-8", errors="replace").splitlines()
        rows += [
            {"line": ln, "attack": False, "step": ""}
            for ln in rng.sample(lines, min(500, len(lines)))
        ]
    DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCH, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return len(rows)


def _bench() -> list[dict[str, Any]]:
    if not BENCH.is_file():
        return []
    return [
        json.loads(ln) for ln in BENCH.read_text("utf-8").splitlines() if ln.strip()
    ]


# --- measuring ---------------------------------------------------------------------------


def measure(st: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detection / false alarms / speed of the setup ``st`` on the benchmark."""
    from src.soc import parsers

    old = state()
    _save_state(st)
    parsers.reset_cache()
    try:
        t0 = time.perf_counter()
        hits = []
        rule_fp: Counter[str] = Counter()
        rule_tp: Counter[str] = Counter()
        for r in rows:
            ev = parsers.parse_line(r["line"])
            alert = bool(ev and ev["techniques"])
            hits.append(alert)
            if alert and ev:
                rule = _rule_of(ev)
                (rule_tp if r["attack"] else rule_fp)[rule] += 1
        secs = time.perf_counter() - t0
    finally:
        _save_state(old)
        parsers.reset_cache()
    att = [h for h, r in zip(hits, rows, strict=True) if r["attack"]]
    nor = [h for h, r in zip(hits, rows, strict=True) if not r["attack"]]
    steps = {r["step"] for r in rows if r["attack"]}
    caught = {r["step"] for h, r in zip(hits, rows, strict=True) if r["attack"] and h}
    return {
        "attack_lines_caught": sum(att) / max(1, len(att)),
        "steps_caught": len(caught) / max(1, len(steps)),
        "false_alarm_rate": sum(nor) / max(1, len(nor)),
        "events_per_second": len(rows) / max(secs, 1e-9),
        "ms_per_event": 1000 * secs / max(1, len(rows)),
        "rule_fp": dict(rule_fp),
        "rule_tp": dict(rule_tp),
    }


def _rule_of(ev: dict[str, Any]) -> str:
    d = str(ev.get("detail", ""))
    return (
        "sigma:" + d.split(" | Sigma: ")[1].split(" [host")[0]
        if " | Sigma: " in d
        else "kind:" + str(ev["kind"])
    )


def _better(new: dict[str, Any], old: dict[str, Any]) -> tuple[bool, str]:
    if new["steps_caught"] < old["steps_caught"] - 1e-9:
        return False, "fewer attack steps caught"
    if new["attack_lines_caught"] < old["attack_lines_caught"] - 0.002:
        return False, "fewer attack lines caught"
    if new["false_alarm_rate"] > old["false_alarm_rate"] + 1e-9:
        return False, "more false alarms"
    gains = []
    if new["false_alarm_rate"] < old["false_alarm_rate"] - 1e-9:
        gains.append("fewer false alarms")
    if new["attack_lines_caught"] > old["attack_lines_caught"] + 1e-9:
        gains.append("more attacks caught")
    if new["events_per_second"] > old["events_per_second"] * 1.05:
        gains.append("faster")
    return (bool(gains), ", ".join(gains) or "no improvement")


# --- the loop ----------------------------------------------------------------------------


def _analyst_false_alarms() -> Counter[str]:
    """Rules analysts marked as false alarms (RCL reviews)."""
    from src.soc.rcl import RCL

    c: Counter[str] = Counter()
    for r in RCL().rows():
        if r.get("verdict") is False or r.get("real") is False:
            c[_rule_of({"detail": r.get("detail", ""), "kind": r.get("kind", "")})] += 1
    return c


def iterate(max_challengers: int = 5) -> dict[str, Any]:
    """One evolution step: try challengers in turn, keep the first that is better."""
    rows = _bench()
    if not rows:
        return {
            "status": "no benchmark (run evolve.build_benchmark() while datasets exist)"
        }
    champ = state()
    base = measure(champ, rows)
    challengers: list[tuple[str, dict[str, Any]]] = []
    if not champ.get("cache"):
        challengers.append(
            ("parse cache for repeated lines", {**champ, "cache": 50_000})
        )
    noisy = Counter(base["rule_fp"])
    noisy.update(_analyst_false_alarms())
    for rule, n in noisy.most_common():
        if rule in champ.get("suppressed", []) or base["rule_tp"].get(rule):
            continue  # a rule that ever catches an attack is never suppressed
        challengers.append((f"suppress noisy rule {rule} ({n} false alarms)",
                            {**champ, "suppressed": [*champ.get("suppressed", []), rule]}))  # fmt: skip
    decision: dict[str, Any] = {"ts": time.time(), "iteration": champ.get("iteration", 0) + 1,
                                "champion": {k: v for k, v in base.items() if not k.startswith("rule_")}}  # fmt: skip
    for name, ch in challengers[:max_challengers]:
        res = measure(ch, rows)
        ok, why = _better(res, base)
        decision.setdefault("tried", []).append({"challenger": name, "kept": ok, "why": why,
            **{k: v for k, v in res.items() if not k.startswith("rule_")}})  # fmt: skip
        if ok:
            ch["iteration"] = decision["iteration"]
            _save_state(ch)
            decision["kept"] = name
            break
    DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(decision) + "\n")
    return decision


def history() -> list[dict[str, Any]]:
    if not HISTORY.is_file():
        return []
    return [
        json.loads(ln) for ln in HISTORY.read_text("utf-8").splitlines() if ln.strip()
    ]


def _path() -> Path:
    return DIR
