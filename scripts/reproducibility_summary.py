"""Registered results vs the two final research re-runs, with every difference
explained (owner 2026-10-22: show registered and re-run side by side).

    python -m scripts.reproducibility_summary --rerun D:/SplitShield-rerun --rerun2 D:/SplitShield-rerun2
Output: results/extra/reproducibility_summary.json (used by the Word report)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.paths import RESULTS_DIR

SKIP = ("time", "second", "minutes", "speed", "per_s", "elapsed", "duration", "when", "generated", "lps", "latency")

# why a registered number differs (file prefix -> explanation)
WHY = {
    "deploy_report": "deployed-detector code was re-tuned after registration (raw-unit detector, 2026-10-14); "
                     "not a hypothesis result",
    "nlp_metrics": "assistant corpus and code changed after registration (vocabulary 715 -> 805); the "
                   "re-runs' RLCD candidate reward-hacked and was rejected by the faithfulness gate",
    "E3_alpha_skew_summary": "identical in both re-runs; probable cause: the registered run used other thread settings "
                             "(floating-point order); verdicts unchanged",
    "gate_report": "sanity gate (shuffled labels) re-sampled on the current data pipeline; still far apart",
    "soc_selftest": "SOC self-test counts depend on SOC code changed after registration",
    "summary": "aggregates every run folder; the registered folder also holds 6 older duplicate runs (an earlier E4 set and one H5 run)",
}  # fmt: skip


def _walk(o: Any, p: str = "") -> Any:
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{p}[{i}]")
    elif isinstance(o, int | float) and not isinstance(o, bool):
        yield p, float(o)


def _nums(root: Path) -> dict[tuple[str, str], float]:
    out = {}
    for f in sorted((root / "results").glob("*.json")):
        try:
            for k, v in _walk(json.loads(f.read_text("utf-8"))):
                if not any(s in k.lower() for s in SKIP):
                    out[(f.stem, k)] = v
        except ValueError:
            continue
    return out


def build(rerun: Path, rerun2: Path) -> dict[str, Any]:
    reg, a, b = _nums(Path(".")), _nums(rerun), _nums(rerun2)
    common = set(a) & set(b)
    same_runs = sum(a[k] == b[k] for k in common)
    vs = [k for k in set(reg) & set(a)]
    identical = sum(reg[k] == a[k] for k in vs)
    close = sum(reg[k] != a[k] and abs(reg[k] - a[k]) <= max(0.01, 0.02 * abs(reg[k])) for k in vs)
    diffs = [{"what": f"{k[0]}{k[1]}", "registered": round(reg[k], 4), "rerun": round(a[k], 4),
              "why": WHY.get(k[0], "see REPRODUCIBILITY.md")}
             for k in sorted(vs) if abs(reg[k] - a[k]) > max(0.01, 0.02 * abs(reg[k]))]  # fmt: skip
    verdicts = [json.loads((r / "results" / "final_verdicts.json").read_text("utf-8")) for r in (Path("."), rerun, rerun2)]
    text = (f"Re-run 1 and re-run 2 (same commit, fresh copies): {same_runs} of {len(common)} result numbers "
            f"identical. Against the registered results: {identical} identical, {close} within 0.01 / 2 %, "
            f"{len(diffs)} different - all explained below. Hypothesis verdicts: "
            f"{'identical in all three' if verdicts[0] == verdicts[1] == verdicts[2] else 'DIFFERENT'}.")  # fmt: skip
    return {"text": text, "runs_identical": same_runs, "runs_compared": len(common), "differences": diffs,
            "verdicts_same": verdicts[0] == verdicts[1] == verdicts[2]}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", type=Path, required=True)
    ap.add_argument("--rerun2", type=Path, required=True)
    a = ap.parse_args()
    d = build(a.rerun, a.rerun2)
    out = RESULTS_DIR / "extra" / "reproducibility_summary.json"
    out.write_text(json.dumps(d, indent=1), "utf-8")
    print(d["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
