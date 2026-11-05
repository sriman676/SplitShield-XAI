"""H2c power analysis (owner 2026-10-24): how many stage-6 instances per seed are needed.

H2b (docs/PREREG_H2b.md) had 16 instances per seed: all 5 seeds had a positive median
paired difference (PGD drift above the random-matched control), only 2 reached p < 0.05.
This estimates the per-seed effect size from H2b's Wilcoxon p-values (z = Phi^-1(1 - p),
d ~ z / sqrt(n)) and simulates the registered rule (one-sided paired Wilcoxon, median > 0,
p < 0.05; verdict = at least 3 of 5 seeds) for candidate instance counts.

H2b is used here as a pilot only; H2c runs on new seeds.

    python -m scripts.power_h2c [--h2b results/extra/H2b.json] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, wilcoxon

from utils.paths import RESULTS_DIR

CANDIDATES = (16, 32, 48, 64, 96, 128, 192)


def effect_sizes(h2b: dict[str, Any]) -> list[float]:
    return [float(norm.isf(r["p"]) / sqrt(r["n_instances"])) for r in h2b["rows"]]


def seed_power(d: float, n: int, sims: int, rng: np.random.Generator) -> float:
    hits = 0
    for _ in range(sims):
        x = rng.normal(d, 1.0, n)
        hits += int(np.median(x) > 0 and wilcoxon(x, alternative="greater").pvalue < 0.05)
    return hits / sims


def majority(p: float) -> float:
    """P(at least 3 of 5 seeds supported) when each seed is supported with probability p."""
    return sum(comb(5, k) * p**k * (1 - p) ** (5 - k) for k in range(3, 6))


def run(h2b_path: Path, sims: int = 2000, seed: int = 0) -> dict[str, Any]:
    h2b = json.loads(h2b_path.read_text("utf-8"))
    ds = effect_sizes(h2b)
    rng = np.random.default_rng(seed)
    scenarios = {"mean d": float(np.mean(ds)), "conservative d (25th pct)": float(np.percentile(ds, 25)),
                 "smallest positive-seed d": float(min(d for d in ds if d > 0)),
                 "pessimistic d = 0.25 (about 2/3 of the pilot mean)": 0.25}  # fmt: skip
    table = []
    for n in CANDIDATES:
        row: dict[str, Any] = {"n": n}
        for name, d in scenarios.items():
            p = seed_power(d, n, sims, rng)
            row[name] = {"seed_power": round(p, 3), "verdict_power": round(majority(p), 3)}
        table.append(row)
    return {"pilot": "H2b (seeds 5-9, 16 instances each)", "per_seed_d": [round(d, 3) for d in ds],
            "scenarios": {k: round(v, 3) for k, v in scenarios.items()}, "simulations": sims,
            "rule": "one-sided paired Wilcoxon, median > 0 and p < 0.05 per seed; >= 3 of 5 seeds",
            "table": table}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2b", type=Path, default=RESULTS_DIR / "extra" / "H2b.json")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "extra" / "H2c_power.json")
    ap.add_argument("--sims", type=int, default=2000)
    a = ap.parse_args()
    res = run(a.h2b, a.sims)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), "utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
