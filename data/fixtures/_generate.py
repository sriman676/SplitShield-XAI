"""Generate committed synthetic fixture CSVs (04_DATA §6).

Tiny, deterministic, schema-shaped data with deliberate duplicates, NaN, Inf,
and all 9 families present so the cleaning steps (Task 1) are actually exercised
by pytest, CI, and ``--fast``. Run once; the emitted CSV is committed.

    python data/fixtures/_generate.py

This is a mini-schema, NOT the full ~91-column CIC schema — the real schema is
unverified at kit time (11_LIMITATIONS §3) and schema-matching fixtures are
expanded in Task 1 once the download is verified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = {
    "BENIGN": 90,
    "DoS": 25,
    "DDoS": 25,
    "PortScan": 20,
    "BruteForce": 15,
    "WebAttack": 12,
    "Bot": 10,
    "Infiltration": 8,
    "Heartbleed": 8,
}

# Mini-schema spanning all four mutability classes for later constraint tests.
COLUMNS = [
    "Destination Port",  # IMMUTABLE
    "Flow Duration",  # INCREASE_ONLY
    "Total Fwd Packets",  # INCREASE_ONLY
    "Total Backward Packets",  # INCREASE_ONLY
    "Fwd Header Length",  # INCREASE_ONLY
    "Flow IAT Mean",  # FREE_BOUNDED
    "Idle Mean",  # FREE_BOUNDED
    "Flow Bytes/s",  # DERIVED
    "Flow Packets/s",  # DERIVED
]


def build() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for label, n in FAMILIES.items():
        for _ in range(n):
            fwd = rng.integers(1, 500)
            bwd = rng.integers(0, 500)
            dur = rng.integers(1, 1_000_000)
            rows.append(
                {
                    "Destination Port": int(rng.choice([80, 443, 22, 21, 53, 8080])),
                    "Flow Duration": int(dur),
                    "Total Fwd Packets": int(fwd),
                    "Total Backward Packets": int(bwd),
                    "Fwd Header Length": int(fwd * 20),
                    "Flow IAT Mean": float(rng.uniform(0, 1e5)),
                    "Idle Mean": float(rng.uniform(0, 1e5)),
                    "Flow Bytes/s": float((fwd + bwd) * 1e6 / dur),
                    "Flow Packets/s": float((fwd + bwd) * 1e6 / dur),
                    "Label": label,
                }
            )
    df = pd.DataFrame(rows, columns=COLUMNS + ["Label"])

    # Deliberate defects for the cleaning steps to catch.
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # exact duplicates
    df.loc[10, "Flow IAT Mean"] = np.nan  # NaN
    df.loc[11, "Flow Bytes/s"] = np.inf  # +Inf
    df.loc[12, "Flow Packets/s"] = -np.inf  # -Inf
    return df


def main() -> None:
    df = build()
    out = Path(__file__).resolve().parent / "fixture.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {df['Label'].nunique()} classes)")


if __name__ == "__main__":
    main()
