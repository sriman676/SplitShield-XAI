"""Re-choose the phishing alert threshold on VALIDATION data that includes real
popular websites (owner 2026-10-15).

Why: the blind test (OpenPhish + Tranco top-20k) flagged 12.6 % of popular sites,
because PhiUSIIL's legitimate hosts do not look like real top sites. The model is
unchanged; only its threshold is re-chosen, on validation data only:

  phishing      PhiUSIIL validation hosts (label 1)
  legitimate    PhiUSIIL validation legit hosts + a seeded sample of Tranco ranks
                20,001-100,000 (ranks <= 20,000 were the blind pool: never used)
  rule          highest validation F1 whose false-positive rate on the Tranco
                validation sample is <= FPR_CAP
  band          the "suspicious" tier starts at the lowest probability that at most
                SUSPICIOUS_CAP of the Tranco validation sample reaches (it used to be
                a fixed 0.2, which made the band 0.2-0.95 once the threshold rose)

Reported only (never used to choose): PhiUSIIL TEST metrics of the rebuilt seed-0
split, and the false-positive rate on a separate Tranco ranks 100,001-200,000 sample.
The blind OpenPhish/Tranco set is not re-scored. The new threshold is saved as a new
bundle version (the previous one is kept as ``prev``).

    python -m scripts.phishing_threshold [--no-save]
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

import numpy as np
import pandas as pd

from src.mlareas import common, phishing
from utils.paths import RESULTS_DIR

FPR_CAP = 0.025
SUSPICIOUS_CAP = (
    0.05  # at most 5 % of real popular sites may show as "suspicious" or worse
)
ORIGINAL_THRESHOLD = 0.5  # deployed 2026-10-15 morning; always shown as the baseline
VAL_RANKS, TEST_RANKS = (20_001, 100_000), (100_001, 200_000)
N_TRANCO = 10_000
OUT = RESULTS_DIR / "extra"


def tranco() -> pd.DataFrame:
    """The saved Tranco snapshot (data/raw/tranco/top-1m.csv, 2026-10-15). 2026-10-23: this
    used to download the live list, which changes daily: the threshold then depended on the
    run date, and sites in the blind pool (ranks 1-20,000 of the snapshot) could move into
    the validation ranks."""
    from scripts.improve_phishing import tranco as snapshot

    return snapshot()


def _sample(df: pd.DataFrame, ranks: tuple[int, int], seed: int) -> list[str]:
    pool = (
        df[(df["rank"] >= ranks[0]) & (df["rank"] <= ranks[1])]["domain"]
        .astype(str)
        .tolist()
    )
    random.Random(seed).shuffle(pool)
    return [f"https://{d}" for d in pool[:N_TRANCO]]


def _proba(urls: list[str], b: dict[str, Any]) -> np.ndarray:
    p, _ = phishing._score(urls, b)
    return p


def choose(p_val: np.ndarray, y_val: np.ndarray, p_tr: np.ndarray) -> dict[str, Any]:
    """Validation grid over every distinct calibrated probability."""
    cands = np.unique(np.concatenate([p_val, p_tr, [0.5]]))
    rows = []
    y_all = np.concatenate([y_val, np.zeros(len(p_tr), int)])
    p_all = np.concatenate([p_val, p_tr])
    for t in cands:
        if t <= phishing.SUSPICIOUS_AT:
            continue  # keep the 'suspicious' band below the alert threshold
        m = phishing.binary_metrics(y_all, p_all, float(t))
        rows.append({"threshold": float(t), "f1": m["f1"], "precision": m["precision"],
                     "recall": m["recall"], "fpr_tranco_val": float((p_tr >= t).mean()),
                     "fpr_phiusiil_val": float((p_val[y_val == 0] >= t).mean())})  # fmt: skip
    ok = [r for r in rows if r["fpr_tranco_val"] <= FPR_CAP]
    best = max(ok or rows, key=lambda r: (r["f1"], -r["threshold"]))
    return {"chosen": best, "constraint_met": bool(ok), "grid_size": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    b = phishing.load(refresh=True)
    # rebuild the seed-0 split exactly as training did (for TEST reporting only)
    data = phishing.load_phiusiil()
    _X, y, _names, keys, _stats = phishing._variant_data(data, "url_host")
    _tr, va, te = phishing.split_indices(y, int(b.get("seed", 0)))
    if [keys[i] for i in va] != list(b["val_hosts"]):
        raise RuntimeError("rebuilt validation split differs from the deployed bundle")
    test_hosts, y_test = [keys[i] for i in te], y[te]
    tr_df = tranco()
    t_val, t_test = _sample(tr_df, VAL_RANKS, 0), _sample(tr_df, TEST_RANKS, 1)

    p_val = _proba(list(b["val_hosts"]), b)
    y_val = np.asarray(b["val_y"]).astype(int)
    p_tval, p_ttest = _proba(t_val, b), _proba(t_test, b)
    p_test = _proba(test_hosts, b)
    pick = choose(p_val, y_val, p_tval)
    thr_new, thr_cur = pick["chosen"]["threshold"], float(b["threshold"])
    sus_cands = sorted(float(x) for x in np.unique(p_tval) if x <= thr_new)
    sus_new = next(
        (s for s in sus_cands if (p_tval >= s).mean() <= SUSPICIOUS_CAP), thr_new
    )
    sus_cur = float(b.get("suspicious_at", phishing.SUSPICIOUS_AT))

    def report(t: float) -> dict[str, Any]:
        return {
            "validation_phiusiil": phishing.binary_metrics(y_val, p_val, t),
            "validation_tranco_fpr": float((p_tval >= t).mean()),
            "test_phiusiil": phishing.binary_metrics(y_test, p_test, t),
            "test_tranco_100k_200k_fpr": float((p_ttest >= t).mean()),
        }

    old, new = report(ORIGINAL_THRESHOLD), report(thr_new)
    cur = report(thr_cur)

    def band(s: float) -> dict[str, float]:
        return {"suspicious_at": s,
                "tranco_val_suspicious_or_worse": float((p_tval >= s).mean()),
                "tranco_test_suspicious_or_worse": float((p_ttest >= s).mean()),
                "phiusiil_val_legit_suspicious_or_worse": float((p_val[y_val == 0] >= s).mean()),
                "phiusiil_val_phishing_suspicious_or_worse": float((p_val[y_val == 1] >= s).mean())}  # fmt: skip

    # champion/challenger on the validation objective (F1 with the FPR cap)
    def objective(r: dict[str, Any]) -> float:
        return (
            r["validation_phiusiil"]["f1"]
            if r["validation_tranco_fpr"] <= FPR_CAP
            else -1.0
        )

    decision = common.champion_challenger(objective(cur), objective(new), 0.0, 0.0,
                                          "val_f1_with_tranco_fpr_cap")  # fmt: skip
    doc = {"rule": f"max validation F1 s.t. Tranco-validation FPR <= {FPR_CAP:.1%}",
           "tranco_validation_ranks": VAL_RANKS, "tranco_test_ranks": TEST_RANKS,
           "tranco_per_sample": N_TRANCO, "blind_pool_ranks_excluded": [1, 20_000],
           "old_threshold": ORIGINAL_THRESHOLD, "new_threshold": thr_new, "choice": pick,
           "old": old, "new": new, "decision": decision, "saved": False,
           "suspicious_band": {"old": band(phishing.SUSPICIOUS_AT), "new": band(sus_new),
                               "rule": f"lowest probability reached by <= {SUSPICIOUS_CAP:.0%} "
                                       "of the Tranco validation sample"}}  # fmt: skip
    changed = thr_new != thr_cur or sus_new != sus_cur
    if decision["promoted"] and changed and not a.no_save:
        cand = {k: v for k, v in b.items() if k != "_det"}
        cand.update(threshold=thr_new, suspicious_at=sus_new,
                    version=int(b.get("version", 1)) + 1,
                    threshold_basis=doc["rule"], metrics=new["test_phiusiil"])  # fmt: skip
        common.save_bundle(phishing.AREA, cand)
        phishing.load(refresh=True)
        doc["saved"] = True
        doc["version"] = cand["version"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phishing_threshold.json").write_text(
        json.dumps(doc, indent=1, default=float), "utf-8"
    )
    write_md(doc)
    print(json.dumps({k: doc[k] for k in ("old_threshold", "new_threshold", "saved")}))


def write_md(d: dict[str, Any]) -> None:
    def row(name: str, r: dict[str, Any]) -> str:
        v, t = r["validation_phiusiil"], r["test_phiusiil"]
        return (f"| {name} | {v['f1']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | "
                f"{r['validation_tranco_fpr']:.1%} | {t['f1']:.3f} | {t['precision']:.3f} | "
                f"{t['recall']:.3f} | {t['fpr_legit']:.1%} | {r['test_tranco_100k_200k_fpr']:.1%} |")  # fmt: skip

    lines = [
        "# 🎣 Phishing threshold re-chosen on validation (owner 2026-10-15)", "",
        "The model is unchanged; only the alert threshold was re-chosen. Rule: "
        f"{d['rule']}. Phishing = PhiUSIIL validation hosts; legitimate = PhiUSIIL "
        f"validation legit + {d['tranco_per_sample']:,} Tranco domains, ranks "
        f"{d['tranco_validation_ranks'][0]:,}-{d['tranco_validation_ranks'][1]:,}. Ranks "
        "1-20,000 (the blind test's pool) were not used. Test columns are reported only.",
        "", "| threshold | val F1 | val precision | val recall | FPR Tranco val | "
        "test F1 | test precision | test recall | test FPR (PhiUSIIL legit) | "
        "FPR Tranco 100k-200k (test) |", "|---|---|---|---|---|---|---|---|---|---|",
        row(f"old {d['old_threshold']:.3f}", d["old"]),
        row(f"new {d['new_threshold']:.3f}", d["new"]), "",
        f"Constraint met: {d['choice']['constraint_met']}. Decision: "
        f"{d['decision']['decision']}; saved as a new version: {d['saved']}.", "",
        "**Suspicious band** (shown to the user, no alert): "
        f"{d['suspicious_band']['rule']}. Old start {d['suspicious_band']['old']['suspicious_at']:.2f} "
        f"-> new {d['suspicious_band']['new']['suspicious_at']:.3f}. Popular sites shown as "
        "suspicious-or-worse (Tranco 100k-200k, test): "
        f"{d['suspicious_band']['old']['tranco_test_suspicious_or_worse']:.1%} -> "
        f"{d['suspicious_band']['new']['tranco_test_suspicious_or_worse']:.1%}; PhiUSIIL "
        "validation phishing at suspicious-or-worse: "
        f"{d['suspicious_band']['old']['phiusiil_val_phishing_suspicious_or_worse']:.1%} -> "
        f"{d['suspicious_band']['new']['phiusiil_val_phishing_suspicious_or_worse']:.1%}.", "",
        "The blind OpenPhish + Tranco top-20k result (recall 0.89, 12.6 % of popular "
        "sites flagged) stays as registered; it is not re-scored. Any later check on "
        "that set is a follow-up, not blind.",
    ]  # fmt: skip
    (OUT / "phishing_threshold.md").write_text("\n".join(lines) + "\n", "utf-8")


if __name__ == "__main__":
    main()
