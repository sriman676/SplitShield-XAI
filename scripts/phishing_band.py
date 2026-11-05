"""Re-choose the phishing "suspicious" band on VALIDATION (owner 2026-10-17).

The v4 model flags almost no real popular sites at its phishing threshold, so the old
band rule (lowest probability reached by <= 5 % of the Tranco validation sample) put
the band very low (0.268) and well-known sites showed as "suspicious". Owner rule:
at most 2 % of the Tranco VALIDATION sample (ranks 20,001-100,000) may be
"suspicious" or worse. The phishing threshold itself is unchanged (validated).
The model file is updated in place (``current/``); ``prev/`` keeps v3 untouched.
TEST numbers (Tranco 100,001-200,000 and PhiUSIIL test) are reported only.

    python -m scripts.phishing_band [--no-save]
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np

from scripts import phishing_threshold as pt
from scripts.improve_phishing import tranco
from src.mlareas import common, phishing
from utils.paths import RESULTS_DIR

BAND_CAP = 0.02  # owner 2026-10-17: <= 2 % of real popular sites "suspicious" or worse
OUT = RESULTS_DIR / "extra" / "phishing_band.json"


def choose_band(p_val_tranco: np.ndarray, threshold: float, cap: float = BAND_CAP) -> float:
    """Lowest probability s <= threshold that at most ``cap`` of the validation sites reach."""
    cands = np.sort(np.unique(p_val_tranco[p_val_tranco <= threshold]))
    for s in cands:
        if float((p_val_tranco >= s).mean()) <= cap:
            return float(s)
    return float(threshold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    b = phishing.load(refresh=True)
    thr, old_band = float(b["threshold"]), float(b.get("suspicious_at", phishing.SUSPICIOUS_AT))
    tdf = tranco()
    t_val, t_test = pt._sample(tdf, pt.VAL_RANKS, 0), pt._sample(tdf, pt.TEST_RANKS, 1)

    def prob(urls: list[str]) -> np.ndarray:
        return phishing._score(urls, b)[0]

    pv, ptt = prob(t_val), prob(t_test)
    band = choose_band(pv, thr)
    data = phishing.load_phiusiil()
    X, y, _n, keys, _s = phishing._variant_data(data, "url_host")
    _tr, _va, te = phishing.split_indices(y, int(b.get("seed", 0)))
    p_te = prob([keys[i] for i in te])
    spot = ["https://www.google.com", "https://www.microsoft.com", "https://github.com",
            "https://www.amazon.com", "https://www.paypa1-login.tk/x",
            "http://secure-update-micros0ft.xyz/login", "https://apple-id-verify.com"]  # fmt: skip
    doc = {
        "what": "phishing suspicious band re-chosen on Tranco VALIDATION (<= 2 % suspicious or worse)",
        "version": b.get("version"), "threshold_unchanged": thr,
        "band_before": old_band, "band_after": band,
        "validation_tranco_suspicious_or_worse": {"before": float((pv >= old_band).mean()),
                                                  "after": float((pv >= band).mean())},
        "test_report_only": {
            "tranco_100k_200k_suspicious_or_worse": {"before": float((ptt >= old_band).mean()),
                                                      "after": float((ptt >= band).mean())},
            "tranco_100k_200k_phishing": float((ptt >= thr).mean()),
            "phiusiil_test_at_threshold": phishing.binary_metrics(y[te], p_te, thr),
            "phiusiil_test_phishing_suspicious_or_worse": {
                "before": float((p_te[y[te] == 1] >= old_band).mean()),
                "after": float((p_te[y[te] == 1] >= band).mean())},
        },
        "blind_used": False,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }  # fmt: skip
    if not a.no_save:
        b2 = {k: v for k, v in b.items() if k != "_det"}
        b2["suspicious_at"] = band
        b2["suspicious_basis"] = f"lowest probability reached by <= {BAND_CAP:.0%} of Tranco validation (2026-10-17)"
        cur = common.area_dir(phishing.AREA) / "current"
        joblib.dump(b2, cur / "model.joblib")  # in place: prev/ keeps v3
        meta = {k: v for k, v in b2.items() if isinstance(v, str | int | float | dict)}
        (cur / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        b = phishing.load(refresh=True)
        doc["saved"] = True
    doc["spot_check"] = {r["url"]: f"{r['verdict']} {r['phishing_probability']:.3f}"
                         for r in phishing.check_urls(spot, b)}  # fmt: skip
    OUT.write_text(json.dumps(doc, indent=1, default=float), "utf-8")
    print(json.dumps(doc, indent=1, default=float))


if __name__ == "__main__":
    main()
