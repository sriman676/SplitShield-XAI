"""Model fit report (owner 2026-10-16): verdict rules, the controller summary, and the
improvement helpers added with it (phishing host2 features, ICS per-sensor
normalisation, malware precision-keeping threshold)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import fitcheck


@pytest.mark.parametrize(
    ("train", "val", "test", "verdict"),
    [
        (0.97, 0.96, 0.96, "good fit"),
        (1.00, 0.90, 0.91, "overfitting"),
        (0.55, 0.52, 0.50, "underfitting"),
        (0.95, 0.94, 0.70, "split shift"),
    ],
)
def test_split_verdict(train: float, val: float, test: float, verdict: str) -> None:
    assert fitcheck.split_verdict(train, val, test)["verdict"] == verdict


def _pts(train: list[float], val: list[float]) -> list[dict[str, float]]:
    return [{"n": 1000 * 2**i, "train": t, "val": v} for i, (t, v) in enumerate(zip(train, val, strict=True))]


def test_curve_verdict() -> None:
    assert fitcheck.curve_verdict(_pts([0.99, 0.98, 0.98], [0.90, 0.95, 0.955]))["verdict"] == "good fit"
    assert fitcheck.curve_verdict(_pts([0.95, 0.94, 0.94], [0.80, 0.85, 0.90]))["verdict"] == "data-limited"
    assert fitcheck.curve_verdict(_pts([1.0, 1.0, 1.0], [0.85, 0.86, 0.86]))["verdict"] == "overfitting"
    assert fitcheck.curve_verdict(_pts([0.6, 0.6, 0.6], [0.5, 0.55, 0.555]))["verdict"] == "underfitting"
    assert fitcheck.curve_verdict([])["verdict"] == "unknown"


def test_model_report_for_the_controller(tmp_path: Path) -> None:
    doc = {"generated": "2026-10-16 23:00", "models": {
        "zeek": {"split": fitcheck.split_verdict(0.99, 0.98, 0.99),
                 "curve": fitcheck.curve_verdict(_pts([0.99, 0.99], [0.97, 0.975]))},
        "ics": {"split": fitcheck.split_verdict(0.03, 0.03, 0.24),
                "curve": fitcheck.curve_verdict(_pts([0.02, 0.03], [0.02, 0.03]))},
        "netflow": {"error": "dataset missing"}}}  # fmt: skip
    p = tmp_path / "fit_report.json"
    p.write_text(json.dumps(doc), "utf-8")
    r = fitcheck.model_report(p)
    assert r["flagged"] == ["ics"] and r["models"]["zeek"]["verdict"] == "good fit"
    assert "underfitting" in r["models"]["ics"]["verdict"] and r["age_days"] is not None
    assert fitcheck.model_report(tmp_path / "none.json")["flagged"] == []


def test_phishing_host2_features_keep_host_mode_unchanged() -> None:
    from src.mlareas import phishing as ph

    urls = ["https://www.paypa1-login.tk/x", "google.com", "micros0ft-support.com"]
    assert ph.featurize(urls, "host").shape == (3, len(ph.feature_names("host"))) == (3, 273)
    X = ph.featurize(urls, "host2")
    names = ph.feature_names("host2")
    assert X.shape == (3, len(names)) and len(set(names)) == len(names)
    col = names.index("brand_edit_dist")
    assert X[0, col] == 1 and X[2, col] == 1  # one edit from paypal / microsoft
    assert X[1, col] == 5  # google.com is the brand itself
    assert X[0, names.index("tld_abused")] == 1 and X[1, names.index("tld_common")] == 1
    assert ph._edit_distance("paypal", "paypa1") == 1 and ph._edit_distance("abc", "abcdefghij") == 5


def test_ics_per_sensor_normalisation_equalises_sensors() -> None:
    from src.mlareas import ics

    class FakeNet:  # per-sensor errors fixed by the data: sensor 1 is 100x noisier
        pass

    rng = np.random.default_rng(0)
    arr = np.abs(rng.normal(size=(5000, 3))).astype(np.float32) * np.array([1.0, 100.0, 1.0], np.float32)
    det = ics.Detector("autoencoder", FakeNet(), np.ones(3, np.float32), np.zeros(3, np.float32))
    det.sensor_dev = lambda a, idx: a[idx]  # type: ignore[method-assign, assignment]
    new = ics.rescale_per_sensor(det, arr, np.arange(5000), 0.99)
    new.sensor_dev = lambda a, idx: a[idx] / new.norm  # type: ignore[method-assign, assignment]
    q = np.quantile(new.sensor_dev(arr, np.arange(5000)), 0.99, axis=0)
    assert np.allclose(q, 1.0, atol=0.05)  # every sensor's 99th percentile is now 1
    iso = ics.Detector("isolation_forest", None, np.ones(3), np.zeros(3))
    assert ics.rescale_per_sensor(iso, arr, np.arange(10), 0.99) is iso


def test_malware_threshold_keeps_validation_precision() -> None:
    from scripts.improve_malware import at_threshold, threshold_keep_precision

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    t = threshold_keep_precision(s, y, 0.01, 0.99)
    m = at_threshold(s, y, t)
    assert m["precision"] >= 0.99 and m["fpr"] <= 0.01
    # the lowest such threshold: one score lower would break the precision floor
    lower = np.max(s[s < t])
    assert at_threshold(s, y, float(lower))["precision"] < 0.99 or at_threshold(s, y, float(lower))["fpr"] > 0.01
