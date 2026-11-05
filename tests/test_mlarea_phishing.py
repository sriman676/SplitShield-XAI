"""Phishing URL area: features, SOC API shape, champion/challenger (owner 2026-10-15).

Tiny synthetic URLs only; every model is written to ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.mlareas import common, phishing

WORDS = ["river", "maple", "orbit", "sunny", "harbor", "pixel", "cedar", "lumen",
         "atlas", "nova", "quartz", "willow", "ember", "delta", "fjord", "tango"]  # fmt: skip


def _synthetic(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a, b = rng.choice(WORDS, 2)
        if i % 2:
            url = f"http://{rng.choice(phishing.BRANDS)}-login-{i}{a}.{b}-verify.xyz/x"
            rows.append((url, 0))
        else:
            rows.append((f"https://www.{a}{b}{i}.com", 1))
    df = pd.DataFrame(rows, columns=["URL", "label"])
    df["URLLength"] = df["URL"].str.len()
    df["y"] = (df["label"] == 0).astype(np.int8)
    return df


@pytest.fixture()
def trained(tmp_path: Path) -> dict[str, Any]:
    res = phishing.train(df=_synthetic(), seeds=(0,), families=("logreg",),
                         variants=("url_host",), root=tmp_path)  # fmt: skip
    return {"root": tmp_path, "res": res}


def test_split_url_strips_scheme_www_port_path() -> None:
    host, rest = phishing.split_url("HTTPS://user@www.Example.co.uk:8443/a/b?q=1")
    assert host == "example.co.uk"
    assert rest == "a/b?q=1"
    assert phishing.registered_domain("a.b.example.co.uk") == "example.co.uk"
    assert phishing.registered_domain("10.0.0.1") == "10.0.0.1"


def test_host_features_flags() -> None:
    feats = dict(zip(phishing.HOST_FEATURES,
                     phishing.host_features("paypal-secure.web.app"), strict=True))  # fmt: skip
    assert feats["brand_not_owner"] == 1
    assert feats["free_hosting"] == 1
    assert feats["n_hyphens"] == 1
    ip = dict(
        zip(phishing.HOST_FEATURES, phishing.host_features("192.168.1.9"), strict=True)
    )
    assert ip["is_ip"] == 1


def test_featurize_shapes_and_host_mode_ignores_path() -> None:
    urls = ["https://www.example.com", "http://example.com/login/verify"]
    for mode in ("host", "full"):
        X = phishing.featurize(urls, mode)
        assert X.shape == (2, len(phishing.feature_names(mode)))
        assert X.dtype == np.float32
    host = phishing.featurize(urls, "host")
    assert np.allclose(host[0], host[1])  # same host -> identical features
    assert phishing.featurize([], "host").shape[0] == 0


def test_dedupe_hosts_drops_conflicts() -> None:
    hosts, y, conflicts = phishing.dedupe_hosts(
        ["https://www.a.com", "http://a.com/x", "https://b.com", "https://b.com/z"],
        [0, 0, 0, 1],
    )
    assert hosts == ["a.com"] and list(y) == [0] and conflicts == 1


def test_train_reports_fit_and_api_shape(trained: dict[str, Any]) -> None:
    v = trained["res"]["variants"]["url_host"]
    assert {"train", "val", "test", "fit"} <= set(v["families"]["logreg"])
    assert v["selected"] == "logreg"
    assert v["speed_urls_per_s"] > 0
    out = phishing.check_url("http://paypal-login-77.secure-verify.xyz/a",
                             phishing.load(trained["root"], refresh=True))  # fmt: skip
    assert set(out) >= {"phishing_probability", "verdict", "why"}
    assert 0.0 <= out["phishing_probability"] <= 1.0
    assert out["verdict"] in {"phishing", "suspicious", "legitimate"}
    assert out["why"][0].startswith("1. ")
    assert out["why"][-2].startswith("Verdict:")
    assert out["why"][-1].startswith("What to do:")
    batch = phishing.check_urls(["https://www.example.com", "http://1.2.3.4/x"],
                                phishing.load(trained["root"]))  # fmt: skip
    assert len(batch) == 2


def test_blind_test_uses_frozen_model(trained: dict[str, Any]) -> None:
    b = phishing.load(trained["root"], refresh=True)
    before = b["model"]
    res = phishing.blind_test(
        ["https://www.zzz.com", "http://paypal-login.x.xyz/"], [0, 1], b
    )
    assert {"precision", "recall", "f1", "n", "urls_per_s"} <= set(res)
    assert phishing.load(trained["root"])["model"] is before


def test_champion_challenger_rule() -> None:
    keep = common.champion_challenger(0.9, 0.8, 100.0, 200.0, "f1")
    assert keep["promoted"] is False and keep["decision"] == "keep champion"
    tie = common.champion_challenger(0.9, 0.9, 100.0, 50.0, "f1")
    assert tie["promoted"] is True


def test_improve_rolls_back_on_worse_challenger(trained: dict[str, Any]) -> None:
    root = trained["root"]
    model_file = root / "phishing" / "current" / "model.joblib"
    before = model_file.read_bytes()
    # poison: every legit-looking host labelled phishing and vice versa
    bad = [(f"https://www.{w}{i}.com", 1) for i, w in enumerate(WORDS * 15)]
    bad += [(f"http://paypal-login-{i}.verify.xyz/", 0) for i in range(240)]
    res = phishing.improve(bad, root=root)
    assert res["promoted"] is False
    assert res["challenger_items_per_s"] > 0
    assert model_file.read_bytes() == before
    assert not (root / "phishing" / "prev").exists()


def test_improve_promotes_and_keeps_prev(
    trained: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = trained["root"]
    real = common.champion_challenger
    monkeypatch.setattr(common, "champion_challenger",
                        lambda a, b, c, d, m: real(0.0, 1.0, c, d, m))  # fmt: skip
    res = phishing.improve(pd.DataFrame({"url": ["https://www.newsite.com"], "label": [0]}),
                           root=root)  # fmt: skip
    assert res["promoted"] is True
    assert (root / "phishing" / "prev" / "model.joblib").exists()
    assert phishing.load(root, refresh=True)["version"] == 2


# --- speed + threshold follow-up (owner 2026-10-15) --------------------------------------


def test_large_batches_explain_only_flagged(trained: dict[str, Any]) -> None:
    b = phishing.load(trained["root"], refresh=True)
    urls = [f"https://www.{w}{i}.com" for i, w in enumerate(WORDS)] + [
        "http://paypal-login-77.secure-verify.xyz/a"
    ]
    full = phishing.check_urls(urls, b, explain_all=True)
    fast = phishing.check_urls(urls, b, explain_all=False)
    for f, g in zip(full, fast, strict=True):
        assert (f["verdict"], f["phishing_probability"]) == (
            g["verdict"],
            g["phishing_probability"],
        )
        if f["verdict"] != "legitimate":
            assert f["why"] == g["why"]  # flagged URLs: identical explanation
        else:
            assert g["why"][-2].startswith("Verdict: LEGITIMATE")
            assert g["why"][-1].startswith("What to do:")
    # small batches (a person checking links) keep the full explanation by default
    assert phishing.check_urls(urls[:3], b) == phishing.check_urls(
        urls[:3], b, explain_all=True
    )


def test_suspicious_band_comes_from_the_bundle(trained: dict[str, Any]) -> None:
    b = dict(phishing.load(trained["root"], refresh=True))
    url = "http://paypal-login-77.secure-verify.xyz/a"
    p = phishing.check_url(url, b)["phishing_probability"]
    b["threshold"], b["suspicious_at"] = 1.01, max(0.0, p - 0.01)
    assert phishing.check_url(url, b)["verdict"] == "suspicious"
    b["suspicious_at"] = min(1.0, p + 0.01) if p < 1.0 else 1.005
    assert phishing.check_url(url, b)["verdict"] == "legitimate"


def test_threshold_rule_respects_fpr_cap() -> None:
    from scripts import phishing_threshold as pt

    rng = np.random.default_rng(0)
    y_val = np.r_[np.ones(200, int), np.zeros(200, int)]
    p_val = np.r_[rng.uniform(0.6, 1.0, 200), rng.uniform(0.0, 0.7, 200)]
    p_tranco = rng.uniform(0.0, 0.97, 1000)  # popular sites spread widely
    pick = pt.choose(p_val, y_val, p_tranco)
    chosen = pick["chosen"]
    assert pick["constraint_met"]
    assert chosen["fpr_tranco_val"] <= pt.FPR_CAP
    assert chosen["threshold"] > phishing.SUSPICIOUS_AT
