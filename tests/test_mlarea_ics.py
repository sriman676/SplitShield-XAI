"""OT/ICS area: helpers, SOC API shape, champion/challenger (owner 2026-10-15).

Synthetic 1 Hz "plant" (coupled sinusoids + a binary pump + a constant column);
every model is written to ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.mlareas import common, ics

COLS = ["P1_FT01", "P1_PIT01", "P1_LIT01", "P2_SIT01", "P1_PP01AD", "P1_CONST"]


def _plant(n: int, seed: int, attack: tuple[int, int] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    base = np.sin(t / 60)
    data = {
        "P1_FT01": 50 + 10 * base + rng.normal(0, 0.2, n),
        "P1_PIT01": 2 + 0.5 * base + rng.normal(0, 0.01, n),
        "P1_LIT01": 400 + 30 * np.cos(t / 60) + rng.normal(0, 0.5, n),
        "P2_SIT01": 900 + 5 * base + rng.normal(0, 0.3, n),
        "P1_PP01AD": (base > 0).astype(np.float32),
        "P1_CONST": np.ones(n),
    }
    df = pd.DataFrame(data).astype(np.float32)
    df["Attack"] = 0
    if attack:
        s, e = attack
        df.loc[s : e - 1, "P1_FT01"] += 40.0  # spoofed flow, others unchanged
        df.loc[s : e - 1, "Attack"] = 1
    df.insert(
        0, "timestamp", pd.date_range("2026-01-01", periods=n, freq="s").astype(str)
    )
    return df


@pytest.fixture()
def trained(tmp_path: Path) -> dict[str, Any]:
    frames = ([_plant(3000, 0), _plant(3000, 1)], [_plant(2000, 2, attack=(900, 1100))])
    res = ics.train(seeds=(0,), families=("autoencoder", "isolation_forest"), epochs=1,
                    per_epoch=2000, root=tmp_path, frames=frames)  # fmt: skip
    return {"root": tmp_path, "res": res, "test": frames[1][0]}


def test_prep_drops_constant_and_scales() -> None:
    prep = ics.fit_prep(_plant(500, 0))
    assert "P1_CONST" not in prep.columns and "Attack" not in prep.columns
    x = prep.transform(_plant(500, 0))
    assert x.min() >= -1e-6 and x.max() <= 1 + 1e-6
    assert not prep.continuous[prep.columns.index("P1_PP01AD")]


def test_windows_stay_inside_segments() -> None:
    idx = ics.valid_index([40, 50])
    assert (
        idx[0] == ics.LOOKBACK and (idx >= 40 + ics.LOOKBACK).sum() == 50 - ics.LOOKBACK
    )
    w = ics.gather(np.arange(90, dtype=np.float32)[:, None], idx, ics.offsets())
    assert w.shape == (len(idx), len(ics.offsets()), 1)


def test_inject_faults_labels_match_changes() -> None:
    arr = np.random.default_rng(0).random((5000, 4)).astype(np.float32)
    out, lab = ics.inject_faults(arr, np.array([True, True, False, True]), n_events=3)
    changed = np.abs(out - arr).max(axis=1) > 0
    assert lab.sum() > 0 and not changed[lab == 0].any()
    assert np.array_equal(out[:, 2], arr[:, 2])  # binary column never touched


def test_event_metrics_delay_and_false_alarms() -> None:
    labels = np.zeros(86_400, np.int8)
    labels[1000:1100] = 1
    labels[5000:5050] = 1
    alarm = np.zeros_like(labels)
    alarm[1010:1020] = 1  # detects episode 1 after 10 s
    alarm[1150:1160] = 1  # inside post-attack grace -> not false
    alarm[20000:20005] = 1  # false alarm
    m = ics.event_metrics(alarm, labels)
    assert m["episodes"] == 2 and m["episodes_detected"] == 1
    assert m["delay_s_median"] == 10.0 and m["false_alarms"] == 1


def test_train_and_score_window_api(trained: dict[str, Any]) -> None:
    res = trained["res"]
    assert res["selected"] in {"autoencoder", "isolation_forest"}
    fam = res["families"][res["selected"]]
    assert {"train", "val", "test", "fit"} <= set(fam)
    b = ics.load(trained["root"], refresh=True)
    out = ics.score_window(trained["test"].iloc[700:1050], b)
    assert set(out) >= {"anomaly_score", "alarm", "top_sensors", "why", "threshold"}
    assert isinstance(out["alarm"], bool) and len(out["top_sensors"]) == 3
    assert out["why"][0].startswith("1. ")
    assert out["why"][-2].startswith("Verdict:") and out["why"][-1].startswith(
        "What to do:"
    )
    with pytest.raises(ValueError):
        ics.score_window(trained["test"].iloc[:10], b)


def test_blind_test_scores_frozen(trained: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "hai-test1.csv"
    trained["test"].drop(columns=["P2_SIT01"]).to_csv(path, index=False)
    b = ics.load(trained["root"], refresh=True)
    thr = b["threshold"]
    res = ics.blind_test(path, b)
    assert res["missing_sensors_filled"] == ["P2_SIT01"]
    assert {"point", "events", "rows_per_s"} <= set(res) and res["threshold"] == thr


def test_improve_keeps_champion_when_worse(
    trained: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = trained["root"]
    before = (root / "ics" / "current" / "model.joblib").read_bytes()
    real = common.champion_challenger
    monkeypatch.setattr(common, "champion_challenger",
                        lambda a, b, c, d, m: real(1.0, 0.0, c, d, m))  # fmt: skip
    res = ics.improve(_plant(1500, 5), root=root)
    assert res["promoted"] is False and res["new_normal_rows"] == 1500
    assert (root / "ics" / "current" / "model.joblib").read_bytes() == before


def test_improve_promotes_and_keeps_prev(
    trained: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = trained["root"]
    real = common.champion_challenger
    monkeypatch.setattr(common, "champion_challenger",
                        lambda a, b, c, d, m: real(0.0, 1.0, c, d, m))  # fmt: skip
    res = ics.improve(_plant(1500, 6, attack=(100, 200)), root=root)
    assert res["promoted"] is True and res["new_normal_rows"] == 1400
    assert (root / "ics" / "prev" / "model.joblib").exists()
    assert ics.load(root, refresh=True)["version"] == 2


def test_hybrid_bundle_scores_explains_and_blind(tmp_path: Path) -> None:
    """ICS v3 (2026-10-22): a hybrid bundle (autoencoder + LightGBM) scores windows,
    keeps the autoencoder explanations, renames newer HAI columns and is refused by the
    normal-data inbox update."""
    import lightgbm as lgb

    from src.mlareas import ics_hybrid

    test = _plant(2000, 2, attack=(900, 1100))
    ics.train(seeds=(0,), families=("autoencoder",), epochs=1, per_epoch=2000, root=tmp_path / "ae",
              frames=([_plant(3000, 0), _plant(3000, 1)], [test]))  # fmt: skip
    b = ics.load(tmp_path / "ae", refresh=True)
    det = ics._detector(b)
    arr = b["prep"].transform(test)
    X = ics_hybrid.features(arr, ics_hybrid.sensor_dev(det, arr, [len(arr)]), [len(arr)])
    assert X.shape == (len(arr), len(ics_hybrid.feature_names(b["prep"].columns)))
    y = test["Attack"].to_numpy()
    bst = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), 20)
    h = {k: v for k, v in b.items() if k != "_det"} | {
        "family": ics.HYBRID, "gbm": bst.model_to_string(), "ae_threshold": b["threshold"],
        "threshold": 0.5, "column_map": {"NEW_FT01": "P1_FT01"}}  # fmt: skip
    common.save_bundle(ics.AREA, h, tmp_path / "h")
    hb = ics.load(tmp_path / "h", refresh=True)
    s = ics.bundle_scores(hb, arr, [len(arr)])
    assert s[950:1100].mean() > s[:800].mean()  # the spoofed-flow stretch scores higher
    win = test.iloc[700:1050].rename(columns={"P1_FT01": "NEW_FT01"})  # newer HAI name
    out = ics.score_window(win, hb)
    assert out["alarm"] and out["model"].startswith("hybrid") and out["why"]
    path = tmp_path / "t.csv"
    test.to_csv(path, index=False)
    assert ics.blind_test(path, hb)["model"].startswith("hybrid")
    assert ics.improve(test, tmp_path / "h")["promoted"] is False


def _site_plant(n: int, seed: int, attack: tuple[int, int] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    df = pd.DataFrame({"time": t, "FIT01": np.sin(t / 60) + rng.normal(0, 0.02, n),
                       "LIT01": np.cos(t / 90) + rng.normal(0, 0.02, n),
                       "PIT01": 0.5 * np.sin(t / 60) + rng.normal(0, 0.02, n)})  # fmt: skip
    df["attack"] = 0
    if attack:
        a, b = attack
        df.loc[a:b, "LIT01"] += 3.0  # a level reading forced far off its normal pattern
        df.loc[a:b, "attack"] = 1
    return df


def test_site_fit_learns_a_new_plant_and_activates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-10-24: a model fitted on one plant's own normal data finds an attack there,
    and ics.load() prefers the activated plant model over the shipped one."""
    from scripts import ics_site_fit

    b = ics_site_fit.fit_site([_site_plant(6000, 0)], seed=0, epochs=1)
    res = ics_site_fit.score_site(b, [_site_plant(3000, 1, attack=(1500, 1800))])
    assert res["events"]["episodes_detected"] == 1
    sites = tmp_path / "ics" / "sites"
    (sites / "plantA").mkdir(parents=True)
    import joblib

    joblib.dump(b, sites / "plantA" / "bundle.joblib")
    monkeypatch.setattr(common, "area_dir", lambda area, root=None: tmp_path / area)
    assert ics.site_bundle_path() is None  # not active yet
    (sites / "ACTIVE").write_text("plantA", "utf-8")
    assert ics.site_bundle_path() == sites / "plantA" / "bundle.joblib"
    loaded = ics.load(refresh=True)
    assert loaded["threshold"] == b["threshold"]
    ics._CACHE.clear()
