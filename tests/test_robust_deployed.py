"""Tests for scripts/robust_deployed.py on a tiny synthetic detector (owner 2026-10-15)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from scripts import robust_deployed as rd
from src.attack.constraints import FeatureConstraints
from src.models.ensemble import EnsembleModel
from src.models.mlp import SharedMLP

lightgbm = pytest.importorskip("lightgbm")

NAMES = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Flow IAT Mean"]


def _setup() -> tuple[Any, Any, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 800
    raw = np.column_stack(
        [
            rng.choice([80, 443], n),
            rng.integers(1_000, 100_000, n),
            rng.integers(1, 50, n),
            rng.uniform(10, 1_000, n),
        ]
    ).astype(np.float64)
    y = (raw[:, 3] < 400).astype(np.int64)  # "attack" = short inter-arrival times
    sc = StandardScaler().fit(raw)
    Xs = sc.transform(raw).astype(np.float32)
    gbm = lightgbm.LGBMClassifier(n_estimators=30, num_leaves=7, verbose=-1).fit(Xs, y)
    torch.manual_seed(0)
    mlp = SharedMLP(n_features=4, n_classes=2, hidden_layers=2, hidden_width=16)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    xt, yt = torch.as_tensor(Xs), torch.as_tensor(y)
    for _ in range(200):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(mlp(xt), yt).backward()
        opt.step()
    mlp.eval()
    det = SimpleNamespace(
        model=mlp,
        ensemble=EnsembleModel(mlp, {"lgbm": gbm}, {"mlp": 1.0, "lgbm": 1.0}).eval(),
        label_names=["BENIGN", "DoS"],
        energy_threshold=0.0,
        alert_threshold_high=0.5,
    )
    cons = FeatureConstraints.from_run(NAMES, sc, raw, corrected=True)
    return det, cons, Xs, y


def _check_realism(cons: Any, x0: np.ndarray, xa: np.ndarray) -> None:
    r0, ra = cons.to_raw(x0), cons.to_raw(xa)
    np.testing.assert_allclose(ra[:, 0], r0[:, 0], atol=1e-6)  # port frozen
    tol = 1e-4 * (np.abs(r0[:, 1:3]) + 1)  # float32 inputs: originals are ~integers
    assert (ra[:, 1:3] >= r0[:, 1:3] - tol).all()  # counts/duration only grow
    np.testing.assert_allclose(ra[:, 1:3], np.rint(ra[:, 1:3]), atol=1e-6)


def test_attacks_respect_constraints() -> None:
    det, cons, Xs, y = _setup()
    x0 = Xs[y == 1][:40].astype(np.float64)
    rng = np.random.default_rng(1)
    _check_realism(cons, x0, rd.attack_random(x0, 0.5, cons, rng))
    _check_realism(cons, x0, rd.attack_pgd(det.model, x0, 0.5, cons, 0, iters=5))
    xq, q = rd.attack_query(
        lambda z: rd.proba(det.ensemble, z), x0, 0.5, cons, 0, steps=3
    )
    _check_realism(cons, x0, xq)
    assert (q >= 1).all()


def test_query_never_lowers_benign_probability() -> None:
    det, cons, Xs, y = _setup()
    x0 = Xs[y == 1][:40].astype(np.float64)
    xq, _ = rd.attack_query(
        lambda z: rd.proba(det.ensemble, z), x0, 1.0, cons, 0, steps=4
    )
    before = rd.proba(det.ensemble, x0)[:, 0]
    after = rd.proba(det.ensemble, xq)[:, 0]
    assert (after >= before - 1e-6).all()


def test_run_seed_summary_and_report(tmp_path: Path) -> None:
    det, cons, Xs, y = _setup()
    runs = [
        rd.run_seed(det, Xs, y, cons, s, n=30, eps_list=(0.2, 1.0), query_steps=2)
        for s in (0, 1)
    ]
    assert set(runs[0]["attacks"]) == {
        f"{f}@{e:g}" for f in rd.FAMILIES for e in (0.2, 1.0)
    }
    for o in runs[0]["attacks"].values():
        assert 0.0 <= o["success_rate"] <= 1.0
        assert "flipped" not in o
    summary = rd.summarize(runs)
    assert summary["query@1"]["queries_mean"] >= 1
    v = rd.verdict(summary, (0.2, 1.0))
    assert len([s for s in v if s.startswith("eps ")]) == 4
    doc = {
        "detector": "synthetic", "dataset": "synthetic", "members": ["lgbm", "mlp"],
        "n_flows": runs[0]["n_flows"], "seeds": [0, 1], "epsilons": [0.2, 1.0],
        "clean": rd.clean_metrics(det, Xs, y), "summary": summary,
        "verdict": rd.verdict(summary, (0.2, 1.0)), "caveats": ["c"], "runs": runs,
        "scaler_means_reproduced": "4/4", "scaler_means_differing": [],
    }  # fmt: skip
    assert doc["clean"]["ensemble"]["accuracy"] > 0.8
    rd.write_md(doc, tmp_path / "r.md")
    rd.plot(summary, (0.2, 1.0), tmp_path / "r.png")
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "Verdict" in text and "| query | 1 |" in text
    assert (tmp_path / "r.png").stat().st_size > 0
