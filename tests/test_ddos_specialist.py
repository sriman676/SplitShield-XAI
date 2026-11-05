"""Champion OR DDoS specialist (owner-approved exception 2026-10-17): the live
detector path keeps every other flow unchanged and specialist hits become normal,
confirmed DDoS alerts (same format, tiers and explanations)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "artifacts" / "deploy"
pytestmark = pytest.mark.skipif(not (DEPLOY / "columns.json").is_file(), reason="no deployed detector")


class FakeSpecialist:
    """Fires (score 0.9) on rows whose first scaled feature is above ``cut``."""

    def __init__(self, cut: float) -> None:
        self.cut = cut

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = np.where(X[:, 0] > self.cut, 0.9, 0.1)
        return np.c_[1 - s, s]


def _deploy_with(tmp_path: Path, spec: object) -> Path:
    dst = tmp_path / "deploy"
    shutil.copytree(DEPLOY, dst, ignore=shutil.ignore_patterns("ddos_specialist.joblib"))
    joblib.dump({"model": spec, "threshold": 0.5}, dst / "ddos_specialist.joblib")
    cols = json.loads((dst / "columns.json").read_text("utf-8"))
    cols["ddos_specialist"] = {"file": "ddos_specialist.joblib", "threshold": 0.5}
    cols.pop("flow_stack", None)  # the specialist layer on its own (the stack: test_flow_stack.py)
    (dst / "columns.json").write_text(json.dumps(cols), "utf-8")
    return dst


def _benign_like(det: object, n: int = 200) -> pd.DataFrame:
    med = np.asarray(det.input_stats["median"], dtype=np.float32)  # type: ignore[attr-defined]
    rng = np.random.default_rng(0)
    X = med * rng.uniform(0.9, 1.1, size=(n, len(med))).astype(np.float32)
    return pd.DataFrame(X, columns=det.feature_names)  # type: ignore[attr-defined]


def test_cascade_changes_only_specialist_rows(tmp_path: Path) -> None:
    from src.soc import engine
    from src.soc.model import Detector, SpecialistCascade

    plain = Detector(_deploy_with(tmp_path / "a", FakeSpecialist(np.inf)))
    df = _benign_like(plain)
    base = Detector(DEPLOY) if "ddos_specialist" not in json.loads((DEPLOY / "columns.json").read_text("utf-8")) else None
    xs = plain.scaler.transform(df.to_numpy(np.float32))
    cut = float(np.median(xs[:, 0]))
    det = Detector(_deploy_with(tmp_path / "b", FakeSpecialist(cut)))
    assert isinstance(det.ensemble, SpecialistCascade)
    lp = engine.flow_logits(plain, df)  # specialist never fires
    lc = engine.flow_logits(det, df)
    fired = np.asarray(det.ensemble.last_specialist)
    benign = det.label_names.index("BENIGN")
    champion_benign = lp.argmax(1) == benign
    assert np.array_equal(fired, champion_benign & (xs[:, 0] > cut))
    assert np.array_equal(lp[~fired], lc[~fired])
    if base is not None:  # without a specialist entry the scorer is the plain champion
        assert np.allclose(engine.flow_logits(base, df), lp)
    assert fired.any(), "the test data must make the fake specialist fire"
    p = engine.softmax_t(lc[fired], det.temperature)
    assert np.allclose(p[:, det.label_names.index("DDoS")], 0.9, atol=1e-4)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)
    alerts = engine.detect_flows(det, df, "test")
    hit = [a for a in alerts if a["label"] == "DDoS"]
    assert hit and all(a["tier"] == "confirmed" and a["techniques"] for a in hit)
    assert all(a["reason"] and "logits" in a for a in hit)


def test_explanations_reach_the_wrapped_ensemble(tmp_path: Path) -> None:
    from src.explain.ensemble_why import member_weights
    from src.soc.model import Detector

    det = Detector(_deploy_with(tmp_path, FakeSpecialist(np.inf)))
    plain_weights = json.loads((DEPLOY / "columns.json").read_text("utf-8")).get("ensemble") or {"mlp": 1.0}
    assert member_weights(det) == {k: float(v) for k, v in plain_weights.items() if v > 0}
    assert getattr(det.ensemble, "trees", None) is not None or "lgbm" not in plain_weights
