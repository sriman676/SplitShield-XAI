"""Flow stack (owner-approved exception 2026-10-24): the deployed scorer + a multi-dataset
generalist, combined by a small stack; the live output keeps the alert format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "artifacts" / "deploy"
COLS = json.loads((DEPLOY / "columns.json").read_text("utf-8")) if (DEPLOY / "columns.json").is_file() else {}
pytestmark = pytest.mark.skipif("flow_stack" not in COLS, reason="flow stack not deployed")


def test_stack_output_is_a_probability_split_and_drives_confirmed_alerts() -> None:
    from src.soc import engine
    from src.soc.model import Detector, StackCascade

    det = Detector(DEPLOY)
    assert isinstance(det.ensemble, StackCascade)
    assert det.alert_threshold_high == pytest.approx(float(COLS["flow_stack"]["threshold"]))
    assert det.input_stats is not None
    med = np.asarray(det.input_stats["median"], dtype=np.float32)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(med * rng.uniform(0.5, 1.5, size=(300, len(med))).astype(np.float32), columns=det.feature_names)
    p = engine.softmax_t(engine.flow_logits(det, df), det.temperature)
    assert np.allclose(p.sum(1), 1.0, atol=1e-6) and (p >= 0).all()
    benign = det.label_names.index("BENIGN")
    alerts = engine.detect_flows(det, df, "test")
    for a in alerts:
        conf = a["confidence"]
        assert (a["tier"] == "confirmed") == (conf >= det.alert_threshold_high)
    # explanations still reach the wrapped scorer's members
    assert det.ensemble.mlp is not None
    assert benign >= 0
