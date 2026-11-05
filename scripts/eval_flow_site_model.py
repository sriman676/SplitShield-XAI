"""Site-trained flow model on HIKARI (owner 2026-10-24) - NOT BLIND (HIKARI was scored once in blind round 2,
then allowed as development data). 30 % labelled flows of each site train a LightGBM on the 77 CICFlowMeter
features (threshold: 1 % false alarms on its own labelled hold-out); the other 70 % are scored.

    python -m scripts.eval_flow_site_model
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from scripts.blind_test_v2 import _hikari, _rates
from src.soc.columns import align
from src.soc.model import Detector
from utils.paths import PROJECT_ROOT

det = Detector(PROJECT_ROOT / "artifacts" / "deploy")
df = _hikari()
aligned, _ = align(df, det.feature_names)
X = aligned[det.feature_names].apply(lambda c: c.astype(float)).fillna(0).to_numpy(np.float32)
y = df["Label"].fillna(0).astype(float).to_numpy() > 0
out = {}
rng = np.random.default_rng(0)
for site in sorted(df["_set"].unique()):
    idx = np.flatnonzero((df["_set"] == site).to_numpy())
    rng.shuffle(idx)
    cut = int(0.3 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    vcut = int(0.8 * len(tr))
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=0, n_jobs=8, verbose=-1)
    m.fit(X[tr[:vcut]], y[tr[:vcut]])
    pv = m.predict_proba(X[tr[vcut:]])[:, 1]
    neg = np.sort(pv[~y[tr[vcut:]]])[::-1]
    thr = float(neg[int(0.01 * len(neg))]) if len(neg) else 0.5  # 1 % false alarms on the site's labelled part
    pt = m.predict_proba(X[te])[:, 1]
    out[site] = {"labelled_train_flows": int(len(tr)), "threshold": thr, "test": _rates(y[te], pt >= thr)}
(PROJECT_ROOT / "results" / "extra" / "flow_site_model_hikari.json").write_text(json.dumps({"label": "NOT BLIND", **out}, indent=1, default=float), "utf-8")
print(json.dumps(out, indent=1, default=float))
