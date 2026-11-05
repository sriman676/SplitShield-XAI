"""Deployable NetFlow v9 detector: load + score (owner 2026-10-15).

Purpose: the main deployed detector needs CICFlowMeter features, which most
networks do not export. Routers, firewalls and probes (nProbe, softflowd, pmacct,
Zeek-to-NetFlow bridges) export NetFlow v9 / IPFIX fields such as ``IN_BYTES``,
``L4_DST_PORT`` and ``FLOW_DURATION_MILLISECONDS``. ``scripts/deploy_netflow.py``
trains a binary BENIGN-vs-ATTACK detector on NF-CSE-CIC-IDS2018-v2 with the same
recipe as ``deploy.py`` (MLP + LightGBM + RF, members chosen on validation,
temperature and a ~1 %-false-alarm alert threshold fitted on validation) and saves
it to ``artifacts/deploy_netflow/``. This module loads it and scores a DataFrame of
NetFlow records, matching columns BY NAME (case/space-insensitive); missing or
unparsable values are filled with training medians.

    det = load_netflow()            # None when the artifacts are absent
    proba, alerts = score_netflow(df)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.paths import PROJECT_ROOT

NETFLOW_DIR = PROJECT_ROOT / "artifacts" / "deploy_netflow"
_CACHE: dict[str, Any] = {}


def _key(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


class NetflowDetector:
    """Frozen NetFlow detector over ``artifacts/deploy_netflow``."""

    def __init__(self, path: Path = NETFLOW_DIR) -> None:
        import joblib
        import torch

        from src.models.ensemble import EnsembleModel
        from src.models.mlp import SharedMLP

        meta = json.loads((path / "columns.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.feature_names: list[str] = meta["feature_names"]
        self.label_names: list[str] = meta["label_names"]
        self.temperature = float(meta.get("temperature", 1.0))
        self.alert_threshold = float(meta.get("alert_threshold", 0.5))
        self.medians = np.asarray(meta["input_stats"]["median"], dtype=np.float64)
        self.scaler = joblib.load(path / "scaler.joblib")  # our own artifact
        self.model = SharedMLP(
            n_features=len(self.feature_names),
            n_classes=len(self.label_names),
            hidden_layers=meta["hidden_layers"],
            hidden_width=meta["hidden_width"],
            dropout=meta["dropout"],
            norm=meta.get("norm", "none"),
        )
        self.model.load_state_dict(
            torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()
        weights = meta.get("ensemble") or {"mlp": 1.0}
        trees = (
            joblib.load(path / "trees.joblib")
            if (path / "trees.joblib").exists()
            else {}
        )
        self.scorer = EnsembleModel(self.model, trees, weights).eval()

    def align(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Raw feature matrix in training order + names of columns not found."""
        by_key = {_key(c): c for c in df.columns}
        X = np.empty((len(df), len(self.feature_names)), dtype=np.float64)
        missing = []
        for j, name in enumerate(self.feature_names):
            col = by_key.get(_key(name))
            if col is None:
                missing.append(name)
                X[:, j] = self.medians[j]
                continue
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(
                np.float64, na_value=np.nan
            )
            X[:, j] = np.where(np.isfinite(v), v, self.medians[j])
        return X, missing

    def score(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated probabilities (n, 2) and alert flags (attack prob >= threshold)."""
        import torch

        X, _ = self.align(df)
        Xs = np.asarray(self.scaler.transform(X), dtype=np.float32)
        with torch.no_grad():
            logits = self.scorer(torch.as_tensor(Xs))
            p = torch.softmax(logits / self.temperature, 1).numpy()
        benign = self.label_names.index("BENIGN")
        return p, (1.0 - p[:, benign]) >= self.alert_threshold


def save_netflow(
    path: Path,
    mlp: Any,
    trees: dict[str, Any],
    scaler: Any,
    meta: dict[str, Any],
) -> None:
    """Write the artifact folder ``load_netflow`` reads (model.pt, trees, scaler,
    columns.json). ``meta`` must hold feature/label names, the MLP shape, the
    ensemble weights, temperature, alert threshold and input_stats."""
    import joblib
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.save(mlp.state_dict(), path / "model.pt")
    members = {k: v for k, v in trees.items() if k in (meta.get("ensemble") or {})}
    if members:
        joblib.dump(members, path / "trees.joblib", compress=3)
    joblib.dump(scaler, path / "scaler.joblib")
    (path / "columns.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_netflow(path: Path = NETFLOW_DIR) -> NetflowDetector | None:
    """The NetFlow detector, or None when ``path`` has no artifacts (cached)."""
    key = str(Path(path).resolve())
    if key not in _CACHE:
        if not (Path(path) / "columns.json").exists():
            return None
        _CACHE[key] = NetflowDetector(Path(path))
    det: NetflowDetector = _CACHE[key]
    return det


def score_netflow(
    df: pd.DataFrame, path: Path = NETFLOW_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """Score NetFlow records: (probabilities (n, 2), alerts bool (n,)).

    Raises FileNotFoundError when the detector has not been deployed."""
    det = load_netflow(path)
    if det is None:
        raise FileNotFoundError(
            f"no NetFlow detector at {path}; run scripts/deploy_netflow.py"
        )
    return det.score(df)
