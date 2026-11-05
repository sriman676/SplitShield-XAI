"""Random Forest reference point (05_MODELING §3, Tier 1).

Non-split, Stage 1 only, TreeExplainer-exact. Never enters Stages 2-7.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, seed: int
) -> RandomForestClassifier:
    """Train a class-balanced Random Forest baseline."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf
