"""LIME in the same (scaled) feature space as SHAP, for top-k overlap.

The kit asks for LIME in the same feature space as SHAP (03_ARCHITECTURE Stage 3).
SHAP here is computed on the model's scaled inputs, so LIME is too: the predict
function receives scaled features, and the top-k overlap is a set comparison over
feature identity, which is space-invariant. Reporting attributions in raw units
for human reading is a presentation step left to figures, not this comparison.
"""

from __future__ import annotations

import numpy as np
import torch

from src.explain.explainer import _proba_fn
from utils.logging_config import get_logger

_log = get_logger(__name__)


def lime_topk(
    model: torch.nn.Module,
    X_explain: np.ndarray,
    background: np.ndarray,
    feature_names: list[str],
    k: int = 10,
    device: str = "cpu",
) -> list[set[int]]:
    """Return the top-k LIME feature-index set for each explained instance.

    Args:
        model: Model over scaled features.
        X_explain: Instances to explain (scaled).
        background: Training data for LIME's sampler (scaled).
        feature_names: Column names (length n_features).
        k: Top-k cutoff.
        device: Torch device string.

    Returns:
        One set of top-k feature indices per instance.
    """
    from lime.lime_tabular import LimeTabularExplainer

    proba = _proba_fn(model, device)
    n_features = len(feature_names)
    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        mode="classification",
        discretize_continuous=False,
        random_state=0,
    )
    out: list[set[int]] = []
    for row in X_explain:
        exp = explainer.explain_instance(
            row, proba, num_features=n_features, top_labels=1
        )
        label = exp.available_labels()[0]
        ranked = sorted(exp.as_map()[label], key=lambda t: abs(t[1]), reverse=True)
        out.append({idx for idx, _ in ranked[:k]})
    return out
