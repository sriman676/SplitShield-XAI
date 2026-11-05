"""Per-attack explanation + defence report (the XAI deliverable).

For each attack class the model can predict, this pairs the model's own top SHAP
features (why it flagged that class) with curated mitigations from
``config/defense_kb.py`` (how to defend). Output is a human-readable Markdown file
plus a JSON sidecar. Reuses the exact Stage-3 explainer so the attributions are
consistent with the rest of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from config.defense_kb import defense_for
from src.data.preprocess import ProcessedData
from src.explain.explainer import compute_shap
from src.explain.shap_utils import topk_features
from src.training.centralized import predict
from utils.logging_config import get_logger

_log = get_logger(__name__)


def per_class_top_features(
    model: Any,
    data: ProcessedData,
    device: str,
    k: int = 8,
    max_instances: int = 64,
) -> dict[str, list[str]]:
    """Return class-name -> top-k feature names by mean |SHAP| on that class."""
    n_classes = len(data.label_names)
    n = min(max_instances, len(data.X_test))
    x = data.X_test[:n]
    background = data.X_train[: min(100, len(data.X_train))]
    res = compute_shap(
        model, x, background, n_classes, device, kernel_instances=min(5, n)
    )
    kk = min(k, len(data.feature_names))
    out: dict[str, list[str]] = {}
    for c, name in enumerate(data.label_names):
        mean_abs = np.abs(res.attributions[:, :, c]).mean(axis=0)
        out[name] = [data.feature_names[i] for i in sorted(topk_features(mean_abs, kk))]
    return out


def build_report(
    model: Any, data: ProcessedData, device: str, out_dir: Path, k: int = 8
) -> Path:
    """Write attack_defense_report.md (+ .json). Returns the markdown path.

    Only classes actually present in the test set are reported (the rest have no
    instances to explain); each gets its top features and curated defenses.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    present = set(np.unique(data.y_test).tolist())
    pred_present = set(predict(model, data.X_test, device).tolist())
    tops = per_class_top_features(model, data, device, k=k)

    rows: list[dict[str, Any]] = []
    lines = ["# Attack explanation & defence report", ""]
    for c, name in enumerate(data.label_names):
        if name == "BENIGN":
            continue
        kb = defense_for(name)
        seen = c in present or c in pred_present
        rows.append(
            {
                "class": name,
                "observed": seen,
                "top_features": tops.get(name, []),
                "summary": kb["summary"],
                "defenses": kb["defenses"],
            }
        )
        lines += [
            f"## {name}" + ("" if seen else "  _(no test instances)_"),
            f"*What it is:* {kb['summary']}",
            f"*Model's key indicators (top SHAP features):* "
            f"{', '.join(tops.get(name, [])) or 'n/a'}",
            "*How to defend:*",
            *[f"- {d}" for d in cast("list[str]", kb["defenses"])],
            "",
        ]

    md = out_dir / "attack_defense_report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "attack_defense_report.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    _log.info("Wrote attack/defence report for %d classes -> %s", len(rows), md)
    return md
