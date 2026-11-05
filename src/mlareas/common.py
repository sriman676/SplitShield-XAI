"""Shared helpers for the extra ML areas (owner 2026-10-15).

Paths, metric summaries, the over/underfitting verdict and the champion/challenger
model store live here so phishing and ICS follow exactly the same rules:

  * model selection uses VALIDATION data only; test is reported, never chosen on;
  * fit verdict: ``train - val > 0.05`` -> overfitting (tree models fit train
    near-perfectly by design, so the verdict says so rather than hiding it);
  * the store keeps the current model under ``current/`` and the previous one
    under ``prev/`` so a promotion can always be rolled back.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import joblib

GAP_RULE = 0.05


def main_project_root() -> Path:
    """The main checkout root, even when this code runs from a git worktree."""
    here = Path(__file__).resolve().parents[2]
    parts = here.parts
    if ".claude" in parts:
        return Path(*parts[: parts.index(".claude")])
    return here


def artifacts_root() -> Path:
    """Shared model store (override with ``MLAREAS_ARTIFACTS`` for tests)."""
    env = os.environ.get("MLAREAS_ARTIFACTS")
    if env:
        return Path(env)
    return main_project_root() / "artifacts" / "mlareas"


def raw_data_dir(name: str) -> Path:
    return main_project_root() / "data" / "raw" / name


def mean_sd(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(statistics.fmean(vals), 4), "sd": round(sd, 4)}


def fit_verdict(train: float, val: float, tree_model: bool = False) -> dict[str, Any]:
    """Apply the project gap rule to one score (higher = better)."""
    gap = train - val
    if gap > GAP_RULE:
        verdict = "overfitting"
    elif train < 0.7 and val < 0.7:
        verdict = "underfitting"
    else:
        verdict = "good fit"
    note = ""
    if tree_model and gap > GAP_RULE:
        note = "tree ensemble memorises train by design; judge it on val vs test"
    return {
        "train": round(train, 4),
        "val": round(val, 4),
        "gap": round(gap, 4),
        "verdict": verdict,
        "note": note,
    }


def items_per_second(fn: Callable[[], Any], n_items: int, repeats: int = 3) -> float:
    """Best-of-``repeats`` throughput of ``fn`` processing ``n_items``."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return round(n_items / max(best, 1e-9), 1)


# --------------------------------------------------------------- model store


def area_dir(area: str, root: Path | None = None) -> Path:
    return (root or artifacts_root()) / area


def save_bundle(area: str, bundle: dict[str, Any], root: Path | None = None) -> Path:
    """Save ``bundle`` as the current model, moving the old one to ``prev/``."""
    base = area_dir(area, root)
    cur, prev = base / "current", base / "prev"
    if cur.exists():
        if prev.exists():
            shutil.rmtree(prev)  # only our own older model copy, never data
        cur.rename(prev)
    cur.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cur / "model.joblib")
    meta = {k: v for k, v in bundle.items() if isinstance(v, str | int | float | dict)}
    (cur / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return cur


def load_bundle(area: str, root: Path | None = None) -> dict[str, Any]:
    path = area_dir(area, root) / "current" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"no trained {area} model at {path}; run python -m scripts.train_{area}"
        )
    bundle: dict[str, Any] = joblib.load(path)
    return bundle


def champion_challenger(
    champion_score: float,
    challenger_score: float,
    champion_speed: float,
    challenger_speed: float,
    metric: str,
) -> dict[str, Any]:
    """Keep the challenger only if detection is not worse on the fixed val set."""
    keep = challenger_score >= champion_score
    return {
        "decision": "promote challenger" if keep else "keep champion",
        "promoted": keep,
        "metric": metric,
        "champion": round(champion_score, 4),
        "challenger": round(challenger_score, 4),
        "champion_items_per_s": champion_speed,
        "challenger_items_per_s": challenger_speed,
    }
