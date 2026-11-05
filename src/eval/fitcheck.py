"""Automatic over/underfitting check for every centrally trained model.

Rules (thresholds from the 2026-10-13 diagnosis on CIC-IDS2017, where the MLP was
found overconfident and stopped early on a noisy validation curve):

  * overfitting    train macro-F1 exceeds validation by more than ``GAP`` (0.05)
  * under-trained  the best validation epoch is among the last 20 % of a run that
                   hit ``max_epochs`` -> it was still improving when it stopped
  * noisy stop     early stopping fired while validation F1 was still dipping by
                   more than ``NOISE`` below its recent best -> likely a dip, not a
                   plateau (the "stops too early" case)
  * overconfident  validation loss at the end is more than ``CONF`` (25 %) above its
                   minimum while validation F1 did not fall -> probabilities too sharp

Each finding comes with the tweak that addresses it. Research runs only REPORT the
verdict (owner: registered design unchanged); deploy.py applies the tweaks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

GAP = 0.05
NOISE = 0.02  # val macro-F1 dip that counts as noise
CONF = 1.25
TAIL = 5


def diagnose(
    history: list[dict[str, float]],
    train_f1: float,
    max_epochs: int,
    stopped_early: bool,
) -> dict[str, Any]:
    """Verdict from the per-epoch history (``val_f1``, ``val_loss``) and final train F1."""
    if not history:
        return {"verdict": "unknown", "findings": [], "tweaks": []}
    val = np.array([h["val_f1"] for h in history])
    loss = np.array([h["val_loss"] for h in history])
    best = int(val.argmax())
    best_f1 = float(val[best])
    findings: list[str] = []
    tweaks: list[str] = []
    gap = float(train_f1 - best_f1)
    if gap > GAP:
        findings.append(
            f"overfitting: train macro-F1 {train_f1:.3f} vs val {best_f1:.3f}"
        )
        tweaks.append("regularize")
    if not stopped_early and len(val) >= max_epochs and best >= 0.8 * len(val):
        findings.append(f"under-trained: best val epoch {best + 1} of {len(val)}")
        tweaks.append("train_longer")
    tail = val[-TAIL:]
    # dips, not trend: the largest drop below an earlier value in the last epochs
    drawdown = float((np.maximum.accumulate(tail) - tail).max()) if len(tail) else 0.0
    if stopped_early and len(tail) >= 3 and drawdown > NOISE:
        findings.append(
            f"noisy stop: val macro-F1 dipped {drawdown:.3f} below its recent best "
            "when early stopping fired"
        )
        tweaks.append("train_longer")
    if (
        loss.min() > 0
        and float(loss[-1]) > CONF * float(loss.min())
        and val[-1] >= val[int(loss.argmin())] - 0.01
    ):
        findings.append(
            f"overconfident: val loss {loss[-1]:.3f} vs min {loss.min():.3f} with "
            "val F1 not falling"
        )
        tweaks.append("calibrate")
    if "regularize" in tweaks:
        verdict = "overfitting"
    elif "train_longer" in tweaks:
        verdict = "underfitting"
    elif tweaks:
        verdict = "overconfident"
    else:
        verdict = "good fit"
    return {
        "verdict": verdict,
        "findings": findings,
        "tweaks": sorted(set(tweaks)),
        "train_macro_f1": float(train_f1),
        "best_val_macro_f1": best_f1,
        "train_val_gap": gap,
        "best_epoch": best + 1,
        "epochs_run": len(val),
    }


def with_reference(
    fit: dict[str, Any], reference_f1: float, name: str
) -> dict[str, Any]:
    """High-bias underfitting: train AND val sit well below a simple baseline (e.g.
    the Random Forest on the same split), so no train/val gap shows it. Adds the
    finding to ``fit`` (a copy) when train macro-F1 < reference - ``GAP``."""
    if "train_macro_f1" not in fit or fit["train_macro_f1"] >= reference_f1 - GAP:
        return fit
    out = {**fit, "findings": [*fit["findings"]], "tweaks": [*fit["tweaks"]]}
    out["findings"].append(
        f"high bias: train macro-F1 {fit['train_macro_f1']:.3f} is below the "
        f"{name} baseline {reference_f1:.3f}"
    )
    out["tweaks"] = sorted(set(out["tweaks"]) | {"train_longer"})
    if out["verdict"] in ("good fit", "overconfident"):
        out["verdict"] = "underfitting"
    out["reference"] = {"name": name, "macro_f1": reference_f1}
    return out


# --- model fit report (owner 2026-10-16): all deployed models, permanent check -----

SHIFT = 0.05  # |validation - test| above this: the splits differ (distribution shift)
STILL_RISING = 0.01  # val score still gaining this much on the last size step


def split_verdict(
    train: float, val: float, test: float | None = None, low: float = 0.7
) -> dict[str, Any]:
    """Over/underfitting from one train/val/test score triple (higher = better)."""
    findings: list[str] = []
    gap = float(train - val)
    if gap > GAP:
        findings.append(f"overfitting: train {train:.3f} vs validation {val:.3f}")
    if train < low and val < low:
        findings.append(f"underfitting: train {train:.3f} and validation {val:.3f} both low")
    if test is not None and abs(val - test) > SHIFT:
        findings.append(f"split shift: validation {val:.3f} vs test {test:.3f}")
    verdict = ("overfitting" if gap > GAP else "underfitting" if train < low and val < low
               else "split shift" if findings else "good fit")  # fmt: skip
    return {"train": round(float(train), 4), "val": round(float(val), 4),
            "test": None if test is None else round(float(test), 4),
            "gap": round(gap, 4), "verdict": verdict, "findings": findings}  # fmt: skip


def curve_verdict(points: list[dict[str, float]]) -> dict[str, Any]:
    """Learning curve (score vs training-set size) -> what more data / capacity would do.

    * val still rising at the largest size  -> data-limited (more data helps)
    * train high, val flat, gap stays large -> overfitting (variance)
    * train and val both low and flat       -> underfitting (bias; more capacity /
                                               better features help, not more data)
    """
    if len(points) < 2:
        return {"verdict": "unknown", "findings": []}
    pts = sorted(points, key=lambda p: p["n"])
    last, prev = pts[-1], pts[-2]
    rise = float(last["val"] - prev["val"])
    gap = float(last["train"] - last["val"])
    findings = []
    if rise > STILL_RISING:
        findings.append(f"validation still rising (+{rise:.3f} on the last size step): "
                        "more training data should help")  # fmt: skip
    if gap > GAP:
        findings.append(f"train-validation gap {gap:.3f} at full size")
    if last["train"] < 0.7 and last["val"] < 0.7:
        findings.append("train and validation both below 0.7: model too simple or "
                        "features too weak")  # fmt: skip
    verdict = ("underfitting" if last["train"] < 0.7 and last["val"] < 0.7
               else "overfitting" if gap > GAP and rise <= STILL_RISING
               else "data-limited" if rise > STILL_RISING else "good fit")  # fmt: skip
    return {"verdict": verdict, "findings": findings, "final_gap": round(gap, 4),
            "last_step_val_gain": round(rise, 4)}  # fmt: skip


def model_report(path: Any = None) -> dict[str, Any]:
    """The saved fit report (scripts/fit_report.py) summarised for the controller:
    {"models": {name: {"verdict", "findings"}}, "flagged": [...], "age_days"}.
    Cheap (reads one JSON). A model is flagged when its split or curve verdict is not
    "good fit" - the controller notifies once and a person decides what to do."""
    import json
    import time
    from pathlib import Path

    from utils.paths import RESULTS_DIR

    p = Path(path) if path else RESULTS_DIR / "extra" / "fit_report.json"
    try:
        doc = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return {"models": {}, "flagged": [], "age_days": None, "missing": str(p)}
    models = {}
    for name, m in doc.get("models", {}).items():
        verdicts = {m.get("split", {}).get("verdict"), m.get("curve", {}).get("verdict")} - {None}
        findings = [*m.get("split", {}).get("findings", []), *m.get("curve", {}).get("findings", [])]
        bad = sorted(v for v in verdicts if v not in ("good fit", "unknown"))
        models[name] = {"verdict": ", ".join(bad) or "good fit", "findings": findings}
    age = (time.time() - p.stat().st_mtime) / 86400
    return {"models": models, "flagged": [n for n, m in models.items() if m["verdict"] != "good fit"],
            "age_days": round(age, 1), "generated": doc.get("generated")}  # fmt: skip
