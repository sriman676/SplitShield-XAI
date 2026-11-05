"""Single source for all metrics (01_RESEARCH_DESIGN §4).

No metric is computed ad hoc in a stage script — every stage imports from here.
Metrics that require a control (§3) are constructed via ``metric_entry``, which
*raises* when the control is missing. That is how "no result without a control"
becomes a property of the code rather than a good intention (06_EXPERIMENTS §4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

MIN_INTERPRETABLE_SUPPORT = 30  # n_test below this => interpretable: false (§4.1)

# Metrics that are meaningless without their control condition (§3).
REQUIRES_CONTROL: frozenset[str] = frozenset(
    {
        "leakage_ratio",
        "attack_success_rate",
        "drift_spearman",
        "drift_jaccard",
        "drift_cosine",
        "attack_type_balanced_acc",
    }
)


class MissingControlError(ValueError):
    """Raised when a control-requiring metric is built without its control."""


def metric_entry(
    name: str, value: float, control: dict[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    """Build a metrics-JSON entry, enforcing the control requirement.

    Args:
        name: Metric name (checked against ``REQUIRES_CONTROL``).
        value: The metric value.
        control: The control block, or ``None``.
        **extra: Additional fields (e.g. ``denominator``).

    Raises:
        MissingControlError: If ``name`` requires a control and none is given.
    """
    if name in REQUIRES_CONTROL and control is None:
        raise MissingControlError(
            f"Metric {name!r} requires a control (01_RESEARCH_DESIGN §3); "
            "refusing to report it without one."
        )
    return {"value": value, "control": control, **extra}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: tuple[str, ...],
    benign_index: int = 0,
    min_support: int = MIN_INTERPRETABLE_SUPPORT,
) -> dict[str, Any]:
    """Compute the detection-performance block (§4.1).

    macro-F1 is the headline; accuracy is deliberately not reported alone. Classes
    with test support below ``min_support`` are flagged ``interpretable: false``
    and excluded from ``macro_f1_interpretable``.
    """
    n = len(label_names)
    labels = list(range(n))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class: dict[str, Any] = {}
    interp_f1: list[float] = []
    for i, name in enumerate(label_names):
        interpretable = int(support[i]) >= min_support
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "interpretable": interpretable,
        }
        if interpretable:
            interp_f1.append(float(f1[i]))

    # Binary base-rate framing: BENIGN = negative class.
    is_benign_true = y_true == benign_index
    fp = int(np.sum((y_pred != benign_index) & is_benign_true))
    n_benign = int(np.sum(is_benign_true))
    alerts = int(np.sum(y_pred != benign_index))
    # attacks flagged as SOME attack, right family or not: what matters for a family
    # never seen in training (temporal split) and for the SOC alert itself
    is_attack = ~is_benign_true
    n_attack = int(np.sum(is_attack))
    caught = int(np.sum((y_pred != benign_index) & is_attack))

    return {
        "attack_recall": float(caught / n_attack) if n_attack else 0.0,
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1_interpretable": float(np.mean(interp_f1)) if interp_f1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "binary_fpr": float(fp / n_benign) if n_benign else 0.0,
        "alert_count": alerts,
        "n_test": int(len(y_true)),
        "per_class": per_class,
    }


def clean_correct_index(
    y_true: np.ndarray, y_clean_pred: np.ndarray, benign_index: int = 0
) -> np.ndarray:
    """Indices of non-BENIGN samples the clean model classified correctly (§3.5).

    ASR is computed only over these — including already-misclassified samples
    inflates every attack number in the project.
    """
    y_true = np.asarray(y_true)
    y_clean_pred = np.asarray(y_clean_pred)
    return np.where((y_true != benign_index) & (y_clean_pred == y_true))[0]


def attack_success_rate(
    y_adv_pred_on_C: np.ndarray,
    n_C: int,
    control: dict[str, Any],
    benign_index: int = 0,
) -> dict[str, Any]:
    """ASR over the clean-correct set ``C`` (§4.2), with its control and |C|.

    ``y_adv_pred_on_C`` are adversarial predictions for the ``C`` samples only.
    A success is a flip to BENIGN. The denominator ``|C|`` is reported explicitly.
    """
    if n_C == 0:
        return metric_entry("attack_success_rate", 0.0, control=control, denominator=0)
    asr = float(np.mean(np.asarray(y_adv_pred_on_C) == benign_index))
    return metric_entry(
        "attack_success_rate", asr, control=control, denominator=int(n_C)
    )


def achieved_perturbation(
    x_raw: np.ndarray,
    x_adv_raw: np.ndarray,
    classes: dict[str, str],
    feature_names: list[str],
) -> dict[str, Any]:
    """Achieved raw perturbation size, summarized per mutability class (§4.3).

    A single scaled epsilon across heterogeneous features is uninterpretable; this
    reports the realized ``|x_adv - x|`` in raw units, grouped by feature class.
    """
    delta = np.abs(
        np.asarray(x_adv_raw, dtype=np.float64) - np.asarray(x_raw, dtype=np.float64)
    )
    by_class: dict[str, Any] = {}
    for cls in sorted(set(classes.values())):
        cols = [i for i, n in enumerate(feature_names) if classes[n] == cls]
        if not cols:
            continue
        block = delta[:, cols]
        by_class[cls] = {
            "mean": float(block.mean()),
            "max": float(block.max()),
            "p95": float(np.percentile(block, 95)),
        }
    return by_class


def leakage_ratio(mse_attack: float, mse_mean_predictor: float) -> dict[str, Any]:
    """Leakage ratio with its mean-predictor control (§3.2).

    ``1 - MSE_attack / MSE_mean_predictor``. <=0 means the attack learned nothing.
    """
    ratio = 1.0 - (mse_attack / mse_mean_predictor) if mse_mean_predictor else 0.0
    return metric_entry(
        "leakage_ratio",
        ratio,
        control={
            "name": "mean_predictor",
            "mse_attack": float(mse_attack),
            "mse_baseline": float(mse_mean_predictor),
        },
    )
