"""Control baselines (01_RESEARCH_DESIGN §3).

These make raw numbers interpretable: the mean-predictor for reconstruction MSE,
the class prior for attack-type inference, and the shuffled-label check for
pipeline integrity.
"""

from __future__ import annotations

import numpy as np


def mean_predictor_mse(X_train: np.ndarray, X_target: np.ndarray) -> float:
    """MSE of predicting each feature's training mean for every target sample (§3.2).

    This is the denominator of the leakage ratio: an inversion attack that beats
    it has learned something; one that does not has not.
    """
    means = X_train.mean(axis=0, keepdims=True)
    return float(np.mean((X_target - means) ** 2))


def class_prior_balanced_accuracy(y: np.ndarray, n_classes: int) -> float:
    """Balanced accuracy of the majority-class constant predictor (§3.3).

    On imbalanced NIDS data a majority predictor scores high plain accuracy while
    knowing nothing; its balanced accuracy is ~1/n_classes, the honest floor.
    """
    counts = np.bincount(y, minlength=n_classes)
    majority = int(np.argmax(counts))
    y_pred = np.full_like(y, majority)
    # balanced accuracy = mean per-class recall; constant predictor => 1/n_classes
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y, y_pred))


def shuffle_train_labels(y_train: np.ndarray, seed: int) -> np.ndarray:
    """Return a copy of the training labels shuffled within the split (§3.6).

    Retraining on these must collapse macro-F1 to near the class prior; if it does
    not, there is leakage in the pipeline.
    """
    rng = np.random.default_rng(seed)
    shuffled = y_train.copy()
    rng.shuffle(shuffled)
    return shuffled
