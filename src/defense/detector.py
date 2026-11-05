"""D3 — runtime adversarial-input detector (inference-time complement to D1/D2).

D1 (adversarial training) and D2 (activation noise) harden the model at *train*
time. This adds an *inference*-time check: an energy-score threshold calibrated on
clean data. The energy score ``-logsumexp(logits)`` (Liu et al. 2020) is low for
confident in-distribution flows and rises for adversarial / off-manifold inputs,
so a threshold set at the clean data's ``1 - target_fpr`` quantile flags roughly
``target_fpr`` of benign traffic while catching inputs the model is unsure about.

ponytail: energy score over a bespoke OOD/detector network — it needs only logits
(no retraining, no extra head, model-agnostic). If a real deployment shows it
misses a specific evasion, add a Mahalanobis/feature-space term then, not before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def energy_scores(
    model: nn.Module, X: np.ndarray, device: str = "cpu", temperature: float = 1.0
) -> np.ndarray:
    """Energy score ``-T * logsumexp(logits / T)`` per row (higher = more suspicious)."""
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
    return energy_from_logits(logits.detach().cpu().numpy(), temperature)


def energy_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """The one energy formula; the SOC scores live logits with it too."""
    t = torch.as_tensor(np.asarray(logits, dtype=np.float64))
    return (-temperature * torch.logsumexp(t / temperature, dim=1)).numpy()


@dataclass(frozen=True)
class EnergyDetector:
    """Energy-threshold detector. ``flags`` are True where score exceeds threshold."""

    threshold: float
    temperature: float = 1.0
    target_fpr: float = 0.05

    def scores(
        self, model: nn.Module, X: np.ndarray, device: str = "cpu"
    ) -> np.ndarray:
        return energy_scores(model, X, device, self.temperature)

    def flags(self, model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        return self.scores(model, X, device) > self.threshold


def fit_energy_detector(
    model: nn.Module,
    X_clean: np.ndarray,
    device: str = "cpu",
    target_fpr: float = 0.05,
    temperature: float = 1.0,
) -> EnergyDetector:
    """Calibrate the threshold at the ``1 - target_fpr`` quantile of clean scores.

    Fit on clean data the model was NOT trained to reject (e.g. the validation
    split), so the flagged fraction of benign traffic is ``~target_fpr`` by
    construction.
    """
    s = energy_scores(model, X_clean, device, temperature)
    threshold = float(np.quantile(s, 1.0 - target_fpr))
    return EnergyDetector(
        threshold=threshold, temperature=temperature, target_fpr=target_fpr
    )
