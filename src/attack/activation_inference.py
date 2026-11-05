"""T3 — activation inference / feature reconstruction (02_THREAT_MODEL T3, Stage 4c).

Which adversary: T3, the honest-but-curious server. Goal: recover a client's
private input features from the cut-layer activations it legitimately receives.

What it knows and where: ``ServerNet``, the ``ClientNet`` architecture, and an
auxiliary dataset from the same distribution. The knowledge enters where the
auxiliary features are embedded through ``client`` to build (activation, feature)
training pairs, and where the intercepted target activations are inverted.

STATED ASSUMPTION (human-approved, Task 5): the server embeds its auxiliary data
through the *trained* ClientNet (not a shadow model). This is standard in the
inversion literature and gives a transferable, informative signal, but it is
slightly stronger than 'architecture only' — reported leakage is therefore an
upper-ish bound rather than a pure passive lower bound. Recorded in the metrics
JSON ``deviations`` and in 11_LIMITATIONS.

Control (§3.2): the mean-predictor reconstruction baseline. Raw MSE is
uninterpretable; the leakage ratio ``1 - MSE_attack/MSE_mean`` and its per-feature
breakdown are the reported quantities. Reconstruction is in scaled feature space
(all features standardized, so per-feature MSE is comparable); the leakage ratio
is scale-free per feature regardless.

Strongest version NOT implemented: an active server that adapts ServerNet to
maximize recoverability (strictly stronger, 11_LIMITATIONS §1 future work).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.attack.client_attribution import cut_activations
from src.eval.baselines import mean_predictor_mse
from src.eval.metrics import leakage_ratio
from src.models.inversion import InversionNet


def _train_inversion(
    A: np.ndarray, X: np.ndarray, device: str, seed: int, epochs: int
) -> InversionNet:
    torch.manual_seed(seed)
    inv = InversionNet(A.shape[1], X.shape[1]).to(device)
    opt = torch.optim.AdamW(inv.parameters(), lr=1e-3)
    a = torch.as_tensor(A, dtype=torch.float32, device=device)
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    inv.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(inv(a), x)
        loss.backward()
        opt.step()
    return inv


def _per_feature_leakage(
    x_true: np.ndarray, x_hat: np.ndarray, x_aux: np.ndarray
) -> list[float]:
    """Per-feature ``1 - mse_attack/mse_mean`` (§3.2: not just the aggregate)."""
    mse_attack = np.mean((x_hat - x_true) ** 2, axis=0)
    mse_mean = np.mean((x_true - x_aux.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mse_mean > 0, 1.0 - mse_attack / mse_mean, 0.0)
    return [float(r) for r in ratio]


def activation_inference(
    client: nn.Module,
    X_aux_scaled: np.ndarray,
    X_target_scaled: np.ndarray,
    device: str,
    seed: int,
    epochs: int = 200,
) -> dict[str, Any]:
    """Reconstruct target features from activations; report leakage vs the baseline.

    Args:
        client: The trained ``ClientNet`` (embeds aux data — see stated assumption).
        X_aux_scaled: Auxiliary same-distribution features (attacker-held).
        X_target_scaled: The client's private features to reconstruct (evaluation).
        device: Torch device.
        seed: Attack randomness (§5).
        epochs: Inversion training epochs (number-affecting, not hardware-varying).

    Returns:
        A leakage-ratio metrics entry with per-feature ratios and both raw MSEs.
    """
    A_aux = cut_activations(client, X_aux_scaled, device)
    A_tgt = cut_activations(client, X_target_scaled, device)
    inv = _train_inversion(A_aux, X_aux_scaled, device, seed, epochs)

    inv.eval()
    with torch.no_grad():
        x_hat = (
            inv(torch.as_tensor(A_tgt, dtype=torch.float32, device=device))
            .cpu()
            .numpy()
        )

    mse_attack = float(np.mean((x_hat - X_target_scaled) ** 2))
    mse_mean = mean_predictor_mse(X_aux_scaled, X_target_scaled)
    entry = leakage_ratio(mse_attack, mse_mean)
    entry["per_feature_leakage_ratio"] = _per_feature_leakage(
        X_target_scaled, x_hat, X_aux_scaled
    )
    return entry
