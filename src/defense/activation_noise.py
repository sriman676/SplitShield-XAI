"""D2 — activation noise (02_THREAT_MODEL §3, targets T3/T4; H1 P7).

Add calibrated Gaussian noise to the cut-layer activations *before transmission*,
so the honest-but-curious server (and the inversion attack) only ever sees noised
activations. This is a transmission-time defence, so the noise is active at
inference too, not only during training.

**D2 is a sweep, not a single setting** (§3): a single σ is one point and no
insight. The stage sweeps ``cfg.training.sigma_sweep`` from "no effect" to "model
breaks" and plots the privacy/utility frontier (F3) — leakage ratio vs macro-F1.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.attack.activation_inference import activation_inference
from src.models.chained import ChainedModel


class NoisyClient(nn.Module):
    """Wrap a ``ClientNet`` to add N(0, σ) noise to its cut-layer output.

    The noise is seeded (a ``torch.Generator``) so a σ-point is reproducible
    across the macro-F1 and leakage measurements that share it.
    """

    def __init__(self, client: nn.Module, sigma: float, seed: int) -> None:
        super().__init__()
        self.client = client
        self.sigma = float(sigma)
        self.seed = int(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.client(x)
        if self.sigma <= 0.0:
            return z
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        noise = torch.randn(z.shape, generator=g).to(z.device) * self.sigma
        return z + noise


def noise_sweep(
    chained: ChainedModel,
    data: Any,
    sigmas: tuple[float, ...],
    device: str,
    seed: int,
    inversion_epochs: int,
    n_classes: int,
) -> list[dict[str, Any]]:
    """Privacy/utility frontier over σ: macro-F1 and T3 leakage at each noise level.

    Returns one dict per σ with the noised-model macro-F1 and the leakage-ratio
    entry (T3 run through the *noised* client, so the attacker inverts what it
    actually receives). ``sigma=0`` is the undefended reference point.
    """
    from src.training.common import eval_macro_f1

    points: list[dict[str, Any]] = []
    for sigma in sigmas:
        noisy = NoisyClient(chained.client, sigma, seed)
        noisy_chained = ChainedModel(noisy, chained.server)
        macro_f1 = eval_macro_f1(
            noisy_chained, data.X_test, data.y_test, n_classes, device
        )
        leak = activation_inference(
            noisy, data.X_train, data.X_test, device, seed, epochs=inversion_epochs
        )
        points.append(
            {"sigma": float(sigma), "macro_f1": float(macro_f1), "leakage": leak}
        )
    return points
