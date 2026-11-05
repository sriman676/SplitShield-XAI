"""T1 — white-box FGSM / PGD evasion (02_THREAT_MODEL T1, Stage 4a).

Which adversary: T1, the evading attacker. It holds the full model (both halves)
and its gradients — the knowledge enters where ``chained`` is differentiated
below. This is the strongest, least realistic adversary: a real evader does not
hold the defender's weights (02_THREAT_MODEL §2). It is the standard worst-case
upper bound on evadability, not a field-conditions claim.

Objective: targeted toward BENIGN — the T1 goal is "classified BENIGN", which is
exactly what ASR measures (§4.2). We descend the cross-entropy toward the benign
class, so a flip counts as evasion by construction rather than by luck.

Control (makes the output interpretable): the random-matched perturbation of the
same L-infinity norm (``random_matched.py``, §3.1). ASR is reported only over
clean-correct non-benign samples (§3.5).

Strongest version NOT implemented: an attack in true packet space with full
flow-level realizability; here perturbations are feature-space and constrained by
02_THREAT_MODEL §4, so ASR is an upper bound (11_LIMITATIONS §1).

Space (02_THREAT_MODEL §4.2 rule 1): the L-infinity budget ``eps`` is in SCALED
space; constraints are applied in raw units each step via ``project_scaled``. The
achieved raw per-feature perturbation is reported separately (§4.3).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.attack.constraints import FeatureConstraints


def _project(
    x_adv: np.ndarray, x0: np.ndarray, eps: float, constraints: FeatureConstraints
) -> np.ndarray:
    """L-inf ball projection (scaled) then domain-constraint projection (raw)."""
    x_adv = np.clip(x_adv, x0 - eps, x0 + eps)
    return constraints.project_scaled(x_adv, x0)


def pgd(
    chained: nn.Module,
    x_scaled: np.ndarray,
    eps: float,
    iters: int,
    constraints: FeatureConstraints,
    device: str,
    benign_index: int = 0,
    step: float | None = None,
) -> np.ndarray:
    """Constrained PGD toward BENIGN. ``iters=1`` with ``step=eps`` is FGSM.

    Returns the adversarial examples in scaled (model) space.
    """
    was_training = chained.training
    chained.eval()
    x0 = np.asarray(x_scaled, dtype=np.float64)
    x_adv = x0.copy()
    alpha = step if step is not None else (2.5 * eps / max(iters, 1))
    target = torch.full((len(x0),), benign_index, dtype=torch.long, device=device)

    for _ in range(max(iters, 1)):
        t = torch.tensor(x_adv, dtype=torch.float32, device=device, requires_grad=True)
        loss = nn.functional.cross_entropy(chained(t), target)  # minimize -> benign
        (grad,) = torch.autograd.grad(loss, t)
        x_adv = x_adv - alpha * np.sign(grad.detach().cpu().numpy())
        x_adv = _project(x_adv, x0, eps, constraints)

    if was_training:
        chained.train()
    return x_adv


def fgsm(
    chained: nn.Module,
    x_scaled: np.ndarray,
    eps: float,
    constraints: FeatureConstraints,
    device: str,
    benign_index: int = 0,
) -> np.ndarray:
    """Single-step FGSM toward BENIGN (constrained)."""
    return pgd(chained, x_scaled, eps, 1, constraints, device, benign_index, step=eps)
