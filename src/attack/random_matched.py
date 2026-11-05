"""THE CONTROL: random-matched perturbation (01_RESEARCH_DESIGN §3.1).

Research purpose: the single most important control in the project (H2). For
every adversarial example it produces a random perturbation of the *same norm*,
on the *same mutable features*, projected through the *same* constraint set — so
that any excess explanation drift or attack success from the real attack is
attributable to its adversarial *direction*, not merely to perturbation
magnitude. Operates only on this repo's models and public data (02_THREAT_MODEL
§5); written before any real attack (03_ARCHITECTURE §1).

Space convention (02_THREAT_MODEL §4.2 rule 1): perturbations are measured and
matched in SCALED (StandardScaler) space — the space the FGSM/PGD budget lives
in — then projected to raw units for the domain constraints and rescaled. The
matched norm is the *pre-projection* norm, per §3.1.
"""

from __future__ import annotations

import numpy as np

from src.attack.constraints import FeatureConstraints


def perturbation_norm(
    x: np.ndarray, x_adv: np.ndarray, mask: np.ndarray, p: float
) -> np.ndarray:
    """Per-instance norm of the masked perturbation ``x_adv - x`` (p = inf or 2)."""
    delta = (np.asarray(x_adv) - np.asarray(x)) * mask
    if p == np.inf:
        return np.max(np.abs(delta), axis=1)
    return np.linalg.norm(delta, ord=2, axis=1)


def random_matched(
    x_scaled: np.ndarray,
    x_adv_scaled: np.ndarray,
    mask: np.ndarray,
    constraints: FeatureConstraints,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random perturbation matched to ``x_adv``'s per-instance norm (§3.1).

    Args:
        x_scaled: Clean instances in scaled space, shape (n, d).
        x_adv_scaled: The adversarial examples to match, same shape.
        mask: Boolean perturbable-feature mask (constraints.mask); random noise is
            confined to these features, exactly as the attack was.
        constraints: Applied identically to the attack so both are realizable.
        p: Norm to match — ``np.inf`` for FGSM/PGD, ``2`` for the GAN attack.
        rng: Seeded generator (attack randomness is seed-controlled, §5).

    Returns:
        Random-matched examples in scaled space.
    """
    x = np.asarray(x_scaled, dtype=np.float64)
    target = perturbation_norm(x, x_adv_scaled, mask, p)  # (n,)
    n, d = x.shape
    delta = np.zeros((n, d), dtype=np.float64)
    m = mask.astype(bool)

    if p == np.inf:
        # ±target on every mutable feature => L-inf norm exactly target.
        signs = rng.choice((-1.0, 1.0), size=(n, d))
        delta[:, m] = signs[:, m] * target[:, None]
    else:
        # random direction on mutable features, scaled to the L2 target.
        g = rng.standard_normal((n, d)) * m
        gnorm = np.linalg.norm(g, ord=2, axis=1, keepdims=True)
        gnorm[gnorm == 0] = 1.0
        delta = g / gnorm * target[:, None]

    x_rand = x + delta
    return constraints.project_scaled(x_rand, x)
