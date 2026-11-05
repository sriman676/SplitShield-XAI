"""T2 — GAN evasion, black-box (02_THREAT_MODEL T2, Stage 4b).

Which adversary: T2, the generative evading attacker. Goal: emit statistically
plausible traffic that the deployed IDS classifies BENIGN.

What it knows and where: black-box query access to the deployed model's
*decisions* only. The generator is trained solely against a discriminator on real
source-attack-class samples (realism); the target model enters ONLY as a label
oracle in ``_oracle_benign_rate`` — no gradients flow through it. This is the
human-approved faithful design: giving the GAN white-box gradients would leak a
capability T2 does not have (02_THREAT_MODEL §6).

Control / interpretability (§4.2, §4.5): ASR alone is meaningless here — a
generator can trivially emit benign-looking vectors. So ASR (oracle benign rate)
is reported *jointly* with the three realism metrics, which are attached as its
control block: per-feature 1-D Wasserstein distance to the real attack class,
domain-constraint violation rate before repair, and nearest-neighbour distance
(guards against a generator that memorizes training samples).

Strongest version NOT implemented: a query-efficient score-based attack that uses
oracle feedback to *optimize* evasion (e.g. NES gradient estimation). Here the
oracle only measures evasion; it does not steer the generator. ASR is therefore a
realism-constrained lower bound on what a query-optimizing T2 could reach.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch import nn

from src.attack.constraints import FeatureConstraints
from src.eval.metrics import metric_entry
from src.models.gan import Discriminator, Generator


def train_gan(
    X_source_scaled: np.ndarray,
    device: str,
    seed: int,
    steps: int,
    noise_dim: int = 16,
    batch_size: int = 64,
) -> Generator:
    """Train a vanilla non-saturating GAN on the source attack class (realism only)."""
    torch.manual_seed(seed)
    n_features = X_source_scaled.shape[1]
    g = Generator(noise_dim, n_features).to(device)
    d = Discriminator(n_features).to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    real = torch.as_tensor(X_source_scaled, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)

    for _ in range(steps):
        idx = rng.integers(0, len(real), size=min(batch_size, len(real)))
        rb = real[idx]
        z = torch.randn(len(rb), noise_dim, device=device)
        fake = g(z)
        # Discriminator step.
        opt_d.zero_grad()
        loss_d = bce(d(rb), torch.ones(len(rb), 1, device=device)) + bce(
            d(fake.detach()), torch.zeros(len(rb), 1, device=device)
        )
        loss_d.backward()
        opt_d.step()
        # Generator step (non-saturating).
        opt_g.zero_grad()
        loss_g = bce(d(fake), torch.ones(len(rb), 1, device=device))
        loss_g.backward()
        opt_g.step()
    return g


@torch.no_grad()
def generate(g: Generator, n: int, device: str, seed: int) -> np.ndarray:
    g.eval()
    torch.manual_seed(seed)
    z = torch.randn(n, g.noise_dim, device=device)
    return g(z).cpu().numpy()


@torch.no_grad()
def _oracle_benign_rate(
    target_model: nn.Module, X_scaled: np.ndarray, device: str, benign_index: int
) -> float:
    """Query the deployed model for decisions only; fraction called BENIGN."""
    target_model.eval()
    pred = target_model(
        torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
    ).argmax(1)
    return float((pred == benign_index).float().mean().cpu().numpy())


def _realism(
    gen_raw: np.ndarray, source_raw: np.ndarray, constraints: FeatureConstraints
) -> dict[str, Any]:
    """The three §4.5 realism metrics on raw (pre-repair) generated samples."""
    wass = [
        float(wasserstein_distance(gen_raw[:, j], source_raw[:, j]))
        for j in range(gen_raw.shape[1])
    ]
    # Nearest-neighbour euclidean distance to the real attack set (memorization guard).
    d2 = (
        np.sum(gen_raw**2, axis=1, keepdims=True)
        - 2 * gen_raw @ source_raw.T
        + np.sum(source_raw**2, axis=1)
    )
    nn_dist = np.sqrt(np.clip(d2.min(axis=1), 0, None))
    return {
        "per_feature_wasserstein": wass,
        "mean_wasserstein": float(np.mean(wass)),
        "violation_rate_pre_repair": constraints.absolute_violation_rate(gen_raw),
        "mean_nn_distance": float(np.mean(nn_dist)),
    }


def gan_evasion(
    target_model: nn.Module,
    X_source_scaled: np.ndarray,
    constraints: FeatureConstraints,
    device: str,
    seed: int,
    steps: int,
    n_generate: int = 256,
    benign_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train the GAN, generate+repair flows, and report ASR jointly with realism.

    Returns:
        ``(generated_scaled_repaired, metrics_entry)``. The ASR value is the oracle
        benign rate over generated flows; its control block carries the realism
        metrics that make it interpretable (§4.5).
    """
    g = train_gan(X_source_scaled, device, seed, steps)
    gen_scaled = generate(g, n_generate, device, seed)
    gen_raw = constraints.to_raw(gen_scaled)
    source_raw = constraints.to_raw(X_source_scaled)

    realism = _realism(gen_raw, source_raw, constraints)

    # Repair to realistic, in-bounds flows before querying the oracle.
    gen_repaired_scaled = constraints.to_scaled(constraints.repair_raw(gen_raw))
    asr = _oracle_benign_rate(target_model, gen_repaired_scaled, device, benign_index)

    entry = metric_entry(
        "attack_success_rate",
        asr,
        control={"name": "gan_realism", **realism},
        denominator=int(n_generate),
    )
    return gen_repaired_scaled, entry
