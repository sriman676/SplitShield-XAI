"""D2 activation-noise mechanism (02_THREAT_MODEL §3).

The D1 adversarial-training path (three model trainings) is exercised by the smoke
run; here we test the noise wrapper, which is the privacy-relevant logic.
"""

from __future__ import annotations

import torch
from torch import nn

from src.defense.activation_noise import NoisyClient


def test_noisy_client_zero_sigma_is_identity() -> None:
    client = nn.Identity()
    noisy = NoisyClient(client, sigma=0.0, seed=0)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert torch.allclose(noisy(x), x)


def test_noisy_client_adds_noise_and_is_reproducible() -> None:
    client = nn.Identity()
    x = torch.zeros(3, 4)
    a = NoisyClient(client, sigma=0.5, seed=7)(x)
    b = NoisyClient(client, sigma=0.5, seed=7)(x)
    c = NoisyClient(client, sigma=0.5, seed=8)(x)
    assert not torch.allclose(a, x)  # noise actually applied
    assert torch.allclose(a, b)  # same seed => reproducible
    assert not torch.allclose(a, c)  # different seed => different draw


def test_noise_magnitude_scales_with_sigma() -> None:
    client = nn.Identity()
    x = torch.zeros(200, 8)
    small = NoisyClient(client, sigma=0.1, seed=1)(x).std().item()
    large = NoisyClient(client, sigma=1.0, seed=1)(x).std().item()
    assert large > small
