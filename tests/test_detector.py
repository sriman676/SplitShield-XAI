"""D3 runtime detector: energy-threshold calibration + adversarial sensitivity.

Uses a fixed linear model and synthetic inputs (deterministic, no training): the
mechanism under test is the energy score + threshold, not a specific dataset.
"""

from __future__ import annotations

import torch
from torch import nn

from src.defense.detector import energy_scores, fit_energy_detector


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(8, 4)


def test_clean_flag_rate_matches_target_fpr() -> None:
    model = _model()
    torch.manual_seed(1)
    clean = (torch.randn(500, 8) * 3.0).numpy()  # peaky logits -> low energy
    det = fit_energy_detector(model, clean, target_fpr=0.1)
    fpr = det.flags(model, clean).mean()
    assert 0.05 <= fpr <= 0.15  # threshold set at the 90th percentile of clean


def test_off_manifold_inputs_flagged_more_than_clean() -> None:
    model = _model()
    torch.manual_seed(1)
    clean = (torch.randn(500, 8) * 3.0).numpy()  # peaky logits -> low energy
    attacked = (torch.randn(500, 8) * 0.01).numpy()  # flat logits -> high energy
    # core property: off-manifold inputs carry higher energy than clean.
    import numpy as np

    sc, sa = energy_scores(model, clean), energy_scores(model, attacked)
    assert sa.mean() > sc.mean()
    assert np.median(sa) > np.median(sc)
    # at a clean-median threshold, attacked is flagged strictly more than clean.
    det = fit_energy_detector(model, clean, target_fpr=0.5)
    assert det.flags(model, attacked).mean() > det.flags(model, clean).mean()
