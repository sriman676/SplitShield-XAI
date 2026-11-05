"""D1 — adversarial training (02_THREAT_MODEL §3, H3).

Coordinator-capability gap (must be stated): a real split-learning client CANNOT
craft white-box adversarial examples against the full model — it does not hold
``ServerNet`` and cannot backpropagate through it. This defence is run by a
simulated coordinator holding BOTH halves, strictly stronger than the
architecture permits. Recorded in the stage metrics ``deviations`` and
11_LIMITATIONS §2.

Evaluation discipline (01_RESEARCH_DESIGN §7): the attacks used to *evaluate* the
defence are regenerated against the DEFENDED model — never the undefended model's
adversarial set. That happens in the stage; this module only hardens the model.

Simplification: one-shot adversarial augmentation — adversarial examples are
generated once against the undefended base model and mixed into the training set
(true labels kept), rather than regenerated every step against the model as it
hardens. Cheaper and sufficient for the fixture machinery; noted in
11_LIMITATIONS §2 as number-affecting. The mix ratio is ``cfg.training.adv_fraction``
(hashed), so it enters run comparability.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from config.base import Config
from src.attack.constraints import FeatureConstraints
from src.attack.fgsm_pgd import pgd
from src.data.preprocess import ProcessedData
from src.models.chained import ChainedModel
from src.training.split_sequential import train_split_sequential


def adversarial_train_split(
    data: ProcessedData, cfg: Config, device: str = "cpu"
) -> tuple[ChainedModel, ChainedModel, dict[str, Any]]:
    """Train an undefended base model, then a defended one on augmented data.

    Args:
        data: Processed dataset.
        cfg: Resolved run config (attack budget reused: ``epsilon``/``attack_iters``).
        device: Torch device string.

    Returns:
        ``(base_chained, defended_chained, defended_result)`` — the base is the
        undefended reference (H3's clean-accuracy control is base vs defended).
    """
    base, _ = train_split_sequential(data, cfg, device)

    fc = FeatureConstraints.from_run(
        data.feature_names,
        data.scaler,
        data.scaler.inverse_transform(data.X_train),
        corrected="derived_v2" in cfg.corrections,
    )
    # Harden against the non-benign training flows the attacker would target.
    nonbenign = np.where(np.asarray(data.y_train) != 0)[0]
    rng = np.random.default_rng(cfg.seed)
    frac = cfg.training.adv_fraction
    if not 0.0 <= frac <= 1.0:  # hashed as given, so never silently clip it
        raise ValueError(f"training.adv_fraction must be in [0, 1], got {frac}")
    m = int(frac * len(nonbenign))
    if m:
        sel = rng.choice(nonbenign, size=m, replace=False)
        x_adv = pgd(
            base,
            data.X_train[sel],
            cfg.training.epsilon,
            cfg.training.attack_iters,
            fc,
            device,
        )
        aug_X = np.concatenate([data.X_train, x_adv])
        aug_y = np.concatenate([data.y_train, data.y_train[sel]])  # keep TRUE labels
    else:
        aug_X, aug_y = data.X_train, data.y_train

    aug_data = replace(data, X_train=aug_X, y_train=aug_y)
    defended, res = train_split_sequential(aug_data, cfg, device)
    return base, defended, res
