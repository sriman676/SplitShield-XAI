"""Deterministic seeding for random, numpy, torch, and dataloader workers.

Seeds control weight init, batch order, the train/val/test split, the Dirichlet
partition draw, and attack randomness (01_RESEARCH_DESIGN §5). The seed is NOT a
config-hash field: it is the axis results are averaged over.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from utils.logging_config import get_logger

_log = get_logger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs. Logs (does not fake) where full determinism is unavailable.

    Args:
        seed: The seed value.
        deterministic: Request deterministic torch algorithms where supported.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:  # some kernels have no deterministic impl
                _log.warning("Full determinism unavailable: %s", exc)
    except ImportError:
        _log.warning("torch not installed; seeded numpy/random only")


def worker_init_fn(worker_id: int) -> None:
    """Seed a dataloader worker so ``num_workers > 0`` stays reproducible."""
    state = cast("Any", np.random.get_state())  # legacy tuple: (str, ndarray, ...)
    seed = (int(state[1][0]) + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int) -> Any:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling, or None."""
    try:
        import torch

        g = torch.Generator()
        g.manual_seed(seed)
        return g
    except ImportError:
        return None


def seeded_loader_kwargs(seed: int) -> dict[str, Callable[[int], None] | Any]:
    """DataLoader kwargs that make worker shuffling reproducible."""
    return {"worker_init_fn": worker_init_fn, "generator": make_generator(seed)}
