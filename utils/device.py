"""Three-tier device detection (07_COMPUTE §2). Called once at startup.

Never call ``torch.cuda.is_available()`` inline elsewhere. The tier is recorded
in every manifest but, per 07_COMPUTE §1, it changes speed only — never what a
run computes.

torch is imported lazily and guarded: absent torch degrades to ``cpu_only`` with
a warning so the scaffold/smoke path runs before the full env is installed.
"""

from __future__ import annotations

import os

from utils.logging_config import get_logger

_log = get_logger(__name__)

DEDICATED_GPU = "dedicated_gpu"
IGPU = "igpu"
CPU_ONLY = "cpu_only"

_TIER: str | None = None
_PRIMARY: str | None = None


def _detect() -> tuple[str, str]:
    """Return ``(compute_tier, primary_device)``. Honors COMPUTE_TIER_OVERRIDE."""
    override = os.environ.get("COMPUTE_TIER_OVERRIDE", "").strip()
    if override:
        _log.info("COMPUTE_TIER_OVERRIDE=%s", override)
        primary = {DEDICATED_GPU: "cuda", IGPU: "mps", CPU_ONLY: "cpu"}.get(
            override, "cpu"
        )
        return override, primary

    try:
        import torch
    except ImportError:
        _log.warning("torch not installed; COMPUTE_TIER=cpu_only")
        return CPU_ONLY, "cpu"

    if torch.cuda.is_available():
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        _log.info("Detected dedicated GPU (%s)", backend)
        return DEDICATED_GPU, "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return IGPU, "mps"
    try:
        import torch_directml

        if torch_directml.is_available():
            return IGPU, "directml"
    except ImportError:
        pass
    return CPU_ONLY, "cpu"


def _ensure() -> None:
    global _TIER, _PRIMARY
    if _TIER is None:
        _TIER, _PRIMARY = _detect()


def get_compute_tier() -> str:
    """Return the detected compute tier string."""
    _ensure()
    assert _TIER is not None
    return _TIER


def get_primary_device() -> str:
    """Return the primary device string (e.g. ``cuda``, ``mps``, ``cpu``)."""
    _ensure()
    assert _PRIMARY is not None
    return _PRIMARY


def get_secondary_device() -> str:
    """Return the device for the client half under cross-device split (CPU)."""
    return "cpu"
