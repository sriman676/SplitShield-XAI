"""Run manifest: identity, provenance, environment (03_ARCHITECTURE §5, 06 §3).

Every stage writes one manifest. A dirty git tree auto-marks the run
``preliminary`` so its numbers never enter a report table.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import yaml

from config.base import Config, config_hash
from utils.device import get_compute_tier, get_primary_device
from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT, run_dir

_log = get_logger(__name__)


def _git(*args: str) -> str:
    """Run a git command in the project root; return stripped stdout or ''."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        _log.warning("git %s failed: %s", " ".join(args), exc)
        return ""


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%MZ")


def make_run_id(cfg: Config, tag: str = "") -> str:
    """Build a unique run id: ``<exp>__<tag>__seed<N>__<utc>__<hash>``."""
    parts = [cfg.experiment_id]
    if tag:
        parts.append(tag)
    parts += [f"seed{cfg.seed}", _utc_stamp(), config_hash(cfg)]
    return "__".join(parts)


def write_manifest(
    cfg: Config, run_id: str, dataset_stats: dict[str, Any]
) -> dict[str, Any]:
    """Write ``manifest.json`` and ``config.resolved.yaml`` for a run.

    Args:
        cfg: The resolved run config.
        run_id: The run identifier (see ``make_run_id``).
        dataset_stats: Row/dedup/class-count stats for the dataset block.

    Returns:
        The manifest dict (also written to disk).
    """
    dirty = bool(_git("status", "--porcelain"))
    preliminary = dirty or cfg.runtime.fast
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": cfg.experiment_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "config_hash": config_hash(cfg),
        "config_resolved": asdict(cfg),
        "seed": cfg.seed,
        "dataset": {"id": cfg.dataset, "version": cfg.dataset_version, **dataset_stats},
        "label_schema": cfg.label_schema,
        "environment": {
            "python": platform.python_version(),
            "torch": _pkg("torch"),
            "shap": _pkg("shap"),
            "sklearn": _pkg("scikit-learn"),
            "os": platform.platform(),
            "device": get_primary_device(),
            "compute_tier": get_compute_tier(),
        },
        "fast_mode": cfg.runtime.fast,
        "deviations": [],
        "preliminary": preliminary,
    }

    d = run_dir(run_id)
    with open(d / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    with open(d / "config.resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)
    return manifest
