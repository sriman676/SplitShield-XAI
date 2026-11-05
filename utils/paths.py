"""Path resolution with traversal guards.

All writes and trusted-artifact loads (scaler, checkpoints) must resolve inside
the project directory. ``ensure_within`` is the single guard used everywhere so
the joblib/torch trust boundary (08_ENGINEERING_STANDARDS §4) is enforced in one
place rather than re-checked ad hoc.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FIXTURES_DIR: Path = DATA_DIR / "fixtures"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
CONFIG_DIR: Path = PROJECT_ROOT / "config"


def ensure_within(path: Path | str, base: Path = PROJECT_ROOT) -> Path:
    """Resolve ``path`` and assert it is contained within ``base``.

    Args:
        path: Candidate path (absolute or relative to cwd).
        base: Directory the path must live inside. Defaults to the project root.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the resolved path escapes ``base`` (path traversal).
    """
    resolved = Path(path).resolve()
    base_resolved = base.resolve()
    if base_resolved not in resolved.parents and resolved != base_resolved:
        raise ValueError(f"Path {resolved} escapes allowed base {base_resolved}")
    return resolved


def run_dir(run_id: str) -> Path:
    """Return (and create) the run directory for ``run_id`` under ``runs/``."""
    d = ensure_within(RUNS_DIR / run_id, RUNS_DIR)
    (d / "metrics").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    return d
