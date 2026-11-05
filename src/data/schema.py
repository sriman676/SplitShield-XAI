"""Column contracts and dtype validation (04_DATA §4 step 1).

Fails loudly on schema surprises rather than letting pandas silently coerce. The
CIC CSVs ship leading spaces in column names and, in some releases, a duplicated
``Fwd Header Length`` header — both are handled explicitly here.
"""

from __future__ import annotations

import pandas as pd

from utils.logging_config import get_logger

_log = get_logger(__name__)


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from all column names (e.g. ``" Label"``)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_duplicate_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that appear more than once (after stripping)."""
    seen: dict[str, int] = {}
    for c in df.columns:
        seen[c] = seen.get(c, 0) + 1
    return [c for c, n in seen.items() if n > 1]


def assert_no_duplicate_columns(df: pd.DataFrame) -> None:
    """Raise if duplicate column names exist — never silently coerce them."""
    dups = find_duplicate_columns(df)
    if dups:
        raise ValueError(
            f"Duplicate column names {dups}; resolve explicitly in the registry "
            "entry before preprocessing (04_DATA §4 step 1)."
        )


def validate_columns(df: pd.DataFrame, expected: frozenset[str] | None) -> None:
    """Assert the expected feature columns are present. ``None`` skips the check.

    Args:
        df: The loaded frame (column names already stripped).
        expected: Required column names, or ``None`` to infer (fixtures).

    Raises:
        ValueError: If any expected column is missing.
    """
    if expected is None:
        return
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Schema mismatch: missing columns {sorted(missing)}")
    _log.info("Schema OK: %d expected columns present", len(expected))
