"""Zeek FlowMeter flow logs -> CICFlowMeter features (owner 2026-10-24; documented units)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.soc.columns import (
    FLOWMETER,
    align,
    detect_format,
    from_flowmeter,
    is_flowmeter,
)


def _row() -> pd.DataFrame:
    cols = {c for v in FLOWMETER.values() for c in ((v,) if isinstance(v, str) else v[1:] if v[0] == "sq" else v)}
    d: dict[str, list[object]] = {c: [2.0] for c in cols}
    d |= {"flow_duration": ["0 days 00:00:01.500000"], "fwd_pkts_payload.std": [3.0], "uid": ["C1"], "Label": [0]}
    return pd.DataFrame(d)


def test_units_and_derived_columns() -> None:
    df = _row()
    assert is_flowmeter(list(df.columns)) and detect_format(list(df.columns)).startswith("Zeek FlowMeter")
    out = from_flowmeter(df)
    assert out["Flow Duration"].iloc[0] == pytest.approx(1.5e6)  # seconds -> microseconds
    assert out["Flow IAT Mean"].iloc[0] == 2.0  # microseconds in both
    assert out["PSH Flag Count"].iloc[0] == 4.0  # fwd + bwd
    assert out["Label"].iloc[0] == 0 and out["uid"].iloc[0] == "C1"  # other columns kept


def test_align_matches_every_deployed_feature() -> None:
    needed = ["Flow Duration", *FLOWMETER]
    _df, info = align(_row(), needed)
    assert info["missing"] == [] and info["share"] == 1.0
