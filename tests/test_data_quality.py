"""Data-quality profile (scripts/data_quality.py) and its report section."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data_quality import findings, profile
from src.eval.final_report import _data_quality


def test_profile_finds_the_known_problems(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Flow ID": [f"f{i}" for i in range(8)],
            "Flow Duration": [10, 10, 5, -1, 7, 8, 9, 3],
            "Copy": [10, 10, 5, -1, 7, 8, 9, 3],
            "Const": [1] * 8,
            "Bytes/s": [1.0, 1.0, np.inf, 2.0, np.nan, 3.0, 4.0, 5.0],
            "Label": ["A", "A", "B", "B", "A", "B", "A", "B"],
        }
    )
    df.loc[1, "Label"] = "B"  # row 1 = row 0 features, other label
    p = profile(df.drop(columns="Flow ID").assign(**{"Flow ID": 0}), "Label")
    assert p["conflicting_label_rows"] == 2
    assert p["rows_with_nan_or_inf"] == 2
    assert p["negative_in_nonnegative_columns"] == {"Flow Duration": 1}
    assert "Const" in p["constant_columns"]
    assert ["Flow Duration", "Copy"] in p["duplicated_columns"]
    assert "Flow ID" in p["identifier_columns"]
    # a text column never makes every row "NaN"
    q = profile(df.assign(**{"Src IP": [f"10.0.0.{i}" for i in range(8)]}), "Label")
    assert q["rows_with_nan_or_inf"] == 2 and q["text_columns"] == ["Flow ID", "Src IP"]
    f = findings(p)
    assert any("label" in x for x in f) and any("negative" in x for x in f)

    (tmp_path / "data_quality.json").write_text(
        json.dumps({"max_rows": 10, "handled_by_pipeline": {"d": "x"},
                    "datasets": {"demo": {**p, "rows_total": 8, "findings": f}}})
    )  # fmt: skip
    lines = _data_quality(tmp_path)
    assert lines[0] == "## Data quality" and any("| demo | 8 |" in ln for ln in lines)
