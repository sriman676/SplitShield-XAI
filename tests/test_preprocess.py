"""Preprocessing gates (08_ENGINEERING_STANDARDS §6).

The leakage test and the rare-class-preservation test are the two blocking gates
for Task 1: everything downstream is worthless if they fail.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from config.base import Config, SubsampleConfig
from src.data.preprocess import preprocess_frame
from src.data.sources import get_spec, load_raw


@pytest.fixture(scope="module")
def processed():
    spec = get_spec("fixture")
    df = load_raw(spec)
    # small caps so both branches (cap + rare-floor) are exercised
    cfg = Config(
        experiment_id="t",
        dataset="fixture",
        subsample=SubsampleConfig(rare_floor=15, class_cap=30),
    )
    return preprocess_frame(df, spec, cfg, seed=0)


def test_dedup_drops_exact_duplicates(processed) -> None:
    assert processed.stats["n_duplicates_dropped"] == 5


def test_invalid_rows_dropped(processed) -> None:
    # 3 deliberate defect rows: one NaN, one +Inf, one -Inf
    assert processed.stats["n_invalid_dropped"] == 3


def test_no_train_test_leakage(processed) -> None:
    """BLOCKING: no exact row appears in both train and test after preprocessing."""
    fr = processed.frame[processed.feature_names]
    train_rows = {tuple(r) for r in fr.iloc[processed.idx_train].to_numpy()}
    test_rows = {tuple(r) for r in fr.iloc[processed.idx_test].to_numpy()}
    assert train_rows.isdisjoint(test_rows)


def test_rare_classes_preserved(processed) -> None:
    """BLOCKING: classes at/below rare_floor keep 100% of their rows."""
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        for fam, n in processed.stats["class_counts"][split].items():
            counts[fam] = counts.get(fam, 0) + n
    # fixture counts (post-clean) for classes <= rare_floor=15
    assert counts["BruteForce"] == 15
    assert counts["WebAttack"] == 12
    assert counts["Bot"] == 10
    assert counts["Infiltration"] == 8
    assert counts["Heartbleed"] == 8
    # a capped class must NOT keep all its rows
    assert counts["BENIGN"] == 30  # capped from 87


def test_scaler_fit_on_train_only(processed) -> None:
    fr = processed.frame[processed.feature_names]
    train_mean = fr.iloc[processed.idx_train].to_numpy(dtype=float).mean(axis=0)
    assert np.allclose(processed.scaler.mean_, train_mean)


def test_every_class_in_all_splits(processed) -> None:
    for split in ("train", "val", "test"):
        present = set(processed.stats["class_counts"][split])
        assert present == set(processed.label_names)


def test_temporal_split_is_time_ordered() -> None:
    """split='temporal' must be time-ordered with no leakage across time.

    Guards the footgun where cfg.split was accepted but silently ignored (random).
    """
    spec = get_spec("fixture")
    df = load_raw(spec).copy()
    # monotone timestamps in row order, then shuffle so time != row position.
    df["Timestamp"] = pd.date_range("2017-07-03", periods=len(df), freq="s")
    df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    tspec = replace(
        spec,
        timestamp_col="Timestamp",
        id_columns=tuple(set(spec.id_columns) | {"Timestamp"}),
    )
    cfg = Config(
        experiment_id="t",
        dataset="fixture",
        split="temporal",
        subsample=SubsampleConfig(rare_floor=15, class_cap=30),
    )
    proc = preprocess_frame(df, tspec, cfg, seed=0)

    assert "__ts__" not in proc.feature_names  # timestamp is not a feature
    ts = pd.to_datetime(proc.frame["__ts__"])
    # every train time precedes every val time precedes every test time
    assert ts.iloc[proc.idx_train].max() <= ts.iloc[proc.idx_val].min()
    assert ts.iloc[proc.idx_val].max() <= ts.iloc[proc.idx_test].min()


def test_temporal_split_requires_timestamp() -> None:
    """split='temporal' without a usable timestamp fails loud (never silent random)."""
    spec = get_spec("fixture")
    df = load_raw(spec)
    cfg = Config(experiment_id="t", dataset="fixture", split="temporal")
    with pytest.raises(ValueError, match="temporal"):
        preprocess_frame(df, spec, cfg, seed=0)


def test_duplicate_columns_dropped_only_when_identical() -> None:
    from src.data.sources import drop_verified_duplicates

    same = pd.DataFrame([[1, 2, 1], [3, 4, 3]], columns=["a", "b", "a"])
    out = drop_verified_duplicates(same, ("a",))
    assert list(out.columns) == ["a", "b"]
    differ = pd.DataFrame([[1, 2, 9], [3, 4, 3]], columns=["a", "b", "a"])
    with pytest.raises(ValueError, match="differs"):
        drop_verified_duplicates(differ, ("a",))


def test_text_identifier_column_does_not_drop_rows() -> None:
    """A text id column (NetFlow 'Attack', DIAD 'Flow ID') is coerced to NaN by the
    invalid-value check; it is dropped in step 5, so it must not veto rows in step 4
    (it used to drop every row: E8 failed with an empty training set)."""
    import numpy as np
    import pandas as pd

    from config.base import build_config, load_experiment
    from src.data.preprocess import preprocess_frame
    from src.data.sources import DatasetSpec

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "Attack": rng.choice(["Benign", "DDoS", "Brute Force"], size=n),
            "Label": rng.integers(0, 2, size=n),
        }
    )
    df.loc[0, "f1"] = np.inf  # a genuinely bad feature row is still dropped
    spec = DatasetSpec(
        id="t", label_col="Label", id_columns=("Attack",), label_scheme="binary"
    )
    cfg = build_config(load_experiment("smoke.yaml"), seed=0)
    data = preprocess_frame(df, spec, cfg, seed=0)
    assert data.stats["n_invalid_dropped"] == 1
    assert "Attack" not in data.feature_names
    assert data.stats["n_after_subsample"] > 300  # was 0: every row dropped
