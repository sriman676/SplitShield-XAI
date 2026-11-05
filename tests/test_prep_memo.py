"""run_pipeline._prep (2026-10-16 speed-up): a stored preprocessing result must be
exactly what preprocess_frame would return again, and its key must cover every
Config field preprocessing reads."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import run_pipeline as rp
from config.base import build_config, load_experiment
from src.data.preprocess import preprocess_frame
from src.data.sources import get_spec, load_raw

ROOT = Path(__file__).resolve().parent.parent


def test_key_covers_every_config_field_preprocessing_reads() -> None:
    src = (ROOT / "src" / "data" / "preprocess.py").read_text("utf-8")
    used = set(re.findall(r"\bcfg\.([a-z_]+(?:\.[a-z_]+)?)", src))
    assert used == set(rp.PREP_CFG_FIELDS), (
        "preprocess.py reads Config fields the _prep key does not cover (or no longer "
        f"reads some it does): {sorted(used ^ set(rp.PREP_CFG_FIELDS))}"
    )


def test_stored_result_is_identical_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rp, "_PREP_CACHE", {})
    monkeypatch.setattr(rp, "_RAW_CACHE", {})
    cfg = build_config(load_experiment("smoke.yaml"), seed=0, fast=True)
    spec = get_spec(cfg.dataset)
    a = rp._prep(spec, cfg, 0)
    fresh = preprocess_frame(load_raw(spec), spec, cfg, seed=0)
    for name in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test",
                 "idx_train", "idx_val", "idx_test"):  # fmt: skip
        assert np.array_equal(getattr(a, name), getattr(fresh, name)), name
    assert a.feature_names == fresh.feature_names and a.label_names == fresh.label_names
    # a different cut depth is the same preprocessing -> the same stored object
    deeper = replace(cfg, model=replace(cfg.model, cut_depth=cfg.model.cut_depth + 1))
    assert rp._prep(spec, deeper, 0) is a
    assert rp._prep(spec, cfg, 1) is not a  # another seed is another split
    with pytest.raises(ValueError):
        a.X_train[0, 0] = 1.0  # in-place change fails loudly
