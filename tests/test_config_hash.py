"""Config-hash contract (08_ENGINEERING_STANDARDS §6).

The hash MUST change with number-affecting fields (e.g. model width) and MUST NOT
change with speed/placement fields (num_workers), the seed, or the raw
batch/grad-accum decomposition when the effective batch is unchanged.
"""

from __future__ import annotations

from dataclasses import replace

from config.base import (
    HASH_EXCLUDED_FIELDS,
    Config,
    ModelConfig,
    RuntimeConfig,
    config_hash,
)


def _cfg() -> Config:
    return Config(experiment_id="t")


def test_hash_changes_with_model_width() -> None:
    a = _cfg()
    b = replace(a, model=replace(a.model, hidden_width=999))
    assert config_hash(a) != config_hash(b)


def test_hash_changes_with_adv_fraction() -> None:
    a = _cfg()
    b = replace(a, training=replace(a.training, adv_fraction=0.25))
    assert config_hash(a) != config_hash(b)


def test_hash_stable_across_num_workers() -> None:
    a = _cfg()
    b = replace(a, runtime=replace(a.runtime, num_workers=8, pin_memory=True))
    assert config_hash(a) == config_hash(b)


def test_hash_stable_across_seed() -> None:
    a = _cfg()
    b = replace(a, seed=4)
    assert config_hash(a) == config_hash(b)


def test_hash_tracks_effective_batch_not_decomposition() -> None:
    a = replace(_cfg(), runtime=RuntimeConfig(batch_size=256, grad_accum=1))
    same = replace(_cfg(), runtime=RuntimeConfig(batch_size=64, grad_accum=4))
    diff = replace(_cfg(), runtime=RuntimeConfig(batch_size=512, grad_accum=1))
    assert config_hash(a) == config_hash(same)  # same effective batch (256)
    assert config_hash(a) != config_hash(diff)  # effective batch 256 vs 512


def test_exclusion_list_is_the_runtime_fields_plus_seed() -> None:
    assert "seed" in HASH_EXCLUDED_FIELDS
    assert "num_workers" in HASH_EXCLUDED_FIELDS
    assert "device" in HASH_EXCLUDED_FIELDS
    # a number-affecting field must NOT be excluded
    assert "hidden_width" not in HASH_EXCLUDED_FIELDS
    assert ModelConfig().hidden_width == 128  # sanity: default present


def test_drift_instances_hash_neutral_at_registered_16() -> None:
    a = _cfg()
    assert a.training.drift_instances == 16
    b = replace(a, training=replace(a.training, drift_instances=128))
    assert config_hash(a) == config_hash(replace(a, training=replace(a.training, drift_instances=16)))
    assert config_hash(a) != config_hash(b)
