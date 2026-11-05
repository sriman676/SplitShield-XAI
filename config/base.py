"""Frozen config, .env overrides, and the config hash (06_EXPERIMENTS §2).

The config hash is the mechanism that keeps the two build/train machines
comparable (07_COMPUTE §1). It covers every field that can affect a number and
*excludes* speed/placement fields. The exclusion list is a single test-covered
constant (``HASH_EXCLUDED_FIELDS``), not an ad-hoc filter.

Two deliberate subtleties (see planning checkpoint):
  * ``seed`` is excluded — it is the replicate axis results are averaged over.
  * The hash covers *effective* batch size (``batch_size * grad_accum``), not the
    raw decomposition, so a laptop using gradient accumulation and a GPU box
    using one big batch hash identically.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from utils.paths import CONFIG_DIR

# --- sub-configs ------------------------------------------------------------


@dataclass(frozen=True)
class SubsampleConfig:
    rare_floor: int = 5_000
    class_cap: int = 60_000


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: int = 4
    hidden_width: int = 128
    dropout: float = 0.2
    norm: str = "none"  # none | layer | batch
    cut_depth: int = 2  # IV only in E4; held constant elsewhere


@dataclass(frozen=True)
class TrainingConfig:
    scheme: str = "centralized"  # centralized | split_sequential | fedavg | splitfed
    clients: int = 4
    dirichlet_alpha: float = 1.0
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    schedule: str = "cosine"
    max_epochs: int = 50
    sl_cycles: int = 30
    patience: int = 8
    local_epochs: int = 1
    imbalance: str = "class_weight"  # class_weight | smote
    scaler: str = "standard"  # standard | minmax
    keep_dest_port: bool = True
    rf_n_estimators: int = 200
    gan_steps: int = 2000
    attack_iters: int = 20
    adv_fraction: float = 0.5  # D1: share of non-benign flows adv-augmented (hashed)
    inversion_epochs: int = 200  # T3 inversion training (number-affecting, hashed)
    epsilon: float = 0.1
    sigma_sweep: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
    shap_background: int = 100
    # stage-6 (H2) instance count; 16 = registered design, left out of the hash at 16 so
    # registered run ids never change (H2c, owner 2026-10-24).
    drift_instances: int = 16


@dataclass(frozen=True)
class RuntimeConfig:
    """Speed/placement only. EXCLUDED from the config hash (07_COMPUTE §1)."""

    device: str = "auto"
    compute_tier: str = ""
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False
    amp: bool = False
    batch_size: int = 256
    grad_accum: int = 1
    cross_device_split: bool = False
    log_level: str = "INFO"
    fast: bool = False
    resume: bool = False


@dataclass(frozen=True)
class Config:
    experiment_id: str
    dataset: str = "cicids2017-improved"
    dataset_version: str = ""
    label_schema: str = "family9"
    split: str = "random"  # random | temporal
    attacks: tuple[str, ...] = ()
    # corrected follow-up switches (owner 2026-10-14): "derived_v2" (verified DERIVED
    # formulas + pre-normalized guard), "attribution_converge". Empty = registered
    # design; left out of the hash when empty so registered run ids never change.
    corrections: tuple[str, ...] = ()
    seed: int = 0  # EXCLUDED from hash
    subsample: SubsampleConfig = field(default_factory=SubsampleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


# --- exclusion list (test-covered constant) ---------------------------------

HASH_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {f.name for f in fields(RuntimeConfig)} | {"seed", "runtime"}
)

# ``--fast`` reductions. These DO affect numbers, so fast runs are always
# marked preliminary and never enter a report table (06_EXPERIMENTS §6).
FAST_OVERRIDES: dict[str, dict[str, Any]] = {
    "training": {
        "max_epochs": 2,
        "sl_cycles": 2,
        "patience": 1,
        "gan_steps": 10,
        "attack_iters": 2,
        "inversion_epochs": 20,
        "shap_background": 20,
        "rf_n_estimators": 10,
        "sigma_sweep": (0.0, 0.5),
    }
}


# --- hashing ----------------------------------------------------------------


def _hash_payload(cfg: Config) -> dict[str, Any]:
    """Build the dict that gets hashed: number-affecting fields only."""
    payload: dict[str, Any] = {}
    for f in fields(cfg):
        if f.name in HASH_EXCLUDED_FIELDS:
            continue
        if f.name == "corrections" and not cfg.corrections:
            continue
        val = getattr(cfg, f.name)
        payload[f.name] = asdict(val) if hasattr(val, "__dataclass_fields__") else val
    if cfg.training.drift_instances == 16:
        payload["training"].pop("drift_instances", None)
    # effective batch is hash-relevant; the raw decomposition is not.
    payload["effective_batch_size"] = cfg.runtime.batch_size * cfg.runtime.grad_accum
    return payload


def config_hash(cfg: Config) -> str:
    """Return a 12-char hex hash over the number-affecting config fields."""
    blob = json.dumps(_hash_payload(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --- loading ----------------------------------------------------------------


def _apply_env(runtime: RuntimeConfig) -> RuntimeConfig:
    """Overlay .env speed/placement values onto a RuntimeConfig."""
    load_dotenv()

    def _b(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        return default if v is None else v.strip().lower() in {"1", "true", "yes"}

    return replace(
        runtime,
        num_workers=int(os.environ.get("NUM_WORKERS", runtime.num_workers)),
        pin_memory=_b("PIN_MEMORY", runtime.pin_memory),
        log_level=os.environ.get("LOG_LEVEL", runtime.log_level),
    )


def _nested(cls: type, raw: dict[str, Any]) -> Any:
    """Build a frozen sub-config from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    if "sigma_sweep" in kwargs and isinstance(kwargs["sigma_sweep"], list):
        kwargs["sigma_sweep"] = tuple(kwargs["sigma_sweep"])
    return cls(**kwargs)


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Load a raw experiment YAML (with sweeps/seeds still expanded by caller)."""
    p = Path(path)
    if not p.is_absolute():
        p = CONFIG_DIR / "experiments" / p
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def experiment_for_dataset(dataset: str, fast: bool = False) -> dict[str, Any]:
    """Raw experiment dict to train ``dataset`` with outside the pipeline
    (self-improve, deploy): the first experiment YAML whose ``dataset`` matches,
    so real data gets its real subsample caps and model size. ``--fast`` and the
    fixture use ``smoke.yaml`` (tiny caps, preliminary)."""
    if not fast and dataset != "fixture":
        for p in sorted((CONFIG_DIR / "experiments").glob("*.yaml")):
            raw = load_experiment(p)
            if raw.get("dataset") == dataset and "matrix" not in raw:
                return {**raw, "dataset": dataset}
    raw = {**load_experiment("smoke.yaml"), "dataset": dataset}
    if not fast and dataset != "fixture":  # no YAML: real data, default caps/size
        raw.pop("subsample", None)
        raw.pop("model", None)
    return raw


def build_config(
    raw: dict[str, Any],
    seed: int,
    fast: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Materialize a single run's frozen Config from a raw experiment dict.

    Args:
        raw: Parsed experiment YAML.
        seed: The seed for this run.
        fast: Apply ``FAST_OVERRIDES`` and mark the run preliminary downstream.
        overrides: Per-run sweep point, e.g. ``{"model": {"cut_depth": 3}}``.

    Returns:
        A frozen ``Config``.
    """
    sub = dict(raw.get("subsample", {}))
    model = dict(raw.get("model", {}))
    training = dict(raw.get("training", {}))

    # sweeps like {cut_depth: [...]} live under `sweep`; the caller picks a point
    for key, val in (overrides or {}).items():
        target = {"subsample": sub, "model": model, "training": training}.get(key)
        if target is not None and isinstance(val, dict):
            target.update(val)

    if fast:
        for section, vals in FAST_OVERRIDES.items():
            {"subsample": sub, "model": model, "training": training}[section].update(
                vals
            )

    runtime = _apply_env(RuntimeConfig(fast=fast))

    return Config(
        experiment_id=raw["id"],
        dataset=raw.get("dataset", "cicids2017-improved"),
        dataset_version=raw.get("dataset_version", ""),
        label_schema=raw.get("label_schema", "family9"),
        split=raw.get("split", "random"),
        attacks=tuple(raw.get("attacks", [])),
        corrections=tuple(raw.get("corrections", [])),
        seed=seed,
        subsample=_nested(SubsampleConfig, sub),
        model=_nested(ModelConfig, model),
        training=_nested(TrainingConfig, training),
        runtime=runtime,
    )
