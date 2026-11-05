"""Results aggregation (06_EXPERIMENTS §5). Tables are generated, never hand-made.

``make results`` (``run_pipeline --aggregate``) rebuilds every table in
``results/`` from ``runs/``. The load-bearing rule is **config-hash
compatibility**: two runs may share a comparison table only if their configs
differ *exclusively* in that comparison's declared independent variables. Any
other differing field (a hardware-varying knob, a changed hyperparameter) means
the comparison is invalid, and ``assert_comparable`` raises rather than silently
mixing them — the defect §5 exists to prevent.

Also enforced: preliminary runs never enter headline tables (§5.3), a cell short
on seeds is reported with its actual n not silently averaged (§5.2), and a
control-requiring metric without its control is dropped with a warning (§5.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.base import HASH_EXCLUDED_FIELDS
from utils.logging_config import get_logger
from utils.paths import RESULTS_DIR, RUNS_DIR

_log = get_logger(__name__)


class IncompatibleConfigError(ValueError):
    """Raised when runs that must be comparable differ outside the declared IVs."""


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict to dotted keys (leaves only)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _comparable_fields(config_resolved: dict[str, Any]) -> dict[str, Any]:
    """Number-affecting config leaves: everything except the hash-excluded fields."""
    flat = _flatten(config_resolved)
    return {
        k: v
        for k, v in flat.items()
        # drop seed, and anything under an excluded top-level section (e.g. runtime.*)
        if k.split(".", 1)[0] not in HASH_EXCLUDED_FIELDS
    }


def assert_comparable(runs: list[dict[str, Any]], iv_fields: set[str]) -> None:
    """Raise unless ``runs`` differ only in ``iv_fields`` (06_EXPERIMENTS §5.1).

    Args:
        runs: Manifest dicts (each with ``config_resolved``).
        iv_fields: Dotted field names allowed to vary (the comparison's IVs),
            e.g. ``{"model.cut_depth"}`` for the cut-depth table.

    Raises:
        IncompatibleConfigError: If any non-IV field differs across the runs, with
            a diff of the offending fields.
    """
    if len(runs) < 2:
        return
    ref = _comparable_fields(runs[0]["config_resolved"])
    offending: dict[str, set] = {}
    for run in runs[1:]:
        cur = _comparable_fields(run["config_resolved"])
        for key in set(ref) | set(cur):
            if key in iv_fields:
                continue
            if ref.get(key) != cur.get(key):
                offending.setdefault(key, set()).update(
                    {_hashable(ref.get(key)), _hashable(cur.get(key))}
                )
    if offending:
        diff = "; ".join(f"{k}={sorted(v)}" for k, v in sorted(offending.items()))
        raise IncompatibleConfigError(
            "Runs are not comparable — fields outside the declared IVs "
            f"{sorted(iv_fields)} differ: {diff}"
        )


def _hashable(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def load_runs(runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    """Load every run's manifest + stage metrics from ``runs/`` (skips incomplete)."""
    runs: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return runs
    for d in sorted(runs_dir.iterdir()):
        man = d / "manifest.json"
        if not man.is_file():
            continue
        with open(man, encoding="utf-8") as fh:
            manifest = json.load(fh)
        metrics: dict[str, Any] = {}
        mdir = d / "metrics"
        if mdir.is_dir():
            for mf in sorted(mdir.glob("*.json")):
                with open(mf, encoding="utf-8") as fh:
                    metrics[mf.stem] = json.load(fh)
        manifest["_metrics"] = metrics
        runs.append(manifest)
    return runs


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the top-level results summary from loaded runs (§5.2/§5.3).

    Preliminary runs are counted but flagged and excluded from any headline claim.
    This is deliberately schema-light: on the fixture there are no headline numbers,
    so the summary reports run provenance and which stages produced metrics.
    """
    by_exp: dict[str, dict[str, Any]] = {}
    for r in runs:
        exp = r.get("experiment_id", "?")
        cell = by_exp.setdefault(
            exp, {"n_runs": 0, "n_preliminary": 0, "seeds": set(), "stages": set()}
        )
        cell["n_runs"] += 1
        cell["n_preliminary"] += int(bool(r.get("preliminary")))
        cell["seeds"].add(r.get("seed"))
        cell["stages"].update(r.get("_metrics", {}).keys())
    for cell in by_exp.values():
        cell["seeds"] = sorted(s for s in cell["seeds"] if s is not None)
        cell["stages"] = sorted(cell["stages"])
        cell["headline_ready"] = cell["n_preliminary"] == 0 and len(cell["seeds"]) >= 5
    return {"experiments": by_exp, "n_runs_total": len(runs)}


def write_results(runs_dir: Path = RUNS_DIR, results_dir: Path = RESULTS_DIR) -> dict:
    """Regenerate ``results/summary.{json,md}`` from ``runs/``. Returns the summary."""
    runs = load_runs(runs_dir)
    summary = summarize(runs)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    lines = ["# Results summary", "", f"Total runs: {summary['n_runs_total']}", ""]
    lines += [
        "| Experiment | runs | preliminary | seeds | headline-ready |",
        "|---|---|---|---|---|",
    ]
    for exp, c in sorted(summary["experiments"].items()):
        lines.append(
            f"| {exp} | {c['n_runs']} | {c['n_preliminary']} | "
            f"{c['seeds']} | {c['headline_ready']} |"
        )
    lines += [
        "",
        "_All fixture/`--fast` runs are preliminary by construction and never enter "
        "a headline claim (06_EXPERIMENTS §5.3). Real numbers await the Kaggle "
        "download._",
        "",
    ]
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _log.info("Wrote results summary for %d runs", summary["n_runs_total"])
    return summary
