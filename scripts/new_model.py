"""Add a new specialist model (owner 2026-10-15: plug-and-play models).

    python -m scripts.new_model dns_tunnel --title "DNS tunnelling model" --family network

Writes three things, refusing to overwrite anything that exists:
  src/mlareas/<name>.py         template: load / score / speed / improve (fill in TODOs)
  tests/test_mlarea_<name>.py   a test that the entry loads and is wired correctly
  config/models.yaml            a new entry, so the controller, health page, risk board,
                                inbox and model cards pick the model up automatically
Train the model, save it under artifacts/mlareas/<name>/current/, then run the tests.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("network", "endpoint", "web", "auth", "ids", "cloud", "phishing",
            "malware", "insider", "ics", "intel", "assistant")  # fmt: skip

MODULE = '''"""{title} (added with scripts/new_model.py).

Contract used by the controller (src/soc/controller.py, config/models.yaml):
  speed()            -> items per second on fixed probe data (probe.npz below)
  improve(path)      -> champion/challenger update from a labelled inbox file; must
                        return {{"promoted": bool, ...}} and judge on FIXED held-out
                        data (holdout.npz), never on its own training rows
  score(rows)        -> one risk score per input row (0..1), used by the SOC
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.paths import PROJECT_ROOT

AREA = "{name}"
DIR = PROJECT_ROOT / "artifacts" / "mlareas" / AREA / "current"


def load() -> dict[str, Any]:
    """The trained bundle (TODO: train it and save model.joblib here)."""
    import joblib

    bundle: dict[str, Any] = joblib.load(DIR / "model.joblib")
    return bundle


def score(X: np.ndarray) -> np.ndarray:
    """Risk 0..1 per row (TODO: adapt to the model's input)."""
    b = load()
    return np.asarray(b["model"].predict_proba(X)[:, 1])


def speed() -> float:
    X = np.load(DIR / "probe.npz")["X"]
    t0 = time.perf_counter()
    score(X)
    return len(X) / max(time.perf_counter() - t0, 1e-9)


def improve(path: Path) -> dict[str, Any]:
    """TODO: refit on replay + the new labelled file; compare with the champion on
    holdout.npz; save only if not worse. Until implemented, nothing changes."""
    return {{"promoted": False, "reason": "improve() not implemented yet", "file": str(path)}}
'''

TEST = '''"""Wiring of the {name} model (written by scripts/new_model.py)."""

from src.soc import controller


def test_{name}_is_registered() -> None:
    m = controller.registry()["{name}"]
    assert m.title == "{title}"
    assert m.extra["family"] == "{family}"
    assert callable(m.speed) and callable(m.learn)
'''

ENTRY = """
{name}:
  title: {title}
  family: {family}
  files: [artifacts/mlareas/{name}/current]
  speed: src.mlareas.{name}:speed
  unit: {unit}
  inbox: "{inbox}"
  learn: src.mlareas.{name}:improve
  entities: [{entities}]
  inputs: {inputs}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--title", required=True)
    ap.add_argument("--family", choices=FAMILIES, required=True)
    ap.add_argument("--unit", default="items/s")
    ap.add_argument("--inbox", default="*.csv")
    ap.add_argument(
        "--entities", default="ip", help="comma list: ip,user,host,url,domain,file"
    )
    ap.add_argument("--inputs", default="labelled CSV")
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", a.name):
        raise SystemExit("name: lowercase letters, digits and _ only")
    reg = ROOT / "config" / "models.yaml"
    if re.search(rf"^{a.name}:", reg.read_text("utf-8"), re.M):
        raise SystemExit(f"{a.name} is already in config/models.yaml")
    mod = ROOT / "src" / "mlareas" / f"{a.name}.py"
    test = ROOT / "tests" / f"test_mlarea_{a.name}.py"
    for f in (mod, test):
        if f.exists():
            raise SystemExit(f"{f} exists; not overwriting")
    kw = vars(a) | {"entities": ", ".join(x.strip() for x in a.entities.split(","))}
    mod.write_text(MODULE.format(**kw), "utf-8")
    test.write_text(TEST.format(**kw), "utf-8")
    with reg.open("a", encoding="utf-8") as fh:
        fh.write(ENTRY.format(**kw))
    print(
        f"added {a.name}: {mod.relative_to(ROOT)}, {test.relative_to(ROOT)}, config/models.yaml"
    )


if __name__ == "__main__":
    main()
