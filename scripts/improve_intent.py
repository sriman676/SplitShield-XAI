"""Improve the assistant's question-understanding (intent) model (owner 2026-10-15).

The phrasings (TEMPLATES) are not changed, so the held-out test stays the same
questions: only the model settings are searched. Settings are chosen on GROUPED
cross-validation over the training templates (whole phrasings held out, never the
test templates); the seed-0 held-out templates (``intent.dataset(0)`` test part)
are reported only. The deployed model is replaced only if held-out macro precision
AND macro recall are both not worse; the previous model is kept as
``intent_prev.joblib``.

    python -m scripts.improve_intent [--no-save]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.nlp import intent
from utils.paths import PROJECT_ROOT, RESULTS_DIR

NLP = PROJECT_ROOT / "artifacts" / "nlp"
OUT = RESULTS_DIR / "extra"
CONFIGS = [
    {"word": (1, 2), "char": (2, 5), "C": 4.0, "cw": None},  # deployed
    {"word": (1, 2), "char": (2, 5), "C": 10.0, "cw": None},
    {"word": (1, 3), "char": (2, 5), "C": 10.0, "cw": None},
    {"word": (1, 2), "char": (1, 6), "C": 10.0, "cw": None},
    {"word": (1, 2), "char": (2, 5), "C": 10.0, "cw": "balanced"},
    {"word": (1, 3), "char": (1, 6), "C": 30.0, "cw": "balanced"},
    {"word": (1, 2), "char": (3, 6), "C": 4.0, "cw": "balanced"},
]


def build(cfg: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("norm", FunctionTransformer(intent.normalize_all)),
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=cfg["word"], sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=cfg["char"], sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=3000, C=cfg["C"], class_weight=cfg["cw"])),
    ])  # fmt: skip


def train_templates(seed: int = 0) -> list[tuple[str, str]]:
    """(intent, template) pairs of the TRAINING part of dataset(seed) (as its split)."""
    rng = random.Random(seed)
    out = []
    for name, temps in intent.TEMPLATES.items():
        temps = temps[:]
        rng.shuffle(temps)
        cut = max(1, len(temps) // 4)
        out += [(name, t) for t in temps[cut:]]
    return out


def grouped_cv_train_only(cfg: dict[str, Any], folds: int = 4, seed: int = 0) -> float:
    """Accuracy with whole training templates held out (test templates never used)."""
    items = train_templates(0)
    rng = random.Random(seed + 11)
    rng.shuffle(items)
    accs = []
    for k in range(folds):
        test_t = {it for i, it in enumerate(items) if i % folds == k}
        Xtr: list[str] = []
        ytr: list[str] = []
        Xte: list[str] = []
        yte: list[str] = []
        for name, t in items:
            for _ in range(12):
                (Xte if (name, t) in test_t else Xtr).append(intent._fill(t, rng))
                (yte if (name, t) in test_t else ytr).append(name)
        Xtr += [intent.augment(x, rng) for x in Xtr]
        ytr += ytr
        accs.append(float(accuracy_score(yte, build(cfg).fit(Xtr, ytr).predict(Xte))))
    return statistics.mean(accs)


def heldout(model: Any) -> dict[str, float]:
    _, _, Xte, yte = intent.dataset(0)
    p = model.predict(Xte)
    return {"accuracy": float(accuracy_score(yte, p)),
            "macro_precision": float(precision_score(yte, p, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yte, p, average="macro", zero_division=0))}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    champ = joblib.load(NLP / "intent.joblib")
    c = heldout(champ)
    rows: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        cv = grouped_cv_train_only(cfg)
        rows.append({"config": cfg, "grouped_cv_train_templates": cv})
        print(cfg, "grouped CV", round(cv, 4), flush=True)
    best = max(rows, key=lambda r: r["grouped_cv_train_templates"])
    Xtr, ytr, _, _ = intent.dataset(0)
    rng = random.Random(1)
    Xtr = Xtr + [intent.augment(x, rng) for x in Xtr]
    ytr = ytr + ytr
    model = build(best["config"]).fit(Xtr, ytr)
    h = heldout(model)
    keep = (h["macro_precision"] >= c["macro_precision"] - 1e-12
            and h["macro_recall"] >= c["macro_recall"] - 1e-12)  # fmt: skip
    doc = {"what": "intent model settings chosen on grouped CV over training templates",
           "champion_heldout": c, "challenger_heldout": h, "candidates": rows,
           "chosen": best, "decision": "promoted" if keep else "kept champion",
           "saved": False, "seconds": round(time.time() - t0)}  # fmt: skip
    if keep and best["config"] != CONFIGS[0] and not a.no_save:
        shutil.copy2(NLP / "intent.joblib", NLP / "intent_prev.joblib")
        joblib.dump(model, NLP / "intent.joblib")
        doc["saved"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "improve_intent.json").write_text(
        json.dumps(doc, indent=1, default=str), "utf-8"
    )
    print(json.dumps({k: doc[k] for k in ("champion_heldout", "challenger_heldout", "decision", "saved")},
                     indent=1, default=str))  # fmt: skip


if __name__ == "__main__":
    main()
