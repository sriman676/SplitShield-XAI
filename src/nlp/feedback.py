"""Learn from every prompt (owner 2026-10-15): 👍 / 👎 feedback on assistant replies.

Each reply on the dashboard can be rated. A rating becomes training signal for the
question-understanding (intent) model:

  * 👍            -> (question, the intent that was used) is a confirmed example;
  * 👎 + correct  -> (question, the intent the analyst picked) is a corrected example;
  * 👎 only       -> logged for review (there is no label to learn from).

``quick_update`` refits the intent model on the original template data plus every
confirmed / corrected example (each repeated with light augmentation so a single
correction counts). The new model replaces the current one ONLY if

  1. accuracy on the fixed held-out templates (never trained on) does not drop, and
  2. it answers at least as many of the rated questions correctly as the current one.

The replaced model is kept as ``intent_prev.joblib`` (``rollback``). Fast enough to
run right after each rating (TF-IDF + logistic regression, CPU, seconds).
Facts in answers come from the knowledge base and the logs, not from this model, so
feedback can change how a question is understood, never what is claimed.
"""

from __future__ import annotations

import json
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT

NLP_DIR = PROJECT_ROOT / "artifacts" / "nlp"
FEEDBACK = NLP_DIR / "question_feedback.jsonl"  # narrative ratings: selfimprove.py
MODEL = NLP_DIR / "intent.joblib"
PREV = NLP_DIR / "intent_prev.joblib"
STATE = NLP_DIR / "feedback_state.json"
REPEAT = 8  # copies (augmented) of each rated question in the training data
_LOCK = threading.Lock()


def intents() -> list[str]:
    from src.nlp.intent import TEMPLATES

    return sorted(TEMPLATES)


def record(
    question: str,
    used_intent: str,
    good: bool,
    correct_intent: str | None = None,
    path: Path = FEEDBACK,
) -> dict[str, Any]:
    """Store one rating; returns the stored row.

    Answers can come from intents the classifier does not decide (``status`` is a
    rule, ``explain_technique`` is ``explain`` for one ATT&CK technique). A 👍 on
    those is stored for review without a label; a correction must name a classifier
    intent."""
    used = {"explain_technique": "explain"}.get(used_intent, used_intent)
    if good:
        label = used if used in intents() else None
    else:
        label = correct_intent
        if label is not None and label not in intents():
            raise ValueError(f"unknown intent {label!r}")
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question.strip()[:500],
        "used_intent": used_intent,
        "good": bool(good),
        "label": label,  # None: thumbs-down without a correction -> review only
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def rows(path: Path = FEEDBACK) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text("utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def labelled(path: Path = FEEDBACK) -> tuple[list[str], list[str]]:
    """Latest label per question (a later rating overrides an earlier one)."""
    last: dict[str, str] = {}
    for r in rows(path):
        q = r["question"].strip().lower()
        if r.get("label"):
            last[q] = r["label"]
    return list(last), list(last.values())


def _accuracy(model: Any, X: list[str], y: list[str]) -> float:
    if not X:
        return 1.0
    pred = model.predict(X)
    return float(sum(p == t for p, t in zip(pred, y, strict=True)) / len(y))


def quick_update(
    path: Path = FEEDBACK, model_path: Path = MODEL, seed: int = 0
) -> dict[str, Any]:
    """Refit with all feedback; keep only if both gates pass (see module doc)."""
    import joblib

    from src.nlp import intent

    fq, fy = labelled(path)
    if not fq:
        return {"status": "nothing to learn", "rated": 0}
    with _LOCK:
        Xtr, ytr, Xte, yte = intent.dataset(seed)
        rng = random.Random(seed + 1)
        Xtr = Xtr + [intent.augment(x, rng) for x in Xtr]
        ytr = ytr + ytr
        for q, y in zip(fq, fy, strict=True):
            Xtr += [q] + [intent.augment(q, rng) for _ in range(REPEAT - 1)]
            ytr += [y] * REPEAT
        t0 = time.time()
        challenger = intent._pipeline().fit(Xtr, ytr)
        champion = joblib.load(model_path) if model_path.is_file() else None
        res: dict[str, Any] = {
            "rated": len(fq),
            "fit_seconds": round(time.time() - t0, 1),
            "heldout_challenger": _accuracy(challenger, Xte, yte),
            "rated_challenger": _accuracy(challenger, fq, fy),
        }
        if champion is not None:
            res["heldout_champion"] = _accuracy(champion, Xte, yte)
            res["rated_champion"] = _accuracy(champion, fq, fy)
        keep = champion is None or (
            res["heldout_challenger"] >= res["heldout_champion"] - 1e-9
            and res["rated_challenger"] >= res["rated_champion"]
        )
        res["status"] = "promoted" if keep else "rejected (would get worse)"
        if keep:
            if model_path.is_file():
                shutil.copy2(model_path, model_path.with_name(PREV.name))
            tmp = model_path.with_suffix(".tmp")
            joblib.dump(challenger, tmp)
            tmp.replace(model_path)
        state = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")} | res
        model_path.with_name(STATE.name).write_text(
            json.dumps(state, indent=1), "utf-8"
        )
    return res


def rollback(model_path: Path = MODEL) -> bool:
    prev = model_path.with_name(PREV.name)
    if not prev.is_file():
        return False
    shutil.copy2(prev, model_path)
    return True


def state(model_path: Path = MODEL) -> dict[str, Any]:
    p = model_path.with_name(STATE.name)
    try:
        out: dict[str, Any] = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return out
