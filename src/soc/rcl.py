"""Reinforcement Calibrated Learning (RCL): the detector learns from analyst feedback.

Every alert the analyst reviews returns a reward: +1 (real attack) or -1 (false
positive). Three things learn from it:

1. **Per-class alert thresholds** (the "reinforcement" part). An alert fires only when
   the calibrated confidence for class c is >= threshold[c]. After feedback,
   stochastic approximation on the reward:
       false positive:  t_c <- t_c + eta_n * (conf - t_c + margin)   (stop alerting there)
       true positive :  t_c <- t_c - eta_n * kappa                   (catch more like it)
   with eta_n = eta0 / sqrt(n_c), clipped to [floor, ceil]. Classes that keep producing
   false alarms become stricter; reliable classes become more sensitive.
2. **Temperature calibration** (the "calibrated" part). Confidence = max softmax(logits/T).
   T is refit on all reviewed alerts by minimising the log-loss of "confidence vs was it
   actually correct", so a 0.9 confidence really means ~90% of such alerts are real.
3. **Replay fine-tuning**: reviewed alerts with their true label are stored as training
   rows; "Retrain from feedback" folds them into the deployed model (continual update).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.soc.sinks import SOC_DIR

STATE = SOC_DIR / "rcl_state.json"
FEEDBACK = SOC_DIR / "feedback.jsonl"

BASE_THRESHOLD = 0.5
FLOOR, CEIL = 0.30, 0.995
ETA0, MARGIN, KAPPA = 0.25, 0.02, 0.05


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-3)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class RCL:
    def __init__(
        self, state_path: Path | None = None, feedback_path: Path | None = None
    ) -> None:
        # resolved at call time (not import time), so tests can redirect the files
        state_path = state_path or STATE
        feedback_path = feedback_path or FEEDBACK
        self.state_path, self.feedback_path = state_path, feedback_path
        s = json.loads(state_path.read_text("utf-8")) if state_path.is_file() else {}
        self.temperature: float = s.get("temperature", 1.0)
        self.thresholds: dict[str, float] = s.get("thresholds", {})
        self.counts: dict[str, int] = s.get("counts", {})
        self.history: list[dict[str, Any]] = s.get("history", [])

    # --- decisions -----------------------------------------------------------
    def threshold(self, cls: str) -> float:
        return self.thresholds.get(cls, BASE_THRESHOLD)

    def calibrated(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, self.temperature)

    def should_alert(self, cls: str, conf: float) -> bool:
        return cls != "BENIGN" and conf >= self.threshold(cls)

    # --- learning ------------------------------------------------------------
    def feedback(
        self,
        alert: dict[str, Any],
        correct: bool,
        true_label: str | None = None,
    ) -> dict[str, Any]:
        """Apply one analyst verdict. Returns the threshold change for display."""
        cls, conf = alert["label"], float(alert["confidence"])
        n = self.counts.get(cls, 0) + 1
        self.counts[cls] = n
        eta = ETA0 / math.sqrt(n)
        before = self.threshold(cls)
        if correct:
            after = before - eta * KAPPA
        else:
            after = before + eta * max(conf - before + MARGIN, MARGIN)
        after = float(min(CEIL, max(FLOOR, after)))
        self.thresholds[cls] = after

        row = {
            "ts": time.time(),
            "alert_id": alert.get("id"),
            "pred": cls,
            "confidence": conf,
            "correct": bool(correct),
            "true_label": true_label or (cls if correct else "BENIGN"),
            "logits": alert.get("logits"),
            "features": alert.get("features"),
        }
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.history.append(
            {
                "ts": row["ts"],
                "class": cls,
                "reward": 1 if correct else -1,
                "threshold": after,
                "temperature": self.temperature,
            }
        )
        self.fit_temperature()
        self.save()
        return {"class": cls, "threshold_before": before, "threshold_after": after}

    def rows(self) -> list[dict[str, Any]]:
        if not self.feedback_path.is_file():
            return []
        return [
            json.loads(ln)
            for ln in self.feedback_path.read_text("utf-8").splitlines()
            if ln.strip()
        ]

    def fit_temperature(self, min_rows: int = 10) -> float:
        """Grid-search T minimising log-loss of max-softmax vs. analyst correctness."""
        rows = [r for r in self.rows() if r.get("logits")]
        if len(rows) < min_rows:
            return self.temperature
        logits = np.array([r["logits"] for r in rows], dtype=np.float64)
        y = np.array([1.0 if r["correct"] else 0.0 for r in rows])
        best_t, best = self.temperature, float("inf")
        for t in np.geomspace(0.25, 8.0, 60):
            p = np.clip(softmax(logits, t).max(axis=1), 1e-6, 1 - 1e-6)
            nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if nll < best:
                best_t, best = float(t), nll
        self.temperature = best_t
        return best_t

    def precision_curve(self, window: int = 20) -> list[dict[str, float]]:
        """Rolling share of reviewed alerts that were real, for the learning chart."""
        out, rewards = [], []
        for i, h in enumerate(self.history):
            rewards.append(1.0 if h["reward"] > 0 else 0.0)
            out.append(
                {"review": i + 1, "precision": float(np.mean(rewards[-window:]))}
            )
        return out

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "temperature": self.temperature,
                    "thresholds": self.thresholds,
                    "counts": self.counts,
                    "history": self.history[-5000:],
                },
                indent=2,
            ),
            "utf-8",
        )
