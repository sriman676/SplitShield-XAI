"""Per-site calibration of the flow detector (owner 2026-10-24).

On a network the detector was not trained on, its attack scores shift and the
validated threshold can alert on far too many ordinary flows (blind round 2, HIKARI:
14 % false alarms). The fix, like the ICS per-plant fit: watch the site's own traffic
(labels are not needed - almost all traffic is normal) and choose the threshold at
which at most ``TARGET`` of the site's flows would alert. It is never looser than the
validated threshold, and it is only PROPOSED: a user accepts it on Customise ->
Proposals & history (it then becomes the flow sensitivity, undoable there).

The live detector keeps a bounded random sample of flow attack scores
(artifacts/soc/flow_scores.json, reservoir of ``KEEP`` scores with first / last time).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.paths import PROJECT_ROOT

SCORES = PROJECT_ROOT / "artifacts" / "soc" / "flow_scores.json"
TARGET = 0.01  # at most 1 % of the site's flows alert
KEEP = 200_000  # reservoir size
MIN_FLOWS = 50_000  # flows seen before a proposal
MIN_DAYS = 3.0  # days of traffic before a proposal


def threshold_for(scores: np.ndarray, validated: float, target: float = TARGET) -> float:
    """The lowest threshold at which at most ``target`` of these flows alert, never
    below ``validated``."""
    if len(scores) == 0:
        return validated
    q = float(np.quantile(np.asarray(scores, dtype=np.float64), 1.0 - target))
    return float(min(0.9999, max(validated, np.nextafter(q, 1.0))))


_MEM: dict[str, Any] = {}  # path -> {"doc": ..., "saved": time}
FLUSH_S = 60.0


def observe(scores: Any, path: Path | None = None, rng: random.Random | None = None,  # noqa: ANN401
            flush: bool = False) -> None:  # fmt: skip
    """Add live flow attack scores to the reservoir (kept in memory, written at most
    every FLUSH_S seconds; never raises)."""
    p = path or SCORES
    m = _MEM.get(str(p))
    if m is None:
        try:
            doc = json.loads(p.read_text("utf-8")) if p.is_file() else {}
        except (OSError, ValueError):
            doc = {}
        m = _MEM[str(p)] = {"doc": doc, "saved": time.time()}
    doc = m["doc"]
    keep: list[float] = doc.setdefault("scores", [])
    seen = int(doc.get("seen", 0))
    r = rng or random.Random()
    for v in np.asarray(scores, dtype=np.float64).ravel().tolist():
        seen += 1
        if len(keep) < KEEP:
            keep.append(round(v, 6))
        else:
            j = r.randrange(seen)
            if j < KEEP:
                keep[j] = round(v, 6)
    now = time.time()
    doc |= {"seen": seen, "first": doc.get("first", now), "last": now}
    if flush or now - m["saved"] >= FLUSH_S:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(doc), "utf-8")
            m["saved"] = now
        except OSError:
            pass


def status(path: Path | None = None) -> dict[str, Any]:
    p = path or SCORES
    try:
        doc = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return {"seen": 0, "days": 0.0}
    days = (float(doc.get("last", 0)) - float(doc.get("first", 0))) / 86_400
    return {"seen": int(doc.get("seen", 0)), "days": round(days, 2), "kept": len(doc.get("scores", []))}


def propose(path: Path | None = None) -> dict[str, Any] | None:
    """A stricter flow threshold for this site, as a proposal (None when not yet enough
    traffic or the validated threshold already keeps alerts under the target)."""
    from src.soc import custom, rulectl

    st = status(path)
    if st["seen"] < MIN_FLOWS or st["days"] < MIN_DAYS:
        return None
    base = custom.validated("flow")
    if base is None:
        return None
    scores = np.asarray(json.loads((path or SCORES).read_text("utf-8"))["scores"])
    thr = threshold_for(scores, base)
    cur = custom.threshold("flow", base)
    share_now = float((scores >= cur).mean())
    if thr <= cur + 1e-6 or share_now <= TARGET:
        return None
    return rulectl.add_proposal("sens:flow", {"value": round(thr, 4)},
                                f"on {st['seen']:,} flows of this site over {st['days']:.1f} days, "
                                f"{share_now:.1%} would alert at the current threshold {cur:.3f}; "
                                f"{thr:.3f} keeps it at {TARGET:.0%}")  # fmt: skip
