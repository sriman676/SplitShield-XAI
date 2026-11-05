"""Autopilot: the SOC's own decisions, hourly (owner decision 2026-10-14).

Runs from a scheduled task. Every change is validated and rolled back when worse;
nothing here blocks traffic (auto-blocking stays manual by owner choice).

  1. Retrain from feedback - once analysts have reviewed ``NEW_FEEDBACK`` more
     alerts since the last retrain, fine-tune the deployed detector on them
     (engine.retrain_from_feedback: 30 % held out, kept only if it improves).
  2. Adapt to drift - measure live traffic against the training data (PSI,
     src/soc/drift.py). Drifted + enough fresh reviews -> retrain now; drifted
     without labels -> raise a notice asking analysts to review recent alerts
     (labels are what the model can safely learn from).
  3. Improve the assistant - once a night (01:00-05:00), run the gated ShieldLM
     self-improvement cycle (src/nlp/selfimprove.py; backup + gate as before).
  4. Controller (src/soc/controller.py, owner 2026-10-15): learn from labelled files
     in artifacts/inbox/<model>/ and from assistant ratings, for all 9 models; every
     update (including 1 and 3) runs under its snapshot + speed guard; health watch
     with notifications.
  5. Nightly (02:00-05:00, once a day): speed profiling of every model and the rule
     engine, and code speed-up PROPOSALS for the owner (src/soc/codeopt.py).

Skips everything while the training run is active (the run owns the CPU).
Writes artifacts/soc/autopilot_state.json and appends artifacts/soc/autopilot_log.jsonl.

    python -m scripts.autopilot [--force-assistant]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOC = ROOT / "artifacts" / "soc"
STATE = SOC / "autopilot_state.json"
LOG = SOC / "autopilot_log.jsonl"
NEW_FEEDBACK = 20
DRIFT_FEEDBACK = 10
SAMPLE_KEEP = 5000
NIGHT = range(1, 6)
ASSISTANT_EVERY_S = 20 * 3600
PROFILE_HOURS = range(2, 6)



def _load() -> dict[str, Any]:
    try:
        return json.loads(STATE.read_text("utf-8"))
    except (OSError, ValueError):
        return {"feedback_used": 0, "assistant_last": 0.0}


def _save(st: dict[str, Any]) -> None:
    SOC.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), "utf-8")
    tmp.replace(STATE)


def _log(entry: dict[str, Any]) -> None:
    SOC.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), **entry}, default=str) + "\n")
    print(f"[autopilot] {entry}")


def training_active() -> bool:
    from scripts.supervisor import LOCK, _pid_alive

    try:
        return _pid_alive(int(LOCK.read_text().strip()))
    except (OSError, ValueError):
        return False


def drift_now(det: Any) -> dict[str, Any]:
    """Drift report on the rolling flow sample (trimmed to the newest rows)."""
    import numpy as np

    from src.soc import engine
    from src.soc.drift import report

    rows = engine.read_jsonl(engine.FLOW_SAMPLE)
    if len(rows) > SAMPLE_KEEP:  # rolling window, by design of the sample buffer
        rows = rows[-SAMPLE_KEEP:]
        tmp = engine.FLOW_SAMPLE.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
        tmp.replace(engine.FLOW_SAMPLE)
    stats = getattr(det, "input_stats", None) or {}
    if not stats.get("deciles"):
        return {"status": "no_reference", "note": "redeploy to store training deciles"}
    width = len(det.feature_names)
    x = np.array([r["x"] for r in rows if len(r.get("x", [])) == width])
    return (
        report(x, det.feature_names, stats["deciles"])
        if len(x)
        else {
            "status": "not_enough_data",
            "rows": 0,
        }
    )


def run(force_assistant: bool = False, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    st = _load()
    if training_active():
        _log({"action": "skip", "reason": "training run active"})
        return {"skipped": True}
    from src.soc import engine
    from src.soc.rcl import RCL

    out: dict[str, Any] = {}
    det = engine.load_detector()
    if det is not None:
        reviewed = len([r for r in RCL().rows() if r.get("features")])
        fresh = reviewed - int(st.get("feedback_used", 0))
        drift = drift_now(det)
        st["drift"] = drift | {"ts": now}
        out["drift"] = drift.get("status")
        retrain_reason = None
        if fresh >= NEW_FEEDBACK:
            retrain_reason = f"{fresh} new analyst reviews"
        elif drift.get("drifted") and fresh >= DRIFT_FEEDBACK:
            retrain_reason = f"traffic drift + {fresh} new reviews"
        if retrain_reason:
            from src.soc import controller

            res = controller.guarded(
                "flow", lambda: engine.retrain_from_feedback(det), retrain_reason
            )
            st["feedback_used"] = reviewed
            _log({"action": "retrain_detector", "why": retrain_reason, "result": res})
            out["retrain"] = res
        elif drift.get("drifted"):
            st["notice"] = (
                "Live traffic has drifted away from the training data "
                f"({drift['share_strong']:.0%} of features shifted). Review recent "
                f"alerts (thumbs up/down) - after {DRIFT_FEEDBACK} reviews the "
                "detector adapts automatically."
            )
            _log({"action": "drift_notice", "drift": drift})
        else:
            st.pop("notice", None)
    hour = time.localtime(now).tm_hour
    due = now - float(st.get("assistant_last", 0)) >= ASSISTANT_EVERY_S
    if force_assistant or (hour in NIGHT and due):
        from src.nlp import selfimprove

        ready = selfimprove.ready()
        if ready.get("missing"):
            _log({"action": "assistant_skip", "missing": ready["missing"]})
        else:
            from src.soc import controller

            res = controller.guarded("shieldlm", selfimprove.improve, "nightly cycle")
            st["assistant_last"] = now
            _log({"action": "improve_assistant", "result": res})
            out["assistant"] = res
    # self-evolving rules / speed (src/soc/evolve.py): one gated iteration per hour
    try:
        from src.soc import evolution, evolve

        # the user may have frozen the rule engine (Settings -> Evolution)
        step = evolve.iterate() if evolution.allowed("rules") else {"status": "frozen"}
        if step.get("kept"):
            _log(
                {
                    "action": "evolve_kept",
                    "change": step["kept"],
                    "champion": step.get("champion"),
                }
            )
        out["evolve"] = step.get("kept") or step.get("status") or "no better challenger"
    except Exception as exc:  # noqa: BLE001 - the loop must never break the autopilot
        _log({"action": "evolve_error", "error": str(exc)[:200]})
    # all 9 models: inbox files + assistant ratings, under the snapshot/speed guard
    try:
        from src.soc import controller

        steps = controller.cycle()
        out["controller"] = [f"{x['model']}: {x['decision']}" for x in steps]
        for x in steps:
            _log(
                {"action": "controller", "model": x["model"], "decision": x["decision"]}
            )
        controller.watch()
    except Exception as exc:  # noqa: BLE001 - never break the autopilot
        _log({"action": "controller_error", "error": str(exc)[:200]})
    # risk-board boost: switched on only once the owner's reviews validate it
    try:
        from src.soc import riskboard

        st["riskboard"] = riskboard.learn_from_reviews().get("status")
    except Exception as exc:  # noqa: BLE001
        _log({"action": "riskboard_error", "error": str(exc)[:200]})
    # proposals for the users (owner 2026-10-24: proposed, a user accepts on Customise):
    # rule changes from reviews, and the per-site flow calibration
    try:
        from src.soc import rulectl, sitecal

        st["rule_proposals"] = sum(1 for p in rulectl.propose() if p.get("status") == "open")
        st["flow_calibration"] = "proposed" if sitecal.propose() else "nothing to propose"
    except Exception as exc:  # noqa: BLE001
        _log({"action": "proposals_error", "error": str(exc)[:200]})
    # local retention (Settings -> Places & backup): once a day, between 01:00 and 05:00
    if 1 <= hour <= 5 and time.strftime("%Y-%m-%d", time.localtime(now)) != st.get("retention_day"):
        try:
            from src.soc import retention

            res = retention.apply(now)
            st["retention_day"] = time.strftime("%Y-%m-%d", time.localtime(now))
            _log({"action": "retention", "result": {k: v for k, v in res.items() if k != "archives_removed"}})
        except Exception as exc:  # noqa: BLE001
            _log({"action": "retention_error", "error": str(exc)[:200]})
    # over- / underfitting check (owner 2026-10-17): the learning-curve report is
    # rebuilt weekly at night when 11 GB are free; a changed verdict is notified once
    try:
        from src.eval import fitcheck
        from utils.sysinfo import free_ram_gb

        fit = fitcheck.model_report()
        stale = fit["age_days"] is None or fit["age_days"] >= 7
        # never from tests or sandboxes (2026-10-17: a test faking 03:00 started the real
        # 16-minute report inside the research-run gate)
        real = not os.environ.get("SPLITSHIELD_IN_TRIAL") and not os.environ.get("PYTEST_CURRENT_TEST")
        if real and stale and 2 <= hour <= 5 and free_ram_gb() >= 11:
            import subprocess
            import sys

            subprocess.run([sys.executable, "-m", "scripts.fit_report"], timeout=4 * 3600,
                           capture_output=True)  # fmt: skip
            fit = fitcheck.model_report()
            _log({"action": "fit_report", "flagged": fit["flagged"]})
        seen = st.get("fit_verdicts", {})
        now_v = {m: v["verdict"] for m, v in fit["models"].items()}
        changed = [m for m in fit["flagged"] if seen.get(m) != now_v[m]]
        if changed:
            from src.soc.controller import notify

            notify("fit check: " + "; ".join(f"{m} - {now_v[m]}" for m in changed))
        st["fit_verdicts"] = now_v
    except Exception as exc:  # noqa: BLE001
        _log({"action": "fit_error", "error": str(exc)[:200]})
    # scheduled SOC reports (Reports page; default daily 07:00 yesterday + Monday's
    # last 7 days): each schedule runs once a day on the first cycle after its time
    try:
        from src.soc import reports

        made = reports.run_due(now)
        if made:
            _log({"action": "scheduled_reports",
                  "files": [str(next(iter(r["files"].values()))) for r in made]})  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        _log({"action": "report_error", "error": str(exc)[:200]})
    # where the SOC runs: weekly re-guess (confirm mode asks; auto mode switches)
    try:
        from src.soc import places

        places.check(engine.recent_events())
    except Exception as exc:  # noqa: BLE001
        _log({"action": "place_error", "error": str(exc)[:200]})
    # nightly speed profile + code proposals (owner approves on the dashboard)
    if (
        hour in PROFILE_HOURS
        and now - float(st.get("profile_last", 0)) >= ASSISTANT_EVERY_S
    ):
        try:
            from src.soc import codeopt

            prof = codeopt.profile()
            props = codeopt.propose()
            st["profile_last"] = now
            _log({"action": "profile", "throughput": prof.get("throughput"),
                  "slower": prof.get("slower"),
                  "proposals_ready": sum(p["status"] == "ready for approval" for p in props)})  # fmt: skip
        except Exception as exc:  # noqa: BLE001
            _log({"action": "profile_error", "error": str(exc)[:200]})
    st["last_run"] = now
    _save(st)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-assistant", action="store_true")
    a = ap.parse_args()
    from utils.logging_config import configure_logging

    configure_logging()
    run(force_assistant=a.force_assistant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
