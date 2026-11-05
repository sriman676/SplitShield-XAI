"""Every writable SOC state file, redirected to one folder (2026-10-17).

The single list used by the tests (tests/conftest.py, via monkeypatch) and by the
every-combination stress test (scripts/combo_stress.py, one folder per combination):
nothing they do can touch the real SOC store, settings, users, models' history or
reports. Read-only inputs (models, config, rules) stay real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def targets(root: Path) -> list[tuple[Any, str, Any]]:
    """(module, attribute, new value) for every writable state path."""
    from src.nlp import feedback, selfimprove
    from src.soc import (
        allowlist,
        auth,
        codeopt,
        controller,
        custom,
        engine,
        places,
        rcl,
        reports,
        riskboard,
        rulectl,
        sinks,
        sitecal,
        unknown,
    )

    store = root / "soc_store"
    st = root / "soc_state"
    out: list[tuple[Any, str, Any]] = [(engine, "FLOW_SAMPLE", root / "flow_sample.jsonl")]
    out += [(engine, n, store / getattr(engine, n).name) for n in ("EVENTS", "ALERTS", "PUSHED", "STATUS")]
    out += [
        (controller, "HOME", st / "controller"),
        (controller, "HISTORY", st / "controller" / "history.jsonl"),
        (controller, "STATE", st / "controller" / "state.json"),
        (controller, "INBOX", st / "inbox"),
        (codeopt, "HOME", st / "controller"),
        (codeopt, "PROFILE", st / "controller" / "profile.json"),
        (codeopt, "PROPOSALS", st / "controller" / "code_proposals.json"),
        (codeopt, "BACKUPS", st / "controller" / "code_backups"),
        (places, "STATE", st / "place.json"),
        (places, "BUDGET", st / "push_budget.json"),
        (riskboard, "VALIDATED", st / "riskboard_validation.json"),
        (reports, "OUT", st / "reports"),
        (custom, "RULES_DIR", st / "custom_rules"),
        (custom, "SENSITIVITY", st / "sensitivity.json"),
        (allowlist, "PATH", st / "allowlist.json"),
        (feedback, "FEEDBACK", st / "question_feedback.jsonl"),
        (feedback, "STATE", st / "feedback_state.json"),
        (sinks, "SETTINGS", st / "settings.json"),
        (sinks, "OUTBOX", st / "outbox_alerts.jsonl"),
        (rcl, "STATE", st / "rcl_state.json"),
        (rcl, "FEEDBACK", st / "feedback.jsonl"),
        (unknown, "SUGGESTIONS", st / "parser_suggestions.jsonl"),
        (selfimprove, "FEEDBACK", st / "narrative_feedback.jsonl"),
        (engine, "USER_DATA", st / "user_datasets"),
        (auth, "ENV", st / ".env"),
        (rulectl, "OVERRIDES", st / "rule_overrides.json"),
        (rulectl, "SETTINGS", st / "engine_settings.json"),
        (rulectl, "PROPOSALS", st / "rule_proposals.json"),
        (rulectl, "HISTORY", st / "rule_history.jsonl"),
        (sitecal, "SCORES", st / "flow_scores.json"),
    ]
    return out


def isolate(root: Path) -> None:
    """Point this process's SOC at ``root`` (for worker processes; no undo)."""
    import os

    from src.soc import allowlist, eventindex

    eventindex.close()
    for mod, attr, value in targets(root):
        setattr(mod, attr, value)
    allowlist._CACHE["key"] = None
    os.environ["SPLITSHIELD_IN_TRIAL"] = "1"  # never start code-proposal trials
    os.environ["SPLITSHIELD_NO_NOTIFY"] = "1"
    from src.soc import auth

    os.environ.pop(auth.KEY, None)
