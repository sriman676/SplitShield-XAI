"""Silent agents raise one alert, coming back one event; agent clock skew is kept
(gap audit 2026-10-22)."""

from __future__ import annotations

import time

from src.soc import collector, engine, heartbeat


def _seen(name: str, ago_s: float) -> None:
    store = collector._load("agents.json")
    store[name]["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - ago_s))
    collector._save("agents.json", store)


def test_silent_agent_alerts_once_and_back_once() -> None:
    collector.new_agent("boss", "web1")
    collector.new_agent("boss", "never")  # never connected: not silent
    _seen("web1", 45 * 60)
    out = heartbeat.check(send=False)
    assert [e["kind"] for e in out] == ["source_silent"] and out[0]["techniques"] == ["T1562"]
    assert heartbeat.check(send=False) == []  # one alert per silence
    alerts = engine.read_jsonl(engine.ALERTS)
    assert any(a.get("kind") == "source_silent" for a in alerts)
    _seen("web1", 10)
    assert [e["kind"] for e in heartbeat.check(send=False)] == ["source_back"]


def test_clock_skew() -> None:
    assert (heartbeat.skew(time.time() - 300) or 0) >= 299
    assert heartbeat.skew("nonsense") is None
    collector.new_agent("boss", "pc9")
    store = collector._load("agents.json")
    store["pc9"]["skew_s"] = -400.0
    collector._save("agents.json", store)
    assert heartbeat.clock_problems() == {"pc9": -400.0}
