"""Live Windows-log watching (src/soc/winlive.py): wevtutil XML -> SOC events, record
bookmarks (nothing read twice, nothing missed), clear permission messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.soc import engine, winlive

NS = "http://schemas.microsoft.com/win/2004/08/events/event"


def _xml(records: list[tuple[int, int, str]]) -> str:
    evs = "".join(
        f"<Event xmlns='{NS}'><System><Provider Name='Microsoft-Windows-Security-Auditing'/>"
        f"<EventID>{eid}</EventID><TimeCreated SystemTime='2026-10-16T08:00:0{i % 10}.0Z'/>"
        f"<EventRecordID>{rid}</EventRecordID><Channel>Security</Channel></System>"
        f"<EventData><Data Name='TargetUserName'>bob</Data><Data Name='IpAddress'>203.0.113.9</Data>"
        f"</EventData><RenderingInfo Culture='en-US'><Message>{msg}</Message></RenderingInfo></Event>"
        for i, (rid, eid, msg) in enumerate(records)
    )
    return f"<Events>{evs}</Events>"


FAIL = ("An account failed to log on.\n\nAccount For Which Logon Failed:\n"
        "  Account Name: bob\n\nNetwork Information:\n  Source Network Address: 203.0.113.9")  # fmt: skip


@pytest.fixture()
def fake_logs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Security holds records 1..N; Sysmon is denied; PowerShell is not installed."""
    box: dict = {"security": [], "queries": []}

    def wev(args: list[str]) -> tuple[int, str]:
        log = args[1]
        box["queries"].append(args)
        if log == "Security":
            recs = box["security"]
            if "/rd:true" in args:  # newest record
                return 0, _xml(recs[-1:])
            q = next(a for a in args if a.startswith("/q:"))
            after = int(q.split(">")[1].split(")")[0])
            return 0, _xml([r for r in recs if r[0] > after][: winlive.BATCH])
        if "Sysmon" in log:
            return 5, "Failed to read events. Access is denied."
        return 15, "The specified channel could not be found."

    monkeypatch.setattr(winlive, "_wevtutil", wev)
    return box


def test_new_records_once_each_and_status_per_log(fake_logs: dict) -> None:
    fake_logs["security"] = [(i, 4625, FAIL) for i in range(1, 801)]
    s = winlive.poll_once(send=False)
    # first run: only the newest FIRST_BACKFILL records, not the whole history
    assert s["logs"]["Security"] == f"{winlive.FIRST_BACKFILL} new records"
    assert "Event Log Readers" in s["logs"]["Sysmon"]
    assert s["logs"]["PowerShell"] == "not installed on this PC"
    stored = engine.read_jsonl(engine.EVENTS)
    assert len(stored) == winlive.FIRST_BACKFILL
    assert stored[0]["kind"] == "failed_logon" and "T1110" in stored[0]["techniques"]
    assert s["alerts"] > 0
    assert winlive.poll_once(send=False)["events"] == 0  # nothing read twice
    fake_logs["security"] += [(801, 4625, FAIL), (802, 4625, FAIL)]
    assert winlive.poll_once(send=False)["events"] == 2  # nothing missed
    assert winlive.load_state()["logs"]["Security"]["last"] == 802


def test_bookmark_only_moves_after_storing(
    fake_logs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_logs["security"] = [(i, 4625, FAIL) for i in range(1, 11)]

    def broken(*a: object, **k: object) -> list:
        raise OSError("disk full")

    monkeypatch.setattr(engine, "record", broken)
    with pytest.raises(OSError):
        winlive.poll_once(send=False)
    assert "Security" not in winlive.load_state().get("logs", {})  # retried next time
    monkeypatch.undo()


def test_state_lives_next_to_the_store() -> None:
    assert winlive.state_path().parent == Path(engine.EVENTS).parent
