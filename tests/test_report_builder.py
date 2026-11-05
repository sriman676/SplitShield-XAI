"""Report builder (owner 2026-10-16): periods incl. date / time ranges, a global
template that can be edited, per-report overrides, edits after making, Word /
Markdown / HTML, schedules."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.soc import audit, reports
from tests.test_search_reports import EVENTS, NOW, YDAY

ALL = list(reports.SECTIONS)


def test_default_template_keeps_the_original_report() -> None:
    t = reports.load_template()
    assert t["sections"] == reports.DEFAULT_TEMPLATE["sections"]
    md = reports.build("yesterday", events=EVENTS, now=NOW)["markdown"]
    for h in ("## Incidents", "## Findings by model family", "## Top entities",
              "## ATT&CK coverage", "## What to do", "## System health"):  # fmt: skip
        assert h in md


def test_every_section_builds_and_order_is_kept() -> None:
    rep = reports.build(
        "week", events=EVENTS, now=NOW, template={"sections": ALL[::-1]}
    )
    heads = [ln for ln in rep["markdown"].splitlines() if ln.startswith("## ")]
    assert heads[0] == "## Compliance evidence (NIST CSF 2.0, ISO/IEC 27001:2022)" and "## Alerts over time" in heads
    assert heads.index("## Incidents") > heads.index("## Data collected")


def test_override_only_this_report_and_global_edit() -> None:
    one = reports.build("week", events=EVENTS, now=NOW,
                        template={"sections": ["timeline"], "title": "Board pack"})  # fmt: skip
    assert (
        one["markdown"].startswith("# Board pack:")
        and "## Incidents" not in one["markdown"]
    )
    assert (
        reports.load_template()["title"] == reports.DEFAULT_TEMPLATE["title"]
    )  # global unchanged
    reports.save_template({"sections": ["summary", "sources"], "intro": "For the CISO.",
                           "formats": ["html"]}, who="boss")  # fmt: skip
    g = reports.load_template()
    assert g["sections"] == ["summary", "sources"] and g["formats"] == ["html"]
    md = reports.build("week", events=EVENTS, now=NOW)["markdown"]
    assert (
        "For the CISO." in md and "## Data collected" in md and "## Incidents" not in md
    )
    # the global template can still be overridden for one report
    md = reports.build(
        "week", events=EVENTS, now=NOW, template={"sections": ["attack"]}
    )["markdown"]
    assert "## ATT&CK coverage" in md and "## Data collected" not in md
    assert any(
        r["action"] == "report_template" and r["who"] == "boss" for r in audit.read()
    )


@pytest.mark.parametrize(
    "bad", [{"sections": ["nope"]}, {"formats": ["pdf"]}, {"formats": []}]
)
def test_template_rejects_bad_values(bad: dict) -> None:
    with pytest.raises(ValueError):
        reports.check_template(bad)


def test_periods_and_ranges() -> None:
    # 2026-10-15 is a Thursday: this week starts Monday 2026-10-12 00:00
    s, e, _ = reports.period_bounds("this_week", NOW)
    assert time.localtime(s)[:6] == (2026, 10, 12, 0, 0, 0) and e == NOW
    s, _, label = reports.period_bounds("this_month", NOW)
    assert time.localtime(s)[:3] == (2026, 10, 1) and label == "October 2026"
    # time range: only yesterday's two failed logons fall inside
    rep = reports.build(events=EVENTS, now=NOW, start=YDAY - 60, end=YDAY + 120)
    assert rep["alerts"] == 2
    with pytest.raises(ValueError):
        reports.build(events=EVENTS, now=NOW, start=NOW, end=NOW - 1)
    a = reports.parse_when("2026-10-01")
    assert reports.range_label(a, a + 86400) == "2026-10-01"
    assert reports.range_label(a, a + 7 * 86400) == "2026-10-01 to 2026-10-08"
    assert "14:30" in reports.range_label(a + 14.5 * 3600, a + 16 * 3600)


def test_plain_english_ranges() -> None:
    p, _ = reports.period_of("make a report from 2026-10-01 to 2026-10-07")
    s, e = reports.range_of(p)
    assert (s, e) == (
        reports.parse_when("2026-10-01"),
        reports.parse_when("2026-10-08"),
    )
    p, _ = reports.period_of("make a report from 2026-10-01 09:00 to 2026-10-01 17:30")
    s, e = reports.range_of(p)
    assert e - s == 8.5 * 3600
    p, _ = reports.period_of("generate a report for 2026-10-03")
    s, e = reports.range_of(p)
    assert e - s == 86400
    assert reports.period_of("report for this month")[0] == "this_month"


def test_formats_edit_and_escaping(tmp_path: Path) -> None:
    ev = [*EVENTS, {"ts": NOW - 30, "source": "web<script>alert(1)</script>", "kind": "x",
                    "detail": "", "techniques": ["T1190"], "src_ip": "192.0.2.7"}]  # fmt: skip
    rep = reports.build("today", events=ev, now=NOW, template={"sections": ALL})
    files = reports.save_markdown(
        rep["markdown"], "t", ["docx", "md", "html"], tmp_path
    )
    page = files["html"].read_text("utf-8")
    assert (
        "<script>alert" not in page and "&lt;script&gt;" in page and "<table>" in page
    )
    from docx import Document

    assert Document(str(files["docx"])).tables
    edited = rep["markdown"] + "\n\n## Analyst note\n\nChecked by **ana**."
    files = reports.save_markdown(edited, "t", ["html", "md"], tmp_path)
    assert "<strong>ana</strong>" in files["html"].read_text("utf-8")
    assert files["md"].read_text("utf-8").endswith("Checked by **ana**.")


def test_schedules_run_once_a_day(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.soc import controller

    monkeypatch.setattr(controller, "notify", lambda *_a, **_k: None)
    monkeypatch.setattr(reports, "build", lambda *a, **k: {
        "label": "x", "headline": "h", "markdown": "# r\n", "formats": (k.get("template") or {}).get("formats") or ["md"],
        "sections": ["summary"]})  # fmt: skip
    assert [s["name"] for s in reports.load_schedules()] == ["Daily", "Weekly", "Weekly compliance"]
    reports.save_schedules([{"name": "Mon brief", "period": "this_week", "days": [0, 3],
                             "time": "08:30", "formats": ["html"], "folder": str(tmp_path)}],
                           who="boss")  # fmt: skip
    thu_0800 = time.mktime((2026, 10, 15, 8, 0, 0, 0, 0, -1))
    assert reports.run_due(thu_0800) == []  # before its time
    made = reports.run_due(thu_0800 + 3600)
    assert len(made) == 1 and made[0]["files"]["html"].parent == tmp_path
    assert reports.run_due(thu_0800 + 7200) == []  # once per day
    assert reports.run_due(thu_0800 + 86400 + 3600) == []  # Friday is not a chosen day
    for bad in ({"name": "", "period": "week", "days": [0], "time": "07:00"},
                {"name": "a", "period": "fortnight", "days": [0], "time": "07:00"},
                {"name": "a", "period": "week", "days": [], "time": "07:00"},
                {"name": "a", "period": "week", "days": [0], "time": "25:00"}):  # fmt: skip
        with pytest.raises(ValueError):
            reports.check_schedule(bad)
