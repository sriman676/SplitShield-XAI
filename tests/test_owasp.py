"""OWASP Top 10 (owner 2026-10-23): newest edition found and parsed, validated, used to
label web alerts through CWEs, refreshed by the update agent (monthly) with rollback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.soc import owasp, updater

NAMES = ["Broken Access Control", "Security Misconfiguration", "Supply Chain", "Cryptographic Failures",
         "Injection", "Insecure Design", "Authentication Failures", "Integrity Failures",
         "Logging Failures", "Exceptional Conditions"]  # fmt: skip
CWES = {1: [22, 918], 5: [79, 89, 78], 7: [307], 8: [502]}


def fake_site(year: int = 2025) -> dict[str, str]:
    nav = "\n".join(f"    - A{i:02d} {n}: 'en/A{i:02d}_{year}-{n.replace(' ', '_')}.md'" for i, n in enumerate(NAMES, 1))
    site = {f"{owasp.RAW}/{year}/mkdocs.yml": f"nav:\n{nav}\n"}
    for i, n in enumerate(NAMES, 1):
        cw = "\n".join(f"* [CWE-{c} x](https://cwe.mitre.org/data/definitions/{c}.html)" for c in CWES.get(i, [20]))
        site[f"{owasp.RAW}/{year}/docs/en/A{i:02d}_{year}-{n.replace(' ', '_')}.md"] = (
            f"# A{i:02d}:{year} {n}\n\nCWE-999 in the text is not mapped.\n\n## List of Mapped CWEs\n\n{cw}\n")
    return site


def getter(site: dict[str, str]):  # noqa: ANN201
    def get(url: str) -> str:
        if url not in site:
            raise OSError(f"404 {url}")
        return site[url]

    return get


def test_newest_edition_parsed_and_validated() -> None:
    kb = owasp.build(getter(fake_site(2025)), now_year=2027)  # 2027 / 2026 do not exist yet
    assert kb["edition"] == 2025 and len(kb["categories"]) == 10
    inj = kb["categories"][4]
    assert inj["id"] == "A05:2025" and inj["name"] == "Injection" and inj["cwes"] == [78, 79, 89]
    assert owasp.validate(kb) == []
    assert owasp.validate(kb | {"categories": kb["categories"][:9]})


def test_alerts_are_labelled_through_cwes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "owasp.json"
    owasp.save(owasp.build(getter(fake_site(2025)), now_year=2025), path)
    monkeypatch.setattr(owasp, "KB", path)
    owasp.load.cache_clear()
    monkeypatch.setattr(owasp, "load", lambda path=path: json.loads(path.read_text("utf-8")))
    ev = {"kind": "http_request", "techniques": ["T1190"], "detail": "GET /?id=1 UNION SELECT", "rule": "SQL Injection"}
    assert owasp.label(ev)["owasp"] == "A05:2025 Injection"
    trav = {"kind": "http_request", "techniques": ["T1190"], "detail": "GET /../../etc/passwd", "rule": "Path Traversal"}
    assert owasp.label(trav)["owasp"].startswith("A01:2025")
    assert "owasp" not in owasp.label({"kind": "failed_logon", "techniques": ["T1110"], "detail": "x"})


def test_update_agent_fetches_owasp_monthly_with_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lay = updater.Layout(root=tmp_path, out=tmp_path)
    site = fake_site(2025)
    monkeypatch.setattr(updater, "FETCH", lambda url: getter(site)(url).encode())
    monkeypatch.setattr(owasp.time, "gmtime", lambda: type("T", (), {"tm_year": 2025})())
    (tmp_path / "artifacts" / "intel").mkdir(parents=True)
    sample = tmp_path / "artifacts" / "intel" / "benign_sample.jsonl"
    sample.write_text(json.dumps({"line": "GET / 200"}) + "\n", "utf-8")
    rep = updater.run_once(force=True, lay=lay, sources=("owasp_top10",), run_tests_gate=False)
    assert rep["sources"]["owasp_top10"]["status"] == "applied"
    assert json.loads(lay.owasp_json().read_text("utf-8"))["edition"] == 2025
    st = updater.load_state(lay.state)
    assert "owasp_top10" in st["checked"] and not updater.is_due(st, "owasp_top10")
    site.update(fake_site(2026))  # a newer edition next month
    monkeypatch.setattr(owasp.time, "gmtime", lambda: type("T", (), {"tm_year": 2026})())
    rep = updater.run_once(force=True, lay=lay, sources=("owasp_top10",), run_tests_gate=False)
    assert json.loads(lay.owasp_json().read_text("utf-8"))["edition"] == 2026
    assert updater.rollback("owasp_top10", lay)["ok"]
    assert json.loads(lay.owasp_json().read_text("utf-8"))["edition"] == 2025
