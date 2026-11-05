"""OWASP Core Rule Set on access logs (owner 2026-10-23): rule parsing (what is usable on an
access log, what is skipped), transformations, anomaly scoring, the parser hook and the
monthly update with its gates. Uses a tiny CRS-style rule set written here (the real
CRS files stay in the antivirus-excluded folder)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.soc import crs, updater

CONF = r'''
SecRule REQUEST_FILENAME|ARGS|ARGS_NAMES "@rx (?i)\bunion\b.{1,100}?\bselect\b" \
    "id:942999,phase:2,block,t:none,t:urlDecodeUni,t:lowercase,msg:'SQL union select',\
    tag:'attack-sqli',tag:'paranoia-level/1',severity:'CRITICAL'"
SecRule ARGS "@pmFromFile words.data" \
    "id:932999,phase:2,block,t:none,t:lowercase,msg:'Unix command',tag:'attack-rce',tag:'paranoia-level/1',severity:'CRITICAL'"
SecRule REQUEST_HEADERS:User-Agent "@rx sqlmap" \
    "id:913999,phase:1,block,t:none,t:lowercase,msg:'Scanner',tag:'attack-reputation-scanner',tag:'paranoia-level/1',severity:'WARNING'"
SecRule REQUEST_COOKIES "@rx x" "id:942998,phase:2,block,msg:'cookie only',tag:'paranoia-level/1',severity:'CRITICAL'"
SecRule ARGS "@rx y" "id:942997,phase:2,block,msg:'paranoid',tag:'paranoia-level/2',severity:'CRITICAL'"
SecRule ARGS "@detectSQLi" "id:942996,phase:2,block,msg:'libinjection',tag:'paranoia-level/1',severity:'CRITICAL'"
SecRule REQUEST_PROTOCOL "!@within %{tx.allowed_http_versions}" "id:920997,phase:1,block,msg:'protocol',tag:'paranoia-level/1',severity:'CRITICAL'"
SecRule ARGS "@rx a" "id:942995,phase:2,chain,msg:'chained',tag:'paranoia-level/1',severity:'CRITICAL'"
    SecRule ARGS "@rx b" "t:none"
'''


@pytest.fixture()
def rules(tmp_path: Path) -> Path:
    d = tmp_path / "rules"
    d.mkdir()
    (d / "REQUEST-942-APPLICATION-ATTACK-SQLI.conf").write_text(CONF, "utf-8")
    (d / "words.data").write_text("# unix commands\n/etc/shadow\nnc -e\n", "utf-8")
    return d


def test_usable_rules_and_skips(rules: Path) -> None:
    usable, skipped = crs.parse_rules(rules)
    assert sorted(r.id for r in usable) == ["913999", "932999", "942999"]
    assert skipped["paranoia 2"] == 1 and skipped["operator @detectSQLi"] == 1
    assert skipped["needs headers / cookies / body / TX"] == 1  # cookies only
    assert skipped["needs CRS settings (macro)"] == 1 and skipped["chained"] == 2


def test_scoring_and_transformations(rules: Path) -> None:
    sqli = crs.check(crs.request("GET", "/item?id=1%20UNION%20ALL%20SELECT%20pass", "HTTP/1.1", "Mozilla/5.0"), rules)
    assert sqli["attack"] and sqli["score"] == 5 and sqli["classes"] == ["attack-sqli"]
    assert sqli["techniques"] == ["T1190"]
    scan = crs.check(crs.request("GET", "/", "HTTP/1.1", "sqlmap/1.7"), rules)
    assert scan["score"] == 3 and not scan["attack"]  # a warning alone stays under 5
    both = crs.check(crs.request("GET", "/x?f=/etc/shadow", "HTTP/1.1", "sqlmap/1.7"), rules)
    assert both["attack"] and both["score"] == 8
    assert not crs.check(crs.request("GET", "/products?page=2&sort=price", "HTTP/1.1", "Mozilla/5.0"), rules)["attack"]


def test_access_log_lines_get_crs_alerts(rules: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.soc import logformats

    monkeypatch.setattr(crs, "RULES_DIR", rules)
    monkeypatch.setattr(crs, "enabled", lambda: True)
    crs._rules.cache_clear()
    ev = logformats.parse_access('203.0.113.9 - - [17/Oct/2026:01:00:00 +0000] "GET /x?f=/etc/shadow HTTP/1.1" '
                                 '200 10 "-" "curl/8"')  # fmt: skip
    assert ev and ev["kind"] == "web_attack" and "T1190" in ev["techniques"] and ev["crs"]["rules"] == ["932999"]
    ok = logformats.parse_access('203.0.113.9 - - [17/Oct/2026:01:00:00 +0000] "GET /about HTTP/1.1" 200 10 "-" "curl/8"')
    assert ok and "crs" not in ok


def test_monthly_crs_update_gated_and_reversible(rules: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    (root / "artifacts" / "intel").mkdir(parents=True)
    (root / "artifacts" / "intel" / "benign_sample.jsonl").write_text(
        json.dumps({"line": '198.51.100.1 - - [17/Oct/2026:01:00:00 +0000] "GET /index.html HTTP/1.1" 200 5 "-" "Mozilla/5.0"'})
        + "\n", "utf-8")  # fmt: skip
    lay = updater.Layout(root=root, out=root)
    files = {"rules/REQUEST-942-APPLICATION-ATTACK-SQLI.conf": CONF.encode(), "rules/words.data": b"/etc/shadow\n"}
    tree = {"tree": [{"type": "blob", "path": p} for p in files] + [{"type": "blob", "path": "rules/RESPONSE-950-X.conf"}]}

    def fetch(url: str) -> bytes:
        if "git/trees" in url:
            return json.dumps(tree).encode()
        return files[url.split("/v9.9.9/", 1)[1]]

    monkeypatch.setattr(updater, "FETCH", fetch)
    monkeypatch.setattr(updater, "CRS_TAG", lambda: "v9.9.9")
    monkeypatch.setattr(updater, "CRS_MIN_RULES", 2)
    rep = updater.run_once(force=True, lay=lay, sources=("owasp_crs",), run_tests_gate=False)
    assert rep["sources"]["owasp_crs"]["status"] == "applied", rep["sources"]["owasp_crs"]
    cd = lay.crs_dir()
    assert (cd / "VERSION").read_text("utf-8") == "v9.9.9" and (cd / "rules" / "words.data").is_file()
    assert not list(cd.glob(".staging-*"))
    monkeypatch.setattr(updater, "FETCH", lambda url: fetch(url.replace("/v9.9.10/", "/v9.9.9/")))
    monkeypatch.setattr(updater, "CRS_TAG", lambda: "v9.9.10")  # next month's release
    rep = updater.run_once(force=True, lay=lay, sources=("owasp_crs",), run_tests_gate=False)
    assert rep["sources"]["owasp_crs"]["status"] == "applied"
    assert (cd / "VERSION").read_text("utf-8") == "v9.9.10"
    assert updater.rollback("owasp_crs", lay)["ok"]
    assert (cd / "VERSION").read_text("utf-8") == "v9.9.9"
    monkeypatch.setattr(updater, "CRS_MIN_RULES", 50)  # a release with too few usable rules is rejected
    rep = updater.run_once(force=True, lay=lay, sources=("owasp_crs",), run_tests_gate=False)
    assert rep["sources"]["owasp_crs"]["status"] == "rejected"
    assert (cd / "VERSION").read_text("utf-8") == "v9.9.9"  # the live rules are untouched
