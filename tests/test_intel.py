"""Threat-intel feed parsing and lookups (src/soc/intel.py), offline."""

from __future__ import annotations

from pathlib import Path

from src.soc import intel

URLHAUS = """################################################################
# abuse.ch URLhaus Database Dump (CSV - online URLs only)      #
################################################################
#
# id,dateadded,url,url_status,last_online,threat,tags,urlhaus_link,reporter
"1","2026-10-15 06:47:07","http://105.184.125.244:54343/i","online","2026-10-15","malware_download","None","https://urlhaus.abuse.ch/url/1/","x"
"2","2026-10-15 06:40:00","https://evil-dl.example.org/a.exe","online","2026-10-15","malware_download","exe,AsyncRAT","https://urlhaus.abuse.ch/url/2/","x"
"3","2026-10-15 06:30:00","https://raw.githubusercontent.com/u/r/main/p.ps1","online","2026-10-15","malware_download","ps1","https://urlhaus.abuse.ch/url/3/","x"
"""

FEODO = """################################################################
# abuse.ch Feodo Tracker Botnet C2 IP Blocklist (CSV)          #
################################################################
#
"first_seen_utc","dst_ip","dst_port","c2_status","last_online","malware"
"2022-06-04 21:24:53","162.243.103.246","8080","offline","2026-03-07","Emotet"
"2025-12-30 13:56:31","50.16.16.211","443","online","2026-03-12","QakBot"
"2025-12-30 13:56:31","10.0.0.5","443","online","2026-03-12","Bogus"
# end
"""

THREATFOX = """################################################################
# ThreatFox IOCs: recent additions - CSV format                #
################################################################
#
# "first_seen_utc","ioc_id","ioc_value","ioc_type","threat_type","fk_malware","malware_alias","malware_printable","last_seen_utc","confidence_level","is_compromised","reference","tags","anonymous","reporter"
"2026-10-15 06:58:28", "1", "45.9.148.10:4444", "ip:port", "botnet_cc", "win.x", "", "Cobalt Strike", "", "100", "0", "", "", "0", "a"
"2026-10-15 06:58:28", "2", "c2.bad-domain.net", "domain", "botnet_cc", "win.x", "", "Sliver", "", "100", "0", "", "", "0", "a"
"2026-10-15 06:58:28", "3", "https://pay.bad-domain.net/x", "url", "payload_delivery", "win.x", "", "Lumma", "", "100", "0", "", "", "0", "a"
"2026-10-15 06:58:28", "4", "d41d8cd98f00b204e9800998ecf8427e", "md5_hash", "payload", "win.x", "", "Lumma", "", "100", "0", "", "", "0", "a"
"""

OPENPHISH = """http://help.steampowered.com.im/en/wizard/
https://login-micros0ft.example.net/
not a url
"""


def _store(tmp_path: Path) -> Path:
    for name, text in (("urlhaus", URLHAUS), ("feodo", FEODO),
                       ("threatfox", THREATFOX), ("openphish", OPENPHISH)):  # fmt: skip
        intel.write_feed(
            intel.build_feed(name, text.encode(), "2026-10-15T04:00:00Z"), tmp_path
        )
    return tmp_path


def test_parsers_count_and_skip_private() -> None:
    assert len(intel.parse_urlhaus(URLHAUS)) == 3
    feodo = intel.parse_feodo(FEODO)
    assert [i["value"] for i in feodo] == ["162.243.103.246", "50.16.16.211"]
    tf = intel.parse_threatfox(THREATFOX)
    assert {(i["type"], i["value"]) for i in tf} == {
        ("ip", "45.9.148.10"),
        ("domain", "c2.bad-domain.net"),
        ("url", "https://pay.bad-domain.net/x"),
    }
    assert len(intel.parse_openphish(OPENPHISH)) == 2


def test_lookups_with_source(tmp_path: Path) -> None:
    folder = _store(tmp_path)
    m = intel.lookup_ip("162.243.103.246", folder)
    assert m and m[0]["source"] == "feodo" and "Emotet" in m[0]["threat"]
    assert m[0]["fetched"] == "2026-10-15T04:00:00Z"
    assert intel.lookup_ip("105.184.125.244", folder)[0]["source"] == "urlhaus"
    assert intel.lookup_ip("10.0.0.5", folder) == []  # private never matches
    assert intel.lookup_domain("c2.bad-domain.net", folder)[0]["source"] == "threatfox"
    assert intel.lookup_domain("www.c2.bad-domain.net", folder)  # parent match
    assert intel.lookup_domain("bad-domain.net", folder) == []  # not the other way
    assert intel.lookup_url("HTTPS://Evil-DL.example.org/a.exe", folder)
    assert intel.lookup_url("https://evil-dl.example.org/other", folder)  # bad host
    assert intel.lookup_url("https://login-micros0ft.example.net", folder)
    assert intel.lookup_domain("nothing-here.org", folder) == []


def test_shared_hosts_match_only_exact_url(tmp_path: Path) -> None:
    folder = _store(tmp_path)
    assert intel.lookup_domain("raw.githubusercontent.com", folder) == []
    assert intel.lookup_domain("github.com", folder) == []
    assert intel.lookup_url("https://raw.githubusercontent.com/u/r/main/p.ps1", folder)
    assert (
        intel.lookup_url("https://raw.githubusercontent.com/other/file", folder) == []
    )


def test_index_refreshes_when_feed_changes(tmp_path: Path) -> None:
    folder = _store(tmp_path)
    assert intel.lookup_ip("50.16.16.211", folder)
    intel.write_feed(intel.build_feed("feodo", b"", "2026-10-16T04:00:00Z"), folder)
    assert intel.lookup_ip("50.16.16.211", folder) == []
    assert intel.status(folder)["feodo"]["count"] == 0


def test_indicators_and_hits(tmp_path: Path) -> None:
    folder = _store(tmp_path)
    inds = intel.indicators(
        "GET http://105.184.125.244:54343/i from 192.168.1.4 to c2.bad-domain.net ok"
    )
    assert ("ip", "105.184.125.244") in inds and ("ip", "192.168.1.4") not in inds
    assert intel.hits(intel.index(folder), inds) >= 2
    assert intel.hits(intel.index(folder), {("domain", "google.com")}) == 0


def test_missing_folder_is_empty(tmp_path: Path) -> None:
    assert intel.lookup_ip("8.8.8.8", tmp_path / "none") == []
    assert intel.status(tmp_path / "none") == {}
