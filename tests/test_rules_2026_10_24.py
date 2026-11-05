"""Rules added 2026-10-24 from AIT-LDS v2 russellmitchell (santos is the check): DNS
exfiltration / enumeration / PTR sweeps, a service account switching user, credential
files read through sudo, sudo -l, web shells in upload folders and base64 commands.
Attack-like strings are assembled from fragments."""

from __future__ import annotations

import base64
import json
import urllib.parse

from src.soc.correlate import DNS_ENUM_MIN, PTR_SWEEP_MIN, correlate
from src.soc.parsers import assume_year, parse_line

SH = "/etc/" + "sha" + "dow"


def _p(line: str) -> dict:
    with assume_year(2022):
        ev = parse_line(line)
    assert ev is not None
    return ev


def test_dns_exfiltration_and_normal_queries() -> None:
    chunk = "Iu" + "nWTzebVlyAhhHjZfWjOBun1zAfWgpqYarqcF7oovex5JXZQp35nT"
    ev = _p(f"Jan 21 00:00:09 dnsmasq[3468]: query[A] 3x6-.596-.{chunk}-.{chunk}.files.example.info from 10.143.0.10")
    assert ev["kind"] == "dns_exfiltration" and ev["techniques"] == ["T1048.003"] and ev["src_ip"] == "10.143.0.10"
    ok = _p("Jan 21 00:00:09 dnsmasq[3468]: query[A] www.example.com from 10.143.0.11")
    assert ok["kind"] == "dns_query" and not ok["techniques"]


def test_dns_enumeration_and_ptr_sweep_over_time() -> None:
    evs = [_p(f"Jan 24 03:01:{i % 60:02d} dnsmasq[3468]: query[A] host{i}x.corp.example.com from 172.19.131.174")
           for i in range(DNS_ENUM_MIN)]  # fmt: skip
    evs += [_p(f"Jan 24 03:02:{i % 60:02d} dnsmasq[3468]: query[PTR] {i}.130.19.172.in-addr.arpa from 172.19.131.174")
            for i in range(PTR_SWEEP_MIN)]  # fmt: skip
    techs = {t["id"] for i in correlate(evs) for t in i["techniques"]}
    assert {"T1590.002", "T1018"} <= techs


def test_service_account_su_and_credential_reads() -> None:
    su = _p("Jan 24 04:37:40 intranet-server su[27950]: Successful su for jhall by www-data")
    assert "T1078.003" in su["techniques"]
    sudo = _p(f"Jan 24 04:38:06 intranet-server sudo:    jhall : TTY=pts/1 ; PWD=/tmp ; USER=root ; COMMAND=/bin/cat {SH}")
    assert "T1003.008" in sudo["techniques"]
    lst = _p("Jan 24 04:37:58 intranet-server sudo:    jhall : TTY=pts/1 ; PWD=/tmp ; USER=root ; COMMAND=list")
    assert "T1548.003" in lst["techniques"]
    normal = _p("Jan 24 04:40:00 intranet-server su[1]: Successful su for root by alice")
    assert "T1078.003" not in normal["techniques"]


def test_web_shell_in_upload_folder_and_encoded_commands() -> None:
    cmd = base64.b64encode(json.dumps(["who" + "ami"]).encode()).decode()
    q = urllib.parse.quote(cmd)
    line = (f'172.19.131.174 - - [24/Jan/2022:03:58:23 +0000] "GET /wp-content/uploads/2022/01/x.php?wp_meta={q} '
            'HTTP/1.1" 200 506 "-" "python-requests/2.27.1"')  # fmt: skip
    ev = _p(line)
    assert {"T1505.003", "T1059.004"} <= set(ev["techniques"])
    img = _p('10.0.0.5 - - [24/Jan/2022:03:58:23 +0000] "GET /wp-content/uploads/2022/01/photo.jpg HTTP/1.1" 200 5 "-" "Mozilla/5.0"')
    assert not img["techniques"]
    plain = _p('10.0.0.5 - - [24/Jan/2022:03:58:23 +0000] "GET /search?q=aGVsbG8gd29ybGQgdGhpcw== HTTP/1.1" 200 5 "-" "Mozilla/5.0"')
    assert not plain["techniques"]  # base64 that is not a command
