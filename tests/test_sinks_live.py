"""SIEM delivery over real sockets: local receivers speaking each SIEM's protocol.

No Splunk/Wazuh/QRadar install is needed: the receivers check exactly what those
products check on input (HEC auth header + JSON envelope, RFC 3164 syslog + CEF,
one JSON object per line for agent/Filebeat tailing).
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.soc import sinks

ALERT = {
    "ts": 1_760_000_000.5,
    "kind": "detector",
    "title": "DDoS|flood from 203.0.113.45",  # '|' must be escaped in the CEF header
    "severity": 9,
    "src_ip": "203.0.113.45",
    "dst_ip": "192.0.2.10",
    "techniques": [{"id": "T1498"}, "T1046"],
    "confidence": 0.97,
    "reason": "rate=9000 pps\ninjected=line",  # CR/LF must not split the record
}


def _cef_fields(line: str) -> tuple[list[str], dict[str, str]]:
    """Minimal CEF parser (header split on unescaped '|', extension on key=)."""
    cef = line[line.index("CEF:") :]
    parts = re.split(r"(?<!\\)\|", cef, maxsplit=7)
    ext = parts[7]
    keys = list(re.finditer(r"(?:^| )([A-Za-z0-9]+)=", ext))
    vals = {}
    for i, m in enumerate(keys):
        end = keys[i + 1].start() if i + 1 < len(keys) else len(ext)
        vals[m.group(1)] = ext[m.end() : end].replace("\\=", "=").replace("\\\\", "\\")
    return parts[:7], vals


def _check_cef(line: str) -> None:
    header, ext = _cef_fields(line)
    assert header[0] == "CEF:0" and header[1] == "SplitShield"
    assert header[5] == "DDoS\\|flood from 203.0.113.45" and header[6] == "9"
    assert ext["src"] == "203.0.113.45" and ext["dst"] == "192.0.2.10"
    assert ext["cs1"] == "T1498,T1046" and ext["rt"] == "1760000000500"
    assert "\n" not in line and "\r" not in line
    assert ext["msg"] == "rate=9000 pps injected=line"  # "=" escaped, then restored


def test_syslog_udp_rfc3164_cef() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(5)
    cfg = {"host": "127.0.0.1", "port": srv.getsockname()[1], "protocol": "udp"}
    assert sinks._send_syslog(sinks._syslog_line(ALERT), cfg) == "ok"
    data = srv.recv(65535).decode()
    srv.close()
    # <PRI>Mmm dd hh:mm:ss host tag: ; facility auth(4)*8 + crit(2) for severity 9
    assert re.match(
        r"<34>[A-Z][a-z]{2} [ 1-3]\d \d\d:\d\d:\d\d \S+ SplitShield: ", data
    )
    _check_cef(data)


def test_syslog_tcp_newline_framed() -> None:
    srv = socket.create_server(("127.0.0.1", 0))
    got: list[bytes] = []

    def accept() -> None:
        conn, _ = srv.accept()
        with conn:
            buf = b""
            while not buf.endswith(b"\n"):
                buf += conn.recv(65535)
            got.append(buf)

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    cfg = {"host": "127.0.0.1", "port": srv.getsockname()[1], "protocol": "tcp"}
    assert sinks._send_syslog(sinks._syslog_line(ALERT), cfg) == "ok"
    t.join(5)
    records = got[0].decode().split("\n")
    assert records[-1] == "" and len(records) == 2  # exactly one framed record
    _check_cef(records[0])
    # 2026-10-17: the connection is kept open; when the receiver has closed it, the
    # next alert must go over a NEW connection, never into the dead one
    t = threading.Thread(target=accept, daemon=True)
    t.start()
    time.sleep(0.2)  # the receiver's close has reached us
    assert sinks._send_syslog(sinks._syslog_line(ALERT), cfg) == "ok"
    t.join(5)
    srv.close()
    assert len(got) == 2 and got[1].endswith(b"\n")
    sinks._CONN.clear()


class _HEC(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers["Content-Length"]))
        ok = (
            self.path == "/services/collector/event"
            and self.headers.get("Authorization") == "Splunk 0000-test-token"
        )
        _HEC.seen.append({"auth_ok": ok, "body": json.loads(body)})
        self.send_response(200 if ok else 401)
        self.end_headers()
        self.wfile.write(b'{"text":"Success","code":0}' if ok else b"{}")

    def log_message(self, *a: object) -> None:  # keep test output quiet
        pass


def test_splunk_hec_webhook(monkeypatch) -> None:
    monkeypatch.delenv("SOC_TOKEN", raising=False)
    srv = HTTPServer(("127.0.0.1", 0), _HEC)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = sinks.apply_preset(sinks.load_settings(), "Splunk")
        s["jsonl"]["enabled"] = False
        s["webhook"] |= {
            "url": f"http://127.0.0.1:{srv.server_port}/services/collector/event",
            "token": "0000-test-token",
        }
        assert sinks.send(ALERT, s) == {"webhook": "ok"}
        hit = _HEC.seen[-1]
        assert hit["auth_ok"]
        assert hit["body"]["time"] == ALERT["ts"]  # event time, not arrival time
        assert hit["body"]["sourcetype"] == "splitshield:alert"
        assert hit["body"]["event"]["src_ip"] == "203.0.113.45"
        s["webhook"]["token"] = "wrong"
        assert sinks.send(ALERT, s)["webhook"].startswith("error")  # 401 surfaced
    finally:
        srv.shutdown()


def test_jsonl_outbox_for_wazuh_and_filebeat(tmp_path) -> None:
    s = sinks.load_settings()
    s["jsonl"]["path"] = str(tmp_path / "outbox.jsonl")
    s["syslog"]["enabled"] = s["webhook"]["enabled"] = False
    assert sinks.send(ALERT, s) == {"jsonl": "ok"}
    assert sinks.send(ALERT, s) == {"jsonl": "ok"}
    lines = (tmp_path / "outbox.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 2  # one JSON object per line (Wazuh <log_format>json)
    assert json.loads(lines[0])["reason"] == ALERT["reason"]


def test_unreachable_siem_never_raises() -> None:
    s = sinks.load_settings()
    s["jsonl"]["enabled"] = False
    s["syslog"] |= {"enabled": True, "host": "127.0.0.1", "port": 9, "protocol": "tcp"}
    s["webhook"] |= {"enabled": True, "url": "http://127.0.0.1:9/x"}
    res = sinks.send(ALERT, s)  # detection must survive a SIEM outage
    assert res["syslog"].startswith("error") and res["webhook"].startswith("error")
