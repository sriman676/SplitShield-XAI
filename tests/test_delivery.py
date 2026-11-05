"""Guaranteed delivery (queue, circuit breaker, order, dead letter) and the SIEM
formats LEEF 2.0 / ECS / OCSF, plus syslog over TLS (RFC 5425) to a local receiver."""

from __future__ import annotations

import json
import shutil
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from src.soc import delivery, sinks
from tests.test_sinks_live import ALERT


class FakeSiem:
    """A sink that can be switched off and on; records what arrived, in order."""

    def __init__(self) -> None:
        self.up = True
        self.got: list[Any] = []

    def __call__(self, alert: dict[str, Any], _s: dict[str, Any]) -> str:
        if not self.up:
            return "error: connection refused"
        self.got.append(alert["n"])
        return "ok"


def _settings(**delivery_cfg: Any) -> dict[str, Any]:
    s = sinks.load_settings()
    s["delivery"] |= delivery_cfg
    return s


def test_outage_queues_in_order_then_delivers_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    siem, s = FakeSiem(), _settings()
    assert delivery.deliver({"n": 0}, s, {"webhook": siem}) == {"webhook": "ok"}
    siem.up = False
    out = delivery.deliver({"n": 1}, s, {"webhook": siem})["webhook"]
    assert out.startswith("error") and out.endswith("(queued for retry)")
    t0 = time.perf_counter()
    for n in range(2, 50):  # breaker open: queued at once, no waiting on the SIEM
        assert delivery.deliver({"n": n}, s, {"webhook": siem})["webhook"].startswith("queued")
    assert time.perf_counter() - t0 < 5
    st = delivery.status()["webhook"]
    assert st["queued"] == 49 and st["breaker_open"]
    siem.up = True
    con = delivery._db()
    con.execute("UPDATE stats SET open_until=0")  # the backoff time is over
    con.close()
    monkeypatch.setattr(sinks, "network_senders", lambda _s: {"webhook": siem})
    delivery.flush(s)
    assert siem.got == list(range(50))  # nothing lost, original order
    st = delivery.status()["webhook"]
    assert st["queued"] == 0 and st["retried"] == 49 and not st["breaker_open"]


def test_new_alert_waits_behind_older_queued_ones() -> None:
    siem, s = FakeSiem(), _settings()
    siem.up = False
    delivery.deliver({"n": 1}, s, {"webhook": siem})
    con = delivery._db()
    con.execute("UPDATE stats SET open_until=0")  # breaker closed again
    con.close()
    siem.up = True
    delivery.deliver({"n": 2}, s, {"webhook": siem})
    assert siem.got == [1, 2]


def test_dead_letter_by_age_and_size_and_requeue() -> None:
    siem, s = FakeSiem(), _settings(max_queue=3, max_age_days=1)
    siem.up = False
    for n in range(5):
        delivery.deliver({"n": n}, s, {"syslog": siem})
    assert delivery.status()["syslog"]["queued"] == 3
    dead = delivery.dead_letters()
    assert [d["alert"]["n"] for d in dead] == [0, 1] and "queue full" in dead[0]["why"]
    con = delivery._db()
    con.execute("UPDATE queue SET created=created-2*86400")  # two days old
    con.close()
    delivery._flush_sink(delivery._db(), "syslog", s, siem)
    assert delivery.status()["syslog"]["queued"] == 0 and len(delivery.dead_letters()) == 5
    assert delivery.requeue_dead() == 5  # nothing thrown away
    assert delivery.status()["syslog"]["queued"] == 5 and delivery.dead_letters() == []
    assert list(Path(sinks.SETTINGS).parent.glob("dead_letter.requeued-*.jsonl"))


def test_send_uses_the_queue_and_never_raises() -> None:
    s = sinks.load_settings()
    s["jsonl"]["enabled"] = False
    s["syslog"] |= {"enabled": True, "host": "127.0.0.1", "port": 9, "protocol": "tcp"}
    first = sinks.send(ALERT, s)["syslog"]
    assert first.startswith("error") and "queued" in first
    assert sinks.send(ALERT, s)["syslog"].startswith("queued")
    assert delivery.status()["syslog"]["queued"] == 2


# --- formats ---------------------------------------------------------------------------


def test_leef_2() -> None:
    line = sinks.to_leef(ALERT | {"user": "bob^x", "host": "PC1"})
    head, _, attrs = line.partition("|^|")
    assert head == "LEEF:2.0|SplitShield|SplitShield-XAI|1.0|detector"
    kv = dict(p.split("=", 1) for p in attrs.replace("\\^", "\x00").split("^"))
    assert kv["src"] == "203.0.113.45" and kv["sev"] == "9" and kv["devTime"] == "1760000000500"
    assert kv["usrName"] == "bob\x00x"  # the delimiter inside a value is escaped
    assert kv["mitreTechnique"] == "T1498,T1046" and "\n" not in line


def test_ecs() -> None:
    d = sinks.to_ecs(ALERT | {"user": "bob", "dst_port": 443})
    assert d["@timestamp"] == "2025-10-09T08:53:20.500Z"
    assert d["event"]["kind"] == "alert" and d["event"]["severity"] == 9
    assert d["source"]["ip"] == "203.0.113.45" and d["destination"] == {"ip": "192.0.2.10", "port": 443}
    assert d["threat"]["framework"] == "MITRE ATT&CK"
    assert d["threat"]["technique"]["id"] == ["T1498", "T1046"]
    assert d["user"]["name"] == "bob"


def test_ocsf_detection_finding() -> None:
    d = sinks.to_ocsf(ALERT)
    assert (d["class_uid"], d["category_uid"], d["type_uid"]) == (2004, 2, 200401)
    assert d["time"] == 1760000000500 and d["severity_id"] == 5 and d["severity"] == "Critical"
    assert d["finding_info"]["attacks"][0]["technique"]["uid"] == "T1498"
    assert d["confidence_score"] == 97 and d["src_endpoint"] == {"ip": "203.0.113.45"}
    json.dumps(d)  # serialisable


def test_jsonl_outbox_in_ecs(tmp_path: Path) -> None:
    s = sinks.apply_preset(sinks.load_settings(), "Elastic")
    s["jsonl"]["path"] = str(tmp_path / "o.jsonl")
    assert sinks.send(ALERT, s) == {"jsonl": "ok"}
    assert json.loads((tmp_path / "o.jsonl").read_text("utf-8"))["event"]["kind"] == "alert"


def test_syslog_5424_header() -> None:
    line = sinks._syslog_line(ALERT, {"format": "leef", "header": "rfc5424"}).decode()
    assert line.startswith("<34>1 2025-10-09T08:53:20.500Z ") and " SplitShield - - - LEEF:2.0|" in line


# --- syslog over TLS -----------------------------------------------------------------


def _self_signed(tmp: Path) -> tuple[Path, Path]:
    openssl = shutil.which("openssl") or next(
        (str(p) for p in (Path(r"C:\Program Files\Git\usr\bin\openssl.exe"),) if p.is_file()), None)  # fmt: skip
    if not openssl:
        pytest.skip("openssl not available to make a test certificate")
    cert, key = tmp / "c.pem", tmp / "k.pem"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)  # fmt: skip
    return cert, key


def _read_frame(tls: ssl.SSLSocket, buf: bytearray) -> bytes:
    """One RFC 5425 frame: '<length> <message>'."""
    while b" " not in buf:
        buf += tls.recv(65536)
    head, _, rest = bytes(buf).partition(b" ")
    n = int(head)
    while len(rest) < n:
        rest += tls.recv(65536)
    buf[:] = rest[n:]
    return rest[:n]


def test_syslog_over_tls_rfc5425(tmp_path: Path) -> None:
    cert, key = _self_signed(tmp_path)
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(str(cert), str(key))
    srv = socket.create_server(("127.0.0.1", 0))
    got: list[bytes] = []
    conns: list[Any] = []

    def serve() -> None:
        conn, _ = srv.accept()
        conns.append(conn)
        with ctx.wrap_socket(conn, server_side=True) as tls:
            buf = bytearray()
            got.append(_read_frame(tls, buf))
            got.append(_read_frame(tls, buf))  # the second alert, same connection

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    cfg = {"host": "localhost", "port": srv.getsockname()[1], "protocol": "tls",
           "ca_file": str(cert), "verify_tls": True, "format": "ocsf"}  # fmt: skip
    assert sinks._send_syslog(sinks._syslog_line(ALERT, cfg), cfg) == "ok"
    assert sinks._send_syslog(sinks._syslog_line(ALERT, cfg | {"format": "leef"}), cfg) == "ok"
    th.join(5)
    assert len(conns) == 1  # one handshake for both alerts
    assert json.loads(got[0].decode().split("SplitShield: ", 1)[1])["class_uid"] == 2004
    assert b"LEEF:2.0|" in got[1]
    sinks._CONN.clear()
    # an untrusted certificate is refused unless the owner opts out of checking
    th = threading.Thread(target=lambda: srv.accept()[0].close(), daemon=True)
    th.start()
    assert sinks._send_syslog(b"x", cfg | {"ca_file": ""}).startswith("error")
    srv.close()


def test_audit_entries_can_go_to_the_siem(tmp_path: Path) -> None:
    from src.soc import audit

    s = sinks.load_settings()
    s["jsonl"]["path"] = str(tmp_path / "o.jsonl")
    sinks.save_settings(s)
    audit.record("boss", "rule_saved", {"rule": "x"})
    assert not (tmp_path / "o.jsonl").exists()  # off by default
    s["forward_audit"] = True
    sinks.save_settings(s)
    audit.record("boss", "rule_saved", {"rule": "x"})
    sent = json.loads((tmp_path / "o.jsonl").read_text("utf-8"))
    assert sent["kind"] == "audit" and sent["user"] == "boss" and sent["audit_hash"]


def test_default_outbox_follows_the_sandbox(tmp_path: Path) -> None:
    """2026-10-17: the default outbox path was fixed at import time, so moved
    stores (tests, sandboxes) still appended alerts to the real outbox."""
    assert sinks.load_settings()["jsonl"]["path"] == str(sinks.OUTBOX)
    assert "artifacts" not in str(sinks.OUTBOX) or "soc_state" in str(sinks.OUTBOX)
    sinks.send(ALERT)
    assert sinks.OUTBOX.is_file()


def test_pseudonymized_names_are_stable_and_off_by_default(tmp_path: Path) -> None:
    s = sinks.load_settings()
    s["jsonl"]["path"] = str(tmp_path / "o.jsonl")
    a = ALERT | {"user": "bob", "host": "PC1"}
    sinks.send(a, s)
    s["pseudonymize"]["enabled"] = True
    sinks.send(a, s)
    sinks.send(a | {"user": "alice"}, s)
    rows = [json.loads(x) for x in (tmp_path / "o.jsonl").read_text("utf-8").splitlines()]
    assert rows[0]["user"] == "bob"  # off by default
    assert rows[1]["user"].startswith("u-") and rows[1]["host"].startswith("h-")
    assert rows[1]["user"] != rows[2]["user"]
    sinks.send(a, s)
    again = json.loads((tmp_path / "o.jsonl").read_text("utf-8").splitlines()[-1])
    assert again["user"] == rows[1]["user"]  # same name, same pseudonym
