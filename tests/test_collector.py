"""Central log collection over HTTPS (agent keys, certificate pinning, LAN-only,
lockout, size limits) and the read-only REST API with tokens."""

from __future__ import annotations

import gzip
import hashlib
import json
import ssl
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from src.soc import audit, collector, engine

LINES = [f"Oct 16 10:00:{i:02d} web1 sshd[{i}]: Failed password for root from 203.0.113.77 "
         f"port {40000 + i} ssh2" for i in range(30)]  # fmt: skip


@pytest.fixture()
def server() -> Iterator[str]:
    srv = collector.make_server(port=0, bind="127.0.0.1")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"https://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _pinned() -> ssl.SSLContext:
    """What the agents do: no CA, but the certificate's SHA-256 must match."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def call(url: str, secret: str = "", body: Any = None, gz: bool = False) -> tuple[int, Any]:
    data = None
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    if body is not None:
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        if gz:
            data = gzip.compress(data)
            headers["Content-Encoding"] = "gzip"
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, context=_pinned(), timeout=30) as r:  # noqa: S310 - test server
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_certificate_fingerprint_matches_what_the_server_presents(server: str) -> None:
    host, port = server[8:].split(":")
    der = ssl.get_server_certificate((host, int(port)))
    assert hashlib.sha256(ssl.PEM_cert_to_DER_cert(der)).hexdigest() == collector.fingerprint()


def test_agent_ingest_goes_through_the_normal_pipeline(server: str) -> None:
    key = collector.new_agent("boss", "web1")
    assert key.startswith("ssa_") and key not in (collector.home() / "agents.json").read_text("utf-8")
    assert call(f"{server}/agent/ping", key) == (200, {"ok": True, "agent": "web1"})
    before = engine.event_count()
    code, out = call(f"{server}/ingest", key, {"source": "auth", "host": "web1", "lines": LINES}, gz=True)
    assert code == 200 and out["ok"] and out["name"].startswith("agent-web1-web1-auth")
    assert engine.event_count() >= before + 30
    assert collector.listing("agents")["web1"]["count"] == 30
    assert any(r["action"] == "agent_created" for r in audit.read())


def test_wrong_or_disabled_keys_and_lockout(server: str) -> None:
    key = collector.new_agent("boss", "pc2")
    collector.set_disabled("agents", "boss", "pc2", True)
    assert call(f"{server}/ingest", key, {"lines": LINES})[0] == 401
    for _ in range(collector.FAIL_LIMIT - 2):
        assert call(f"{server}/ingest", "ssa_wrong", {"lines": LINES})[0] == 401
    assert call(f"{server}/ingest", "ssa_wrong", {"lines": LINES})[0] == 401  # 10th failure
    collector.set_disabled("agents", "boss", "pc2", False)
    assert call(f"{server}/ingest", key, {"lines": LINES})[0] == 429  # address blocked now
    assert any(r["action"] == "collector_blocked" for r in audit.read())


def test_bad_bodies_are_refused(server: str) -> None:
    key = collector.new_agent("boss", "pc3")
    assert call(f"{server}/ingest", key, b"not json")[0] == 400
    assert call(f"{server}/ingest", key, {"nothing": 1})[0] == 400
    assert call(f"{server}/ingest", key, [1, 2])[0] == 400
    bomb = gzip.compress(b"\x00" * (collector.MAX_INFLATED + 1024))  # ~64 KB -> 64 MB
    req = urllib.request.Request(f"{server}/ingest", data=bomb, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Encoding": "gzip"})  # fmt: skip
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, context=_pinned(), timeout=60)  # noqa: S310
    assert e.value.code == 400
    assert call(f"{server}/nowhere", key, {"lines": []})[0] == 404


def test_read_only_api_with_tokens(server: str) -> None:
    key = collector.new_agent("boss", "web1")
    call(f"{server}/ingest", key, {"source": "auth", "lines": LINES})
    tok = collector.new_token("boss", "splunk-soar")
    assert call(f"{server}/api/v1/health", key)[0] == 401  # agent keys cannot read
    code, h = call(f"{server}/api/v1/health", tok)
    assert code == 200 and h["ok"] and h["events"] >= 30
    code, a = call(f"{server}/api/v1/alerts?limit=5", tok)
    assert code == 200 and len(a["alerts"]) <= 5
    assert all("logits" not in r for r in a["alerts"])
    assert call(f"{server}/api/v1/incidents?since=0", tok)[0] == 200
    code, s = call(f"{server}/api/v1/search?q=failed%20password%20for%20root", tok)
    assert code == 200 and s["count"] >= 1
    assert call(f"{server}/api/v1/search?q=", tok)[0] == 400
    assert call(f"{server}/api/v1/alerts?limit=abc", tok)[0] == 400
    assert call(f"{server}/api/v1/../../etc/passwd", tok)[0] == 404
    collector.remove("tokens", "boss", "splunk-soar")
    assert call(f"{server}/api/v1/health", tok)[0] == 401


@pytest.mark.parametrize(("ip", "ok"), [("192.168.1.20", True), ("10.0.0.5", True), ("127.0.0.1", True),
                                        ("172.16.4.4", True), ("fe80::1", True), ("::ffff:192.168.1.2", True),
                                        ("8.8.8.8", False), ("2001:4860::8888", False), ("garbage", False)])  # fmt: skip
def test_lan_only(ip: str, ok: bool) -> None:
    assert collector._allowed_ip(ip, lan_only=True) is ok
    assert collector._allowed_ip(ip, lan_only=False)


def test_names_are_checked() -> None:
    for bad in ("", "a b", "../x", "x" * 65):
        with pytest.raises(ValueError):
            collector.new_agent("boss", bad)
    collector.new_agent("boss", "dup")
    with pytest.raises(ValueError):
        collector.new_agent("boss", "dup")


def test_idle_or_slow_clients_cannot_block_the_receiver(server: str) -> None:
    """Pentest 2026-10-17: a client that connects and never sends a TLS handshake
    must not stop other agents (the handshake used to run in the accept loop)."""
    import socket as _s
    import time as _t

    host, port = server[8:].split(":")
    idle = [_s.create_connection((host, int(port))) for _ in range(3)]  # never handshake
    try:
        key = collector.new_agent("boss", "after-idle")
        t0 = _t.time()
        assert call(f"{server}/agent/ping", key)[0] == 200
        assert _t.time() - t0 < 10
    finally:
        for s in idle:
            s.close()
