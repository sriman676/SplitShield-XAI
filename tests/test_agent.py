"""The cross-platform agent end to end against the real receiver: file following
(rotation-safe), certificate pinning, spool during an outage, nothing lost."""

from __future__ import annotations

import importlib.util
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.soc import collector, engine

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ss_agent", ROOT / "scripts" / "agents" / "splitshield_agent.py")
assert _spec and _spec.loader
agentmod: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agentmod)


def _line(i: int) -> str:
    return f"Oct 17 01:00:{i % 60:02d} web9 sshd[{i}]: Failed password for root from 198.51.100.9 port {50000 + i} ssh2\n"


class Receiver:
    def __init__(self) -> None:
        self.srv: Any = None
        self.port = 0

    def start(self) -> None:
        self.srv = collector.make_server(port=self.port, bind="127.0.0.1")
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture()
def rx() -> Iterator[Receiver]:
    r = Receiver()
    r.start()
    yield r
    r.stop()


def _agent(tmp: Path, port: int, log: Path, fp: str | None = None) -> Any:
    home = tmp / "agent"
    home.mkdir(exist_ok=True)
    cfg = {"server": f"https://127.0.0.1:{port}", "key": collector.new_agent("boss", f"a{port}"),
           "fingerprint": fp or collector.fingerprint(), "files": [str(log)], "journald": False,
           "macos_unified": False, "windows_logs": [], "backfill": True}  # fmt: skip
    (home / "agent.json").write_text(json.dumps(cfg), "utf-8")
    a = agentmod.Agent(home)
    a.load()
    return a


def test_agent_sends_follows_and_survives_an_outage(tmp_path: Path, rx: Receiver) -> None:
    log = tmp_path / "auth.log"
    log.write_text("".join(_line(i) for i in range(20)) + "half-written", "utf-8")
    a = _agent(tmp_path, rx.port, log)
    assert a.ping()[0] == 200
    before = engine.event_count()
    assert a.run_once() == 20  # the half-written last line waits
    assert engine.event_count() == before + 20
    rx.stop()  # receiver down: the agent spools, the bookmark still moves
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(" line\n" + "".join(_line(i) for i in range(20, 35)))
    assert a.run_once() == 16
    assert len(list((a.home / "spool").glob("*.json.gz"))) == 1
    assert engine.event_count() == before + 20
    rx.start()  # back up: the spool is sent first, in order
    assert a.run_once() == 0
    assert not list((a.home / "spool").glob("*.json.gz"))
    assert engine.event_count() == before + 36
    log.write_text("".join(_line(i) for i in range(3)), "utf-8")  # rotated / truncated
    assert a.run_once() == 3


def test_agent_refuses_a_server_with_another_certificate(tmp_path: Path, rx: Receiver) -> None:
    log = tmp_path / "auth.log"
    log.write_text(_line(1), "utf-8")
    a = _agent(tmp_path, rx.port, log, fp="00" * 32)
    with pytest.raises(OSError, match="pinned fingerprint"):
        a.ping()
    before = engine.event_count()
    a.run_once()  # nothing reaches the impostor; the batch is kept in the spool
    assert engine.event_count() == before
    assert len(list((a.home / "spool").glob("*.json.gz"))) == 1


def test_windows_event_xml_rows() -> None:
    xml = ('<Events><Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event"><System>'
           '<Provider Name="Microsoft-Windows-Security-Auditing"/><EventID>4625</EventID>'
           '<EventRecordID>77</EventRecordID><TimeCreated SystemTime="2026-10-17T01:00:00Z"/>'
           '<Channel>Security</Channel><Computer>PC9</Computer></System><EventData>'
           '<Data Name="TargetUserName">bob</Data><Data Name="IpAddress">198.51.100.9</Data>'
           '</EventData></Event></Events>')  # fmt: skip
    (rid, row), = agentmod._win_records(xml)
    assert rid == 77 and row["Id"] == 4625 and row["EventData"]["TargetUserName"] == "bob"
    assert "IpAddress: 198.51.100.9" in row["Message"] and row["Computer"] == "PC9"


def test_a_batch_whose_answer_was_lost_is_stored_once(tmp_path: Path, rx: Receiver) -> None:
    """2026-10-23: the receiver stored a batch but the agent never got the answer (time-out
    under load, outage): the agent spooled it and sent it again, and the receiver stored it
    twice. Batches carry a batch_id and the receiver acknowledges a repeat without storing."""
    log = tmp_path / "auth.log"
    log.write_text("".join(_line(i) for i in range(10)), "utf-8")
    a = _agent(tmp_path, rx.port, log)
    before = engine.event_count()
    real = a._post
    lost = {"n": 0}

    def post_answer_lost(path: str, payload: bytes | None) -> tuple[int, dict[str, Any]]:
        out = real(path, payload)  # the receiver stores it ...
        if path == "/ingest" and lost["n"] == 0:
            lost["n"] += 1
            raise TimeoutError("answer lost")  # ... but the agent never hears back
        return out

    a._post = post_answer_lost
    assert a.run_once() == 10  # spooled, then re-sent from the spool in the same pass
    assert not list((a.home / "spool").glob("*.json.gz"))
    assert engine.event_count() == before + 10  # stored once
    doc = {"source": "x", "host": "h", "lines": [_line(99)], "batch_id": "fixed-id"}
    assert "result" in collector.ingest_batch("boss", doc)
    assert collector.ingest_batch("boss", doc).get("duplicate") is True
    assert engine.event_count() == before + 11


def _ca_and_server(tmp: Path, name: str) -> tuple[Path, Path, Path]:
    """A throwaway CA and a server certificate for 127.0.0.1 signed by it."""
    import datetime as dt
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = dt.datetime.now(dt.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{name} test CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
          .not_valid_after(now + dt.timedelta(days=2))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .sign(ca_key, hashes.SHA256()))  # fmt: skip
    key = ec.generate_private_key(ec.SECP256R1())
    srv = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
           .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
           .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=2))
           .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
           .sign(ca_key, hashes.SHA256()))  # fmt: skip
    d = tmp / name
    d.mkdir()
    (d / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (d / "server.pem").write_bytes(srv.public_bytes(serialization.Encoding.PEM))
    (d / "server.key").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                     serialization.NoEncryption()))  # fmt: skip
    return d / "ca.pem", d / "server.pem", d / "server.key"


def test_receiver_with_the_organisations_own_certificate(tmp_path: Path) -> None:
    """2026-10-23: the receiver can use a company-CA certificate; agents with ca_file verify
    it normally (no fingerprint needed); another CA, or no pin and no CA, is refused."""
    from src.soc import sinks

    ca, cert, key = _ca_and_server(tmp_path, "company")
    other_ca, _, _ = _ca_and_server(tmp_path, "other")
    s = sinks.load_settings()
    s["collector"] |= {"cert_file": str(cert), "key_file": str(key)}
    sinks.save_settings(s)
    rx = Receiver()
    rx.start()
    try:
        log = tmp_path / "auth.log"
        log.write_text("", "utf-8")
        a = _agent(tmp_path, rx.port, log)
        base = dict(a.cfg)
        a.cfg = base | {"ca_file": str(ca), "fingerprint": ""}
        assert a.ping()[0] == 200
        assert collector.fingerprint() == __import__("hashlib").sha256(
            __import__("ssl").PEM_cert_to_DER_cert(cert.read_text("ascii"))).hexdigest()  # pinning still possible
        a.cfg = base | {"ca_file": str(other_ca), "fingerprint": ""}
        with pytest.raises(OSError):
            a.ping()
        a.cfg = base | {"fingerprint": ""}
        with pytest.raises(OSError):
            a.ping()
    finally:
        rx.stop()
