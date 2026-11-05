"""Central log collection + read-only REST API (owner 2026-10-16, enterprise).

One HTTPS server on the LAN (default port 8600):

* ``POST /ingest`` - agents on other machines (Windows, macOS, Linux; see
  scripts/agents/) send batches of logs with their own **agent key**. The logs go
  through exactly the same parsers, rules and models as files analysed on this PC.
* ``GET /api/v1/...`` - SIEM / SOAR tools read health, incidents, alerts and search
  results with an **API token** (read-only: nothing can be changed through the API).

Security
  * TLS with a self-signed certificate made on first start (``collector/cert.pem``);
    agents pin its SHA-256 fingerprint, so no certificate authority is needed and a
    man-in-the-middle with another certificate is refused.
  * Keys and tokens are 256-bit random secrets; only their SHA-256 is stored, and
    each is shown once. They can be disabled or removed at any time (dashboard:
    Settings -> Agents & API, permission ``collection``); every change is audited.
  * LAN only by default: clients outside private / loopback / link-local ranges are
    refused (and the firewall rule from scripts/allow_collector_firewall.ps1 or
    install.sh limits the port to the local subnet).
  * 10 failed authentications from one address within 5 minutes block it for 15
    minutes; bodies are capped (16 MB, 64 MB after gzip); per-agent rate limit.
  * The dashboard itself stays on localhost; this server never serves it.

    python -m src.soc.collector serve [--port 8600]
    python -m src.soc.collector new-agent NAME        # prints the key once
    python -m src.soc.collector new-token NAME        # prints the token once
    python -m src.soc.collector fingerprint
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import socket
import ssl
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 8600
MAX_BODY = 16 * 1024 * 1024
MAX_INFLATED = 64 * 1024 * 1024
FAIL_LIMIT, FAIL_WINDOW_S, BLOCK_S = 10, 300.0, 900.0
API_MAX_ROWS = 1000
CONN_TIMEOUT_S = 30.0  # handshake + each read / write on one connection
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# --- state files -----------------------------------------------------------------------


def home() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).parent / "collector"


def _load(name: str) -> dict[str, Any]:
    try:
        return dict(json.loads((home() / name).read_text("utf-8")))
    except (OSError, ValueError):
        return {}


def _save(name: str, d: dict[str, Any]) -> None:
    p = home() / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), "utf-8")
    tmp.replace(p)


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


# --- certificate -----------------------------------------------------------------------


def _local_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(str(info[4][0]).split("%")[0])
    except OSError:
        pass
    return sorted(ips)


def own_cert() -> tuple[Path, Path] | None:
    """The organisation's own certificate (Settings collector.cert_file / key_file, PEM),
    e.g. issued by the company CA, so agents can verify it the standard way (2026-10-23)."""
    c = settings()
    cf, kf = str(c.get("cert_file") or ""), str(c.get("key_file") or "")
    if not cf or not kf:
        return None
    cert, key = Path(cf), Path(kf)
    if not (cert.is_file() and key.is_file()):
        raise FileNotFoundError(f"collector.cert_file / key_file not found: {cert}, {key}")
    ssl.create_default_context(ssl.Purpose.CLIENT_AUTH).load_cert_chain(str(cert), str(key))  # must load
    return cert, key


def ensure_cert() -> tuple[Path, Path]:
    """The server certificate + key: the organisation's own (own_cert) when configured,
    otherwise self-signed, made once, valid 825 days."""
    own = own_cert()
    if own:
        return own
    cert, key = home() / "cert.pem", home() / "key.pem"
    if cert.is_file() and key.is_file():
        return cert, key
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    pk = ec.generate_private_key(ec.SECP256R1())
    host = socket.gethostname()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"SplitShield-XAI {host}")])
    alt: list[x509.GeneralName] = [x509.DNSName(host), x509.DNSName("localhost")]
    for ip in _local_ips():
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            continue
    now = _dt.datetime.now(_dt.UTC)
    crt = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
           .public_key(pk.public_key()).serial_number(x509.random_serial_number())
           .not_valid_before(now - _dt.timedelta(minutes=5))
           .not_valid_after(now + _dt.timedelta(days=825))
           .add_extension(x509.SubjectAlternativeName(alt), critical=False)
           .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
           .sign(pk, hashes.SHA256()))  # fmt: skip
    home().mkdir(parents=True, exist_ok=True)
    key.write_bytes(pk.private_bytes(serialization.Encoding.PEM,
                                     serialization.PrivateFormat.PKCS8,
                                     serialization.NoEncryption()))  # fmt: skip
    cert.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    return cert, key


def fingerprint() -> str:
    """SHA-256 of the certificate (DER) - agents pin this value."""
    cert, _ = ensure_cert()
    der = ssl.PEM_cert_to_DER_cert(cert.read_text("ascii"))
    return hashlib.sha256(der).hexdigest()


# --- agent keys and API tokens ---------------------------------------------------------


def _new_secret(kind: str, by: str, name: str, prefix: str) -> str:
    from src.soc import audit

    if not _NAME.match(name):
        raise ValueError("name: 1-64 letters, digits, dot, dash or underscore")
    store = _load(f"{kind}.json")
    if name in store:
        raise ValueError(f"{name!r} exists already - remove it first or pick another name")
    secret = prefix + secrets.token_urlsafe(32)
    store[name] = {"hash": _digest(secret), "created": time.strftime("%Y-%m-%d %H:%M"),
                   "by": by, "disabled": False, "last_seen": None, "last_ip": None,
                   "count": 0}  # fmt: skip
    _save(f"{kind}.json", store)
    audit.record(by or "system", f"{kind[:-1]}_created", {"name": name})
    return secret


def new_agent(by: str, name: str) -> str:
    """Make an agent key (returned once; only its hash is stored)."""
    return _new_secret("agents", by, name, "ssa_")


def new_token(by: str, name: str) -> str:
    """Make a read-only API token (returned once; only its hash is stored)."""
    return _new_secret("tokens", by, name, "sst_")


def listing(kind: str) -> dict[str, dict[str, Any]]:
    return {n: {k: v for k, v in e.items() if k != "hash"} for n, e in _load(f"{kind}.json").items()}


def set_disabled(kind: str, by: str, name: str, disabled: bool) -> None:
    from src.soc import audit

    store = _load(f"{kind}.json")
    if name not in store:
        raise ValueError(f"no {kind[:-1]} {name!r}")
    store[name]["disabled"] = bool(disabled)
    _save(f"{kind}.json", store)
    audit.record(by or "system", f"{kind[:-1]}_{'disabled' if disabled else 'enabled'}", {"name": name})


def remove(kind: str, by: str, name: str) -> None:
    from src.soc import audit

    store = _load(f"{kind}.json")
    if store.pop(name, None) is None:
        raise ValueError(f"no {kind[:-1]} {name!r}")
    _save(f"{kind}.json", store)
    audit.record(by or "system", f"{kind[:-1]}_removed", {"name": name})


_SEEN_LOCK = threading.Lock()


def _check(kind: str, secret: str, ip: str, add: int = 0, extra: dict[str, Any] | None = None) -> str | None:
    """Name of the enabled key/token matching ``secret`` (constant-time), else None."""
    if not secret:
        return None
    want = _digest(secret)
    with _SEEN_LOCK:
        store = _load(f"{kind}.json")
        for name, e in store.items():
            if hmac.compare_digest(str(e.get("hash", "")), want):
                if e.get("disabled"):
                    return None
                e["last_seen"], e["last_ip"] = time.strftime("%Y-%m-%d %H:%M:%S"), ip
                e["count"] = int(e.get("count") or 0) + add
                e.update(extra or {})
                _save(f"{kind}.json", store)
                return name
    return None


# --- request handling ------------------------------------------------------------------


def _allowed_ip(ip: str, lan_only: bool) -> bool:
    if not lan_only:
        return True
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return a.is_private or a.is_loopback or a.is_link_local


class _Guard:
    """Failed-auth counting per address and per-agent byte budgets."""

    def __init__(self, mb_per_min: float) -> None:
        self.fails: dict[str, list[float]] = {}
        self.blocked: dict[str, float] = {}
        self.bytes: dict[str, list[tuple[float, int]]] = {}
        self.limit = mb_per_min * 1024 * 1024
        self.lock = threading.Lock()

    def is_blocked(self, ip: str) -> bool:
        with self.lock:
            return self.blocked.get(ip, 0) > time.time()

    def fail(self, ip: str) -> None:
        now = time.time()
        with self.lock:
            f = [t for t in self.fails.get(ip, []) if now - t < FAIL_WINDOW_S] + [now]
            self.fails[ip] = f
            if len(f) >= FAIL_LIMIT:
                self.blocked[ip] = now + BLOCK_S
                self.fails[ip] = []
                from src.soc import audit

                audit.record(f"ip:{ip}", "collector_blocked", {"failures": len(f)}, ok=False)

    def over_budget(self, agent: str, n: int) -> bool:
        now = time.time()
        with self.lock:
            b = [(t, k) for t, k in self.bytes.get(agent, []) if now - t < 60] + [(now, n)]
            self.bytes[agent] = b
            return sum(k for _, k in b) > self.limit


_INGEST_LOCK = threading.Lock()
_DET: dict[str, Any] = {}


def _detector() -> Any:
    if "det" not in _DET:
        from src.soc import engine

        _DET["det"] = engine.load_detector()
    return _DET["det"]


def _inflate(body: bytes) -> bytes:
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = d.decompress(body, MAX_INFLATED)
    if d.unconsumed_tail:
        raise ValueError("body too large after gzip")
    return out


def ingest_batch(agent: str, doc: dict[str, Any]) -> dict[str, Any]:
    """One agent batch -> the normal ingest path. ``doc``: {"source": str, "host": str,
    and either "events": [Get-WinEvent-like rows] or "lines": [log lines]}."""
    from src.soc import engine

    source = re.sub(r"[^\w.-]+", "_", str(doc.get("source") or "log"))[:40]
    host = re.sub(r"[^\w.-]+", "_", str(doc.get("host") or agent))[:64]
    if isinstance(doc.get("events"), list):
        text, ext = json.dumps(doc["events"]), "json"
    elif isinstance(doc.get("lines"), list):
        text, ext = "\n".join(str(x) for x in doc["lines"]), "log"
    else:
        raise ValueError("batch needs 'events' or 'lines'")
    name = f"agent-{agent}-{host}-{source}.{ext}"
    bid = str(doc.get("batch_id") or "")[:64]
    with _INGEST_LOCK:  # one batch at a time through the models
        if bid and _seen(agent, bid):
            # stored before, but the agent never got our answer (outage, timeout):
            # it re-sent from its spool - acknowledge without storing it twice
            return {"name": name, "duplicate": True}
        res = engine.ingest_text(text, name, _detector(), True, observer=host.lower())
        if bid:
            _mark_seen(agent, bid)
    return {"name": name, "result": res}


SEEN_DAYS = 14  # batch ids remembered this long (agents re-send spooled batches)


def _seen_db() -> Any:
    import sqlite3

    path = home() / "batches_seen.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS seen(agent TEXT, id TEXT, ts REAL, PRIMARY KEY(agent, id))")
    return con


def _seen(agent: str, bid: str) -> bool:
    con = _seen_db()
    try:
        return con.execute("SELECT 1 FROM seen WHERE agent=? AND id=?", (agent, bid)).fetchone() is not None
    finally:
        con.close()


def _mark_seen(agent: str, bid: str) -> None:
    con = _seen_db()
    try:
        with con:
            con.execute("INSERT OR IGNORE INTO seen VALUES (?, ?, ?)", (agent, bid, time.time()))
            con.execute("DELETE FROM seen WHERE ts < ?", (time.time() - SEEN_DAYS * 86400,))
    finally:
        con.close()


def _clean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drop = ("logits", "features")
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def api(path: str, q: dict[str, list[str]]) -> Any:
    """Read-only API answers (raises KeyError for unknown paths)."""
    from src.soc import engine

    def num(k: str, d: float) -> float:
        try:
            return float(q.get(k, [d])[0])
        except ValueError as exc:
            raise ValueError(f"{k} must be a number") from exc

    limit = int(max(1, min(num("limit", 200), API_MAX_ROWS)))
    since = num("since", 0)
    if path == "/api/v1/health":
        from src.soc import delivery

        return {"ok": True, "events": engine.event_count(), "time": time.time(),
                "delivery": {k: {"queued": v["queued"], "dead": v["dead"]}
                             for k, v in delivery.status().items()}}  # fmt: skip
    if path == "/api/v1/incidents":
        level = q.get("level", [""])[0]
        rows = [i for i in engine.incidents() if i["end"] >= since and (not level or i["level"] == level)]
        from src.soc import cases

        def case(iid: str) -> dict[str, Any]:  # analysts' note text stays in the SOC (pentest
            c = cases.get(iid)  # re-check 2026-10-23: API tokens are for SIEM integrations)
            return {k: c.get(k) for k in ("status", "assignee", "updated")} | {"notes": len(c.get("notes", []))}

        out_rows = [r | {"case": case(r["id"])} for r in _clean(rows[:limit])]
        return {"incidents": out_rows, "count": len(rows)}
    if path == "/api/v1/alerts":
        rows = [a for a in engine.read_jsonl(engine.ALERTS, last=50_000) if float(a.get("ts") or 0) >= since]
        return {"alerts": _clean(rows[-limit:]), "count": len(rows)}
    if path == "/api/v1/search":
        from src.soc import search

        text = q.get("q", [""])[0][:500]
        if not text.strip():
            raise ValueError("q is required")
        res = search.run_store(text)
        return {"understood": res.get("understood"), "count": res.get("count"),
                "rows": _clean(list(res.get("rows", []))[:limit])}  # fmt: skip
    raise KeyError(path)


def make_handler(guard: _Guard, lan_only: bool) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SplitShield"
        sys_version = ""
        protocol_version = "HTTP/1.1"
        timeout = CONN_TIMEOUT_S  # slow / idle clients are dropped

        def setup(self) -> None:
            self.request.do_handshake()  # in this connection's thread, under the timeout
            super().setup()

        def log_message(self, *_a: Any) -> None:  # no request lines on the console
            pass

        def _send(self, code: int, obj: Any) -> None:
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Strict-Transport-Security", "max-age=31536000")
            self.end_headers()
            self.wfile.write(body)

        def _ip(self) -> str:
            return str(self.client_address[0])

        def _gate(self) -> bool:
            ip = self._ip()
            if not _allowed_ip(ip, lan_only):
                self._send(403, {"error": "only machines on the local network may connect"})
                return False
            if guard.is_blocked(ip):
                self._send(429, {"error": "too many failed attempts - try again later"})
                return False
            return True

        def _bearer(self) -> str:
            h = self.headers.get("Authorization", "")
            return h[7:].strip() if h.lower().startswith("bearer ") else ""

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if not self._gate():
                return
            u = urlparse(self.path)
            if u.path == "/agent/ping":
                name = _check("agents", self._bearer(), self._ip())
                if not name:
                    guard.fail(self._ip())
                    self._send(401, {"error": "unknown or disabled agent key"})
                    return
                self._send(200, {"ok": True, "agent": name})
                return
            name = _check("tokens", self._bearer(), self._ip(), add=1)
            if not name:
                guard.fail(self._ip())
                self._send(401, {"error": "unknown or disabled API token"})
                return
            try:
                self._send(200, api(u.path, parse_qs(u.query)))
            except KeyError:
                self._send(404, {"error": "no such endpoint"})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception:  # noqa: BLE001 - never a traceback to the network
                self._send(500, {"error": "internal error"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._gate():
                return
            if urlparse(self.path).path != "/ingest":
                self._send(404, {"error": "no such endpoint"})
                return
            name = _check("agents", self._bearer(), self._ip())
            if not name:
                guard.fail(self._ip())
                self._send(401, {"error": "unknown or disabled agent key"})
                return
            try:
                n = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                self._send(413, {"error": f"body must be 0-{MAX_BODY} bytes with Content-Length"})
                return
            if guard.over_budget(name, n):
                self._send(429, {"error": "rate limit - slow down and retry"})
                return
            body = self.rfile.read(n)
            try:
                if self.headers.get("Content-Encoding", "").lower() == "gzip":
                    body = _inflate(body)
                doc = json.loads(body.decode("utf-8"))
                if not isinstance(doc, dict):
                    raise ValueError("body must be a JSON object")
                res = ingest_batch(name, doc)
            except (ValueError, zlib.error, UnicodeDecodeError, gzip.BadGzipFile) as exc:
                self._send(400, {"error": str(exc)[:200]})
                return
            except Exception:  # noqa: BLE001 - the agent retries; no traceback leaks
                self._send(500, {"error": "internal error"})
                return
            from src.soc import heartbeat

            sk = heartbeat.skew(doc.get("sent"))
            _check("agents", self._bearer(), self._ip(),
                   add=len(doc.get("events") or doc.get("lines") or []),
                   extra={"skew_s": sk} if sk is not None else None)  # fmt: skip
            self._send(200, {"ok": True, **res})

    return Handler


def settings() -> dict[str, Any]:
    from src.soc import sinks

    return dict(sinks.load_settings().get("collector") or {})


class _TLSServer(ThreadingHTTPServer):
    """HTTPS with the TLS handshake in each connection's own thread (pentest
    2026-10-17: in the accept loop, one client that never handshook stopped every
    other agent) and a time limit on every connection (slow clients)."""

    daemon_threads = True

    def __init__(self, addr: tuple[str, int], handler: Any, ctx: ssl.SSLContext) -> None:
        self.ctx = ctx
        super().__init__(addr, handler)

    def get_request(self) -> tuple[Any, Any]:
        sock, addr = self.socket.accept()
        sock.settimeout(CONN_TIMEOUT_S)
        return self.ctx.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), addr

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # failed handshakes / dropped clients are normal on a network port


def make_server(port: int | None = None, bind: str | None = None) -> ThreadingHTTPServer:
    cfg = settings()
    cert, key = ensure_cert()
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert), str(key))
    guard = _Guard(float(cfg.get("max_mb_per_min", 200)))
    handler = make_handler(guard, bool(cfg.get("lan_only", True)))
    return _TLSServer((bind or cfg.get("bind", "0.0.0.0"), int(port or cfg.get("port", DEFAULT_PORT))),
                      handler, ctx)  # fmt: skip


def main() -> int:
    from utils.logging_config import configure_logging

    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--bind", default=None)
    for c in ("new-agent", "new-token"):
        sub.add_parser(c).add_argument("name")
    sub.add_parser("fingerprint")
    a = ap.parse_args()
    if a.cmd == "serve":
        # load the models BEFORE listening (2026-10-24, Linux host test: the first batch after
        # a start loaded them inside the request; agents timed out and had to resend)
        _detector()
        srv = make_server(a.port, a.bind)
        host = srv.server_address[0]
        host = host.decode() if isinstance(host, bytes) else str(host)
        print(f"SplitShield-XAI log receiver on https://{host}:{srv.server_address[1]}"
              f"  (certificate SHA-256 {fingerprint()})", flush=True)  # fmt: skip
        def watch() -> None:  # silent agents raise an alert (src/soc/heartbeat.py)
            from src.soc import heartbeat

            while True:
                time.sleep(60)
                try:
                    heartbeat.check()
                except Exception:  # noqa: BLE001, S110 - the receiver keeps running
                    pass

        threading.Thread(target=watch, daemon=True).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    if a.cmd == "fingerprint":
        print(fingerprint())
        return 0
    make = new_agent if a.cmd == "new-agent" else new_token
    print(make("local-terminal", a.name))
    print("Shown once - store it now. Certificate fingerprint:", fingerprint())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
