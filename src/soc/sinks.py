"""Send alerts/incidents to a SOC: JSON-lines file, Syslog, webhook.

Formats (owner 2026-10-16, enterprise): CEF (ArcSight; Sentinel, Wazuh, Splunk),
LEEF 2.0 (QRadar), ECS JSON (Elastic), OCSF Detection Finding JSON (Amazon Security
Lake, Google SecOps and other OCSF lakes), or the raw alert JSON. Syslog runs over
UDP, TCP or TLS (RFC 5425, port 6514; server certificate checked against the system
store or your CA file; optional client certificate). Network sinks are guaranteed:
alerts that cannot be delivered are queued and retried (src/soc/delivery.py).

Settings live in ``artifacts/soc/settings.json`` (git-ignored; it may hold a token)
and are edited from the dashboard's "Connect SOC" page, where each SIEM has a preset
so a non-coder only types a host name. A token can also come from the environment
variable ``SOC_TOKEN`` instead of the file.

Presets (what each popular SIEM listens on):
  * Wazuh   - agent tails the JSON-lines file (easiest), or manager syslog UDP 514
  * Splunk  - HTTP Event Collector (HEC) on :8088 with a token
  * Elastic - Filebeat tails the JSON-lines file, or Logstash http input
  * QRadar  - syslog CEF over TCP 514
  * Generic - any syslog server / webhook
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT

SOC_DIR = PROJECT_ROOT / "artifacts" / "soc"
SETTINGS = SOC_DIR / "settings.json"
OUTBOX = SOC_DIR / "outbox_alerts.jsonl"

DEFAULTS: dict[str, Any] = {
    "preset": "Generic",
    "jsonl": {"enabled": True, "path": str(OUTBOX), "format": "raw"},
    "syslog": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 514,
        "protocol": "tcp",  # tcp | tls | udp - acknowledged TCP is the default (2026-10-25); UDP is best effort
        "format": "cef",  # cef | leef | ecs | ocsf
        "header": "rfc3164",  # rfc3164 | rfc5424
        "verify_tls": True,
        "ca_file": "",  # empty: the Windows / system certificate store
        "client_cert": "",  # optional mutual TLS
        "client_key": "",
    },
    "webhook": {
        "enabled": False,
        "url": "",
        "header_name": "",
        "token": "",
        "format": "json",
        # Splunk's HEC port ships a self-signed certificate; trusting it is an
        # explicit opt-in (dashboard checkbox), never the default
        "verify_tls": True,
    },
    "inbox": {"enabled": True, "path": str(PROJECT_ROOT / "data" / "inbox")},
    "listener": {"enabled": False, "port": 5514},
    "auto_block": False,
    # guaranteed delivery: retry for this long, then the dead-letter file
    "delivery": {"max_age_days": 7, "max_queue": 1_000_000},
    "forward_audit": False,  # also send audit-log entries (who changed what) to the SIEM
    # privacy (owner 2026-10-22 gap audit): replace user / host names in everything sent
    # out by a stable keyed hash (the SIEM can still correlate; off by default)
    "pseudonymize": {"enabled": False, "fields": ["user", "host"]},
    # central log collection + read-only REST API (src/soc/collector.py)
    "collector": {"enabled": False, "port": 8600, "bind": "0.0.0.0", "lan_only": True,
                  "max_mb_per_min": 200, "silence_minutes": 30},  # fmt: skip
    # live delay (owner 2026-10-22 "reduce latency for live ones"): how often this
    # computer's logs are read and incidents are pushed to the SIEM (was 10 s / 5 s)
    "live": {"poll_s": 2.0, "push_s": 1.0},
    # OWASP Core Rule Set on web access logs (paranoia level 1, CRS default threshold 5)
    "crs": {"enabled": True},
}
SYSLOG_FORMATS = {"cef": "CEF (ArcSight, Sentinel, Wazuh)", "leef": "LEEF 2.0 (QRadar)",
                  "ecs": "ECS JSON (Elastic)", "ocsf": "OCSF JSON (Security Lake, SecOps)"}  # fmt: skip
JSON_FORMATS = {"raw": "SplitShield alert JSON", "ecs": "ECS (Elastic Common Schema)",
                "ocsf": "OCSF Detection Finding"}  # fmt: skip

PRESETS: dict[str, dict[str, Any]] = {
    "Wazuh": {
        "syslog": {"enabled": True, "port": 514, "protocol": "tcp"},
        "help": "Easiest: on this PC install the Wazuh agent and add to its ossec.conf\n"
        "<localfile>\n  <log_format>json</log_format>\n  <location>{outbox}</location>\n</localfile>\n"
        "Or send syslog: on the Wazuh manager add\n<remote>\n  <connection>syslog</connection>\n"
        "  <port>514</port>\n  <protocol>tcp</protocol>\n  <allowed-ips>THIS-PC-IP/32</allowed-ips>\n</remote>\n"
        "then type the manager's IP in Host below.",
    },
    "Splunk": {
        "webhook": {
            "enabled": True,
            "header_name": "Authorization",
            "format": "splunk_hec",
        },
        "help": "In Splunk: Settings -> Data inputs -> HTTP Event Collector -> New Token. "
        "Copy the token into Token below and set URL to\nhttps://SPLUNK-HOST:8088/services/collector/event",
    },
    "Elastic": {
        "jsonl": {"format": "ecs"},
        "webhook": {"format": "ecs"},
        "help": "Easiest: install Filebeat and add an input that reads {outbox} (json lines, "
        "already in ECS). Or enable Webhook with a Logstash http input URL, e.g. "
        "http://LOGSTASH-HOST:8080",
    },
    "QRadar": {
        "syslog": {"enabled": True, "port": 514, "protocol": "tcp", "format": "leef"},
        "help": "In QRadar add a Log Source of type 'Universal LEEF' (syslog; LEEF 2.0, "
        "delimiter ^). Type the QRadar event collector IP in Host below. For TLS pick "
        "protocol tls and port 6514 and add a 'TLS Syslog' protocol log source.",
    },
    "Sentinel": {
        "syslog": {"enabled": True, "port": 514, "protocol": "tcp", "format": "cef"},
        "help": "Microsoft Sentinel: on a Linux log forwarder install the Azure Monitor "
        "Agent with the 'Common Event Format (CEF) via AMA' data connector, then type the "
        "forwarder's IP in Host below. Alerts arrive in the CommonSecurityLog table.",
    },
    "Security Lake (OCSF)": {
        "jsonl": {"format": "ocsf"},
        "help": "Writes OCSF 1.3 Detection Findings (class 2004) to {outbox}; ship that "
        "file with your collector (e.g. Fluent Bit / Cribl) to Amazon Security Lake, "
        "Google SecOps or any OCSF data lake. Or point the Webhook at an OCSF HTTP "
        "endpoint with format ocsf.",
    },
    "Generic": {"help": "Any syslog server (Host/Port; TLS on 6514 recommended) and/or "
                "any HTTPS webhook URL."},
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            _merge(base[k], v)
            if isinstance(v, dict) and isinstance(base.get(k), dict)
            else v
        )
    return out


_DEFAULT_OUTBOX = str(OUTBOX)


def load_settings() -> dict[str, Any]:
    if SETTINGS.is_file():
        s = _merge(DEFAULTS, json.loads(SETTINGS.read_text("utf-8")))
    else:
        s = json.loads(json.dumps(DEFAULTS))
    # the default outbox follows OUTBOX (2026-10-17: it was fixed at import time, so
    # tests and sandboxes that move OUTBOX still wrote alerts into the real outbox)
    if s["jsonl"].get("path") in (_DEFAULT_OUTBOX, None, ""):
        s["jsonl"]["path"] = str(OUTBOX)
    return s


def live_timing(key: str) -> float:
    """A live-loop interval in seconds from the settings ("poll_s", "push_s"), >= 0.5."""
    try:
        return max(0.5, float(load_settings()["live"][key]))
    except (KeyError, TypeError, ValueError, OSError):
        return float(DEFAULTS["live"][key])


def save_settings(s: dict[str, Any]) -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)  # also on a fresh install
    SETTINGS.write_text(json.dumps(s, indent=2), "utf-8")


def apply_preset(s: dict[str, Any], name: str) -> dict[str, Any]:
    p = {k: v for k, v in PRESETS[name].items() if k != "help"}
    return _merge(s, p) | {"preset": name}


def preset_help(name: str) -> str:
    return PRESETS[name]["help"].replace("{outbox}", str(OUTBOX))


# --- formats -----------------------------------------------------------------


def _cef_escape(v: Any, header: bool = False) -> str:
    # CR/LF would split one alert into several records on a TCP syslog / SIEM feed
    s = str(v).replace("\\", "\\\\").replace("\r", " ").replace("\n", " ")
    return s.replace("|", "\\|") if header else s.replace("=", "\\=")


def to_cef(alert: dict[str, Any]) -> str:
    """ArcSight CEF:0 — understood by Wazuh, Splunk, QRadar, Elastic, Sentinel."""
    techs = ",".join(
        t["id"] if isinstance(t, dict) else t for t in alert.get("techniques", [])
    )
    ext = {
        "rt": int(alert.get("ts", time.time()) * 1000),
        "src": alert.get("src_ip") or "",
        "dst": alert.get("dst_ip") or "",
        "act": alert.get("action", "alert"),
        "cs1Label": "mitre",
        "cs1": techs,
        "cs2Label": "confidence",
        "cs2": alert.get("confidence", ""),
        "msg": alert.get("reason", alert.get("detail", "")),
    }
    tail = " ".join(f"{k}={_cef_escape(v)}" for k, v in ext.items() if v != "")
    return (
        f"CEF:0|SplitShield|SplitShield-XAI|1.0|{_cef_escape(alert.get('kind', 'alert'), True)}|"
        f"{_cef_escape(alert.get('title', 'attack detected'), True)}|{int(alert.get('severity', 5))}|{tail}"
    )


def _techs(alert: dict[str, Any]) -> list[dict[str, str]]:
    """ATT&CK techniques of an alert as {id, name, tactic}."""
    from src.soc.correlate import technique_info

    out = []
    for t in alert.get("techniques", []) or []:
        tid = str(t["id"] if isinstance(t, dict) else t)
        name, tactic = technique_info(tid)
        out.append({"id": tid, "name": str(name), "tactic": str(tactic)})
    return out


def _leef_escape(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace("^", "\\^").replace("\r", " ").replace("\n", " ")


def to_leef(alert: dict[str, Any]) -> str:
    """IBM QRadar LEEF 2.0 with ^ as the attribute delimiter."""
    techs = _techs(alert)
    attrs = {
        "devTime": int(float(alert.get("ts", time.time())) * 1000),
        "devTimeFormat": "epoch",  # milliseconds since 1970
        "cat": alert.get("kind", "alert"),
        "sev": max(1, min(int(alert.get("severity", 5)), 10)),
        "src": alert.get("src_ip") or "",
        "dst": alert.get("dst_ip") or "",
        "dstPort": alert.get("dst_port") or "",
        "usrName": alert.get("user") or "",
        "identHostName": alert.get("host") or "",
        "mitreTechnique": ",".join(t["id"] for t in techs),
        "mitreTactic": ",".join(dict.fromkeys(t["tactic"] for t in techs)),
        "confidence": alert.get("confidence", ""),
        "msg": alert.get("reason", alert.get("detail", "")),
    }
    tail = "^".join(f"{k}={_leef_escape(v)}" for k, v in attrs.items() if v != "")
    event_id = _leef_escape(alert.get("kind", "alert")).replace("|", "/")
    return f"LEEF:2.0|SplitShield|SplitShield-XAI|1.0|{event_id}|^|{tail}"


def _iso(ts: Any) -> str:
    t = float(ts if ts is not None else time.time())
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def to_ecs(alert: dict[str, Any]) -> dict[str, Any]:
    """Elastic Common Schema 8.x document."""
    techs = _techs(alert)
    sev = int(alert.get("severity", 5))
    doc: dict[str, Any] = {
        "@timestamp": _iso(alert.get("ts")),
        "message": alert.get("reason", alert.get("detail", "")) or alert.get("title", ""),
        "event": {"kind": "alert", "module": "splitshield", "dataset": "splitshield.alert",
                  "severity": sev, "risk_score": sev * 10.0, "action": alert.get("kind", "alert"),
                  "category": ["intrusion_detection"], "type": ["info"],
                  "id": str(alert.get("id", ""))},
        "rule": {"name": alert.get("title", alert.get("kind", "alert"))},
        "observer": {"vendor": "SplitShield", "product": "SplitShield-XAI", "type": "ids",
                     "hostname": socket.gethostname()},
        "splitshield": {k: alert[k] for k in ("source", "label", "confidence", "allowlisted")
                        if k in alert},
    }  # fmt: skip
    if alert.get("src_ip"):
        doc["source"] = {"ip": alert["src_ip"]}
    if alert.get("dst_ip") or alert.get("dst_port"):
        doc["destination"] = {k: v for k, v in (("ip", alert.get("dst_ip")),
                                                ("port", alert.get("dst_port"))) if v}  # fmt: skip
    if alert.get("user"):
        doc["user"] = {"name": alert["user"]}
    if alert.get("host"):
        doc["host"] = {"name": alert["host"]}
    if techs:
        doc["threat"] = {"framework": "MITRE ATT&CK",
                         "technique": {"id": [t["id"] for t in techs], "name": [t["name"] for t in techs]},
                         "tactic": {"name": list(dict.fromkeys(t["tactic"] for t in techs))}}  # fmt: skip
    return doc


def to_ocsf(alert: dict[str, Any]) -> dict[str, Any]:
    """OCSF 1.3 Detection Finding (class 2004, category Findings)."""
    techs = _techs(alert)
    sev = int(alert.get("severity", 5))
    sev_id = 5 if sev >= 9 else 4 if sev >= 7 else 3 if sev >= 5 else 2 if sev >= 3 else 1
    doc: dict[str, Any] = {
        "class_uid": 2004, "class_name": "Detection Finding", "category_uid": 2,
        "category_name": "Findings", "activity_id": 1, "activity_name": "Create",
        "type_uid": 200401, "time": int(float(alert.get("ts", time.time())) * 1000),
        "severity_id": sev_id,
        "severity": {1: "Informational", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}[sev_id],
        "status_id": 1, "status": "New",
        "message": alert.get("reason", alert.get("detail", "")) or alert.get("title", ""),
        "metadata": {"version": "1.3.0", "product": {"name": "SplitShield-XAI",
                                                     "vendor_name": "SplitShield", "version": "1.0"}},
        "finding_info": {"uid": str(alert.get("id", "")), "title": alert.get("title", alert.get("kind", "alert")),
                         "types": [str(alert.get("kind", "alert"))],
                         "analytic": {"name": str(alert.get("source", "splitshield")), "type_id": 1,
                                      "type": "Rule"},
                         "attacks": [{"technique": {"uid": t["id"], "name": t["name"]},
                                      "tactic": {"name": t["tactic"]}, "version": "17"} for t in techs]},
        "unmapped": {k: alert[k] for k in ("label", "allowlisted") if k in alert},
    }  # fmt: skip
    if alert.get("confidence") not in (None, ""):
        doc["confidence_score"] = int(round(float(alert["confidence"]) * 100))
    if alert.get("src_ip"):
        doc["src_endpoint"] = {"ip": alert["src_ip"]}
    if alert.get("dst_ip") or alert.get("dst_port"):
        doc["dst_endpoint"] = {k: v for k, v in (("ip", alert.get("dst_ip")),
                                                 ("port", alert.get("dst_port"))) if v}  # fmt: skip
    if alert.get("user"):
        doc["actor"] = {"user": {"name": alert["user"]}}
    if alert.get("host"):
        doc["device"] = {"hostname": alert["host"], "type_id": 0}
    return doc


def format_json(alert: dict[str, Any], fmt: str) -> dict[str, Any]:
    if fmt == "ecs":
        return to_ecs(alert)
    if fmt == "ocsf":
        return to_ocsf(alert)
    return alert


def format_line(alert: dict[str, Any], fmt: str) -> str:
    """One alert as one text line in a syslog format."""
    if fmt == "leef":
        return to_leef(alert)
    if fmt in ("ecs", "ocsf"):
        return json.dumps(format_json(alert, fmt), default=str, separators=(",", ":"))
    return to_cef(alert)


def _syslog_line(alert: dict[str, Any], cfg: dict[str, Any] | None = None) -> bytes:
    cfg = cfg or {}
    pri = 8 * 4 + (2 if alert.get("severity", 5) >= 8 else 4)  # facility auth, crit/warning
    body = format_line(alert, str(cfg.get("format", "cef")))
    host = socket.gethostname()
    if cfg.get("header") == "rfc5424":
        return f"<{pri}>1 {_iso(alert.get('ts'))} {host} SplitShield - - - {body}".encode()
    t = time.localtime(alert.get("ts", time.time()))
    # RFC 3164 TIMESTAMP: day space-padded ("Oct  3"), not zero-padded
    stamp = f"{time.strftime('%b', t)} {t.tm_mday:2d} {time.strftime('%H:%M:%S', t)}"
    return f"<{pri}>{stamp} {host} SplitShield: {body}".encode()


# --- delivery ------------------------------------------------------------------


def network_senders(s: dict[str, Any]) -> dict[str, Any]:
    """Enabled network sinks -> function(alert, settings) -> 'ok' / 'error: ...'."""
    out: dict[str, Any] = {}
    if s["syslog"]["enabled"]:
        out["syslog"] = lambda a, st: _send_syslog(_syslog_line(a, st["syslog"]), st["syslog"])
    if s["webhook"]["enabled"] and s["webhook"]["url"]:
        out["webhook"] = lambda a, st: _send_webhook(a, st["webhook"])
    return out


def _pseudo_key() -> bytes:
    """Per-installation secret (made once, kept next to the settings)."""
    import secrets

    p = SETTINGS.with_name("pseudonym.key")
    try:
        return p.read_bytes()
    except OSError:
        p.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(32)
        p.write_bytes(key)
        return key


def pseudonymize(alert: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    """Alert with user / host names replaced by 'u-<12 hex>' / 'h-<12 hex>' when the
    owner switched it on; the same name always gets the same pseudonym."""
    cfg = s.get("pseudonymize") or {}
    if not cfg.get("enabled"):
        return alert
    import hashlib
    import hmac

    key = _pseudo_key()
    out = dict(alert)
    for f in cfg.get("fields", ["user", "host"]):
        v = out.get(f)
        if v:
            out[f] = f"{f[0]}-" + hmac.new(key, str(v).encode(), hashlib.sha256).hexdigest()[:12]
    return out


def send(alert: dict[str, Any], s: dict[str, Any] | None = None) -> dict[str, str]:
    """Deliver one alert to every enabled sink. Returns sink -> 'ok' / 'queued ...' /
    error text. Network sinks that fail are queued and retried (guaranteed delivery).
    Delivery problems never raise: a SIEM outage must not stop detection."""
    from src.soc import delivery

    s = s or load_settings()
    alert = pseudonymize(alert, s)
    res: dict[str, str] = {}
    if s["jsonl"]["enabled"]:
        try:
            p = Path(s["jsonl"]["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            doc = format_json(alert, str(s["jsonl"].get("format", "raw")))
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(doc, default=str) + "\n")
            res["jsonl"] = "ok"
        except OSError as exc:
            res["jsonl"] = f"error: {exc}"
    senders = network_senders(s)
    if senders:
        res |= delivery.deliver(alert, s, senders)
    return res


def _tls_context(cfg: dict[str, Any]) -> Any:
    import ssl

    ctx = ssl.create_default_context(cafile=cfg.get("ca_file") or None)
    if not cfg.get("verify_tls", True):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # owner opted in on the dashboard
    if cfg.get("client_cert"):
        ctx.load_cert_chain(cfg["client_cert"], cfg.get("client_key") or None)
    return ctx


_CONN: dict[tuple[str, str, int, str], Any] = {}  # one open TCP / TLS connection per receiver
_CONN_LOCK = threading.Lock()


def _tls_open(cfg: dict[str, Any]) -> Any:
    raw = socket.create_connection((cfg["host"], int(cfg["port"])), timeout=5)
    try:
        sk = _tls_context(cfg).wrap_socket(raw, server_hostname=cfg["host"])
    except (OSError, ValueError):
        raw.close()
        raise
    # TLS 1.3 servers send session tickets after the handshake: read them now, or
    # closing later with unread data resets the connection and can lose messages
    sk.settimeout(0.2)
    try:
        sk.recv(1)
    except OSError:  # timeout / SSL errors are OSError subclasses
        pass
    sk.settimeout(5)
    return sk


def _closed_by_peer(sk: Any) -> bool:
    """A syslog receiver never sends data: a readable socket means it closed (or
    reset) the connection. Writing to it could 'succeed' and lose the message."""
    import select

    try:
        readable, _, _ = select.select([sk], [], [], 0)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        sk.setblocking(False)
        return sk.recv(1) == b""
    except (BlockingIOError, ssl.SSLWantReadError):
        return False  # TLS housekeeping only, no application data
    except OSError:
        return True
    finally:
        try:
            sk.setblocking(True)
            sk.settimeout(5)
        except OSError:
            pass


def _send_kept(frame: bytes, cfg: dict[str, Any], tls: bool) -> None:
    """Send over a kept-open connection (one connect / handshake, not one per alert);
    reconnect when the receiver closed it, and once more on a send error."""
    key = ("tls" if tls else "tcp", str(cfg["host"]), int(cfg["port"]), str(cfg.get("ca_file", "")))
    with _CONN_LOCK:
        for attempt in (1, 2):
            sk = _CONN.get(key)
            try:
                if sk is not None and _closed_by_peer(sk):
                    _CONN.pop(key, None)
                    sk.close()
                    sk = None
                if sk is None:
                    sk = _CONN[key] = _tls_open(cfg) if tls else socket.create_connection(
                        (cfg["host"], int(cfg["port"])), timeout=5)  # fmt: skip
                sk.sendall(frame)
                return
            except (OSError, ValueError):
                _CONN.pop(key, None)
                if sk is not None:
                    sk.close()
                if attempt == 2:
                    raise


def _send_syslog(data: bytes, cfg: dict[str, Any]) -> str:
    try:
        proto = cfg.get("protocol", "tcp")
        if proto == "tls":
            _send_kept(str(len(data)).encode() + b" " + data, cfg, tls=True)  # RFC 5425 octet counting
        elif proto == "tcp":
            _send_kept(data + b"\n", cfg, tls=False)  # RFC 6587 newline framing
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
                sk.sendto(data, (cfg["host"], int(cfg["port"])))
        return "ok"
    except (OSError, ValueError) as exc:
        return f"error: {exc}"


def _send_webhook(alert: dict[str, Any], cfg: dict[str, Any]) -> str:
    url = cfg["url"]
    if not url.lower().startswith(("http://", "https://")):
        return "error: URL must start with http:// or https://"
    body = (
        {
            "time": float(alert.get("ts", time.time())),  # event time, not arrival
            "host": socket.gethostname(),
            "event": alert,
            "sourcetype": "splitshield:alert",
            "source": "splitshield-xai",
        }
        if cfg.get("format") == "splunk_hec"
        else format_json(alert, str(cfg.get("format", "json")))
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(body, default=str).encode(),
        headers={"Content-Type": "application/json"},
    )
    token = os.environ.get("SOC_TOKEN") or cfg.get("token", "")
    if token and cfg.get("header_name"):
        prefix = "Splunk " if cfg.get("format") == "splunk_hec" else ""
        req.add_header(cfg["header_name"], prefix + token)
    ctx = None
    if url.lower().startswith("https://") and not cfg.get("verify_tls", True):
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # owner opted in: self-signed SIEM cert
    try:
        with urllib.request.urlopen(req, timeout=8, context=ctx) as r:  # noqa: S310 - scheme checked above
            return "ok" if 200 <= r.status < 300 else f"error: HTTP {r.status}"
    except (urllib.error.URLError, OSError) as exc:
        return f"error: {exc}"


def test_connection(s: dict[str, Any]) -> dict[str, str]:
    """Send one clearly-labelled test alert through every enabled sink."""
    return send(
        {
            "ts": time.time(),
            "kind": "test",
            "title": "SplitShield-XAI connection test",
            "severity": 1,
            "reason": "If you can see this in your SIEM, the connection works.",
            "techniques": [],
        },
        s,
    )
