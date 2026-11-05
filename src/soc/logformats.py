"""Parsers for non-Windows logs (owner 2026-10-15: "all kinds of logs").

Each parser turns a record into the normal event shape (see ``parsers._event``) and
runs the Sigma community rules for its log source (``sigma.match_source``) plus our
own rules (``logrules``). Supported:

  * web server access logs (Apache / Nginx common + combined format)
  * Linux auditd (``type=... msg=audit(...)``), incl. EXECVE -> process creation
  * Zeek TSV logs of any kind (conn, dns, http, ssl, ...) and Zeek JSON
  * AWS CloudTrail (``{"Records": [...]}`` files and single records)
  * Kubernetes audit events, Azure activity / sign-in logs, GCP audit logs,
    Microsoft 365 audit, Okta, GitHub audit (JSON)
  * Sysmon for Linux and Windows events as XML (Splunk XmlWinEventLog export)
  * osquery results (Linux / macOS process events)
  * anything else structured (JSON, key=value): parsed generically and checked by
    the unknown-format detector (``unknown.py``)
"""

from __future__ import annotations

import binascii
import json
import re
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from src.soc import sigma

Event = dict[str, Any]

_ACCESS = re.compile(
    r'^(?P<ip>\S+) \S+ (?P<user>\S+) \[(?P<ts>[^\]]+)\] "(?P<method>[A-Z]{3,10}) '
    r'(?P<target>\S+)(?: (?P<proto>[^"]*))?" (?P<status>\d{3}) (?P<bytes>\S+)'
    r'(?: "(?P<ref>[^"]*)" "(?P<ua>[^"]*)")?'
)
# raw (``audit(1700000000.123:42):``) and interpreted ``ausearch -i`` records
# (``audit(04/16/2025 08:20:23.552:47372) :`` with names instead of numbers)
_AUDIT = re.compile(
    r"^(?:node=\S+ )?type=(?P<type>\S+) msg=audit\((?P<ts>[\d.]+|\d\d/\d\d/\d{4} [\d:.]+)"
    r":(?P<id>\d+)\)\s?:\s?(?P<rest>.*)$"
)
_PROCTITLE = re.compile(r"\bproctitle=(.*)$")
_KV = re.compile(r"""(\w[\w.-]*)=("[^"]*"|'[^']*'|\S+)""")


def _ev(**kw: Any) -> Event:
    from src.soc.parsers import _event

    return _event(**kw)


def _ip(v: Any) -> str | None:
    from src.soc.parsers import _ip as ip

    return ip(v)


def _ts(v: Any) -> float:
    from src.soc.parsers import _ts as ts

    return ts(v)


def finish(
    ev: Event,
    fields: dict[str, Any],
    source: tuple[str | None, str | None, str | None],
    techs: list[str] | None = None,
) -> Event:
    """Run Sigma + our own rules for this log source and attach the techniques."""
    from src.soc import logrules, rulectl

    techs = list(techs or [])
    rulectl.hit(ev, f"own:{ev['kind']}", techs, str(ev["kind"]))
    own = logrules.check(fields, source)
    if own:
        ev["kind"] = own[0]
        techs += [t for t in own[1] if t not in techs]
        rulectl.hit(ev, f"own:{own[0]}", list(own[1]), own[0])
    hits = sigma.match_source(fields, *source)
    for r in hits:
        techs += [t for t in r.techniques if t not in techs]
        rulectl.hit(ev, f"sigma:{r.rule_id}", list(r.techniques), r.title)
    if source[0] in ("linux", "macos") and fields.get("CommandLine"):  # GTFOBins misuse
        from src.soc import lolbins

        img, cmd = str(fields.get("Image") or ""), str(fields["CommandLine"])
        for p in lolbins.match(img, cmd, "linux"):
            techs += [p["technique"]] if p["technique"] not in techs else []
            rulectl.hit(ev, f"lolbas:{p['binary']}", [p["technique"]], f"misuse of {p['binary']}")
    if hits:
        order = {"critical": 0, "high": 1, "medium": 2}
        top = sorted(hits, key=lambda r: order.get(r.level, 3))[0]
        ev["detail"] = f"{ev['detail']} | Sigma: {top.title}"
        if ev["kind"] in ("log", "generic"):
            ev["kind"] = "sigma_rule"
    ev["techniques"] = techs
    return ev


# --- web server access logs ----------------------------------------------------------


def parse_access(line: str, source: str = "web") -> Event | None:
    m = _ACCESS.match(line)
    if not m:
        return None
    try:
        ts = datetime.strptime(m["ts"], "%d/%b/%Y:%H:%M:%S %z").timestamp()
    except ValueError:
        ts = _ts(None)
    return web_request(ts, m["ip"], m["user"], m["method"], m["target"], m["status"], m["proto"] or "",
                       m["ua"] or "", m["ref"] or "", source)  # fmt: skip


def web_request(ts: float, ip: str | None, user: str | None, method: str, target: str, status: str,
                proto: str = "", ua: str = "", ref: str = "", source: str = "web") -> Event:  # fmt: skip
    """One HTTP request (from any log that records it: access logs, key=value server
    logs, Suricata http events) -> event, checked by the web Sigma rules and CRS."""
    stem, _, _query = target.partition("?")
    fields = {
        "c-ip": ip or "",
        "cs-username": user or "-",
        "cs-method": method,
        "cs-uri-stem": stem,
        # rules search the whole request target (path + query) for attack strings
        "cs-uri-query": target,
        "cs-uri": target,
        "c-uri": target,
        "sc-status": status,
        "cs-referer": ref,
        "cs-user-agent": ua,
        "c-useragent": ua,
    }
    ev = _ev(
        ts=ts,
        src_ip=_ip(ip),
        source=source,
        kind="http_request",
        detail=f"{method} {target[:150]} -> {status}",
    )
    from src.soc.parsers import with_entities

    with_entities(ev, user=user if user and user != "-" else None)
    ev = finish(ev, fields, (None, "webserver", None))
    from src.soc import crs

    if crs.enabled():  # OWASP Core Rule Set (owner 2026-10-23; held-out check in docs/RESULTS.md)
        from src.soc import rulectl

        es = rulectl.engine_settings()
        r = crs.check(crs.request(method, target, proto, ua, ref), paranoia=int(es["crs_paranoia"]),
                      threshold=int(es["crs_threshold"]), disabled=rulectl.disabled_ids("crs:"))  # fmt: skip
        if r["attack"]:
            ev["techniques"] = list(dict.fromkeys([*ev["techniques"], *r["techniques"]]))
            ev["crs"] = {"score": r["score"], "rules": r["rules"][:8], "classes": r["classes"]}
            for rid in r["rules"]:
                rulectl.hit(ev, f"crs:{rid}", r["techniques"], r["msg"])
            ev["crs_tags"] = " ".join(r["classes"])
            ev["detail"] = f"{ev['detail']} | CRS {r['score']}: {r['msg']}"[:300]
            if ev["kind"] == "http_request":
                ev["kind"] = "web_attack"
    return ev


_KVQ = re.compile(r'([\w.-]+)="((?:[^"\\]|\\.)*)"')


def parse_kv_access(line: str, source: str = "web") -> Event | None:
    """Web server logs written as key="value" pairs (nginx log_format with quoted
    fields, NGINX Plus, many load balancers): needs a request (or method + uri) and a
    client address."""
    kv = dict(_KVQ.findall(line))
    if not kv:
        return None
    req = kv.get("request", "")
    if req:
        method, target, proto = ([*req.split(" "), "", "", ""])[:3]
    else:
        method = kv.get("request_method") or kv.get("method") or kv.get("http_method", "")
        target = kv.get("request_uri") or kv.get("uri", "")
        if not target and kv.get("uri_path"):  # NGINX Plus / Splunk CIM names
            q = kv.get("uri_query", "")
            target = kv["uri_path"] + (f"?{q}" if q and q != "-" else "")
        proto = kv.get("server_protocol") or kv.get("protocol", "")
    ip = kv.get("src_ip") or kv.get("remote_addr") or kv.get("client_ip") or kv.get("src")
    if not method or not target or not ip:
        return None
    t = kv.get("time_local") or kv.get("time_iso8601") or kv.get("timestamp")
    try:
        ts = datetime.strptime(t, "%d/%b/%Y:%H:%M:%S %z").timestamp() if t and "/" in t else _ts(t)
    except ValueError:
        ts = _ts(None)
    return web_request(ts, ip, kv.get("user") or kv.get("remote_user"), method, target, kv.get("status", ""),
                       proto, kv.get("http_user_agent", ""), kv.get("http_referer", ""), source)  # fmt: skip


# --- Linux auditd ------------------------------------------------------------------


def _unhex(v: str) -> str:
    v = v.strip('"')
    if re.fullmatch(r"(?:[0-9A-F]{2})+", v) and len(v) >= 4:
        try:
            return binascii.unhexlify(v).decode("utf-8", "replace").replace("\x00", " ")
        except (binascii.Error, ValueError):
            return v
    return v


def parse_auditd(line: str) -> Event | None:
    m = _AUDIT.match(line.strip())
    if not m:
        return None
    kv = {k: _unhex(v.strip("'")) for k, v in _KV.findall(m["rest"])}
    inner = kv.get("msg", "")
    if "=" in inner:  # USER_* / SERVICE_* records nest key=value pairs in msg='...'
        for k, v in _KV.findall(inner):
            kv.setdefault(k, v.strip("'\""))
    if m["type"] == "PROCTITLE" and (pt := _PROCTITLE.search(m["rest"])):
        # interpreted records print the whole command line, spaces included
        kv["proctitle"] = _unhex(pt.group(1).strip())
    fields: dict[str, Any] = {"type": m["type"], **kv}
    source: tuple[str | None, str | None, str | None] = ("linux", None, "auditd")
    detail = f"auditd {m['type']}"
    if m["type"] == "PROCTITLE" and kv.get("proctitle"):
        cmd = kv["proctitle"]
        fields.update({"CommandLine": cmd, "Image": cmd.split(" ", 1)[0]})
        detail = f"proctitle: {cmd[:160]}"
    if m["type"] == "EXECVE":
        args = [
            kv[f"a{i}"] for i in range(int(kv.get("argc", "0") or 0)) if f"a{i}" in kv
        ]
        cmd = " ".join(args)
        fields.update({"CommandLine": cmd, "Image": args[0] if args else ""})
        detail = f"exec: {cmd[:160]}"
    elif "exe" in kv:
        detail += f" {kv.get('exe', '')} {kv.get('comm', '')}".rstrip()
    ts = m["ts"]
    try:
        when = (datetime.strptime(ts, "%m/%d/%Y %H:%M:%S.%f").timestamp()
                if "/" in ts else float(ts))  # fmt: skip
    except ValueError:
        when = _ts(None)
    ev = _ev(
        ts=when,
        src_ip=_ip(kv.get("addr")),
        source="auditd",
        kind=f"audit_{m['type'].lower()}",
        detail=detail[:200],
    )
    from src.soc.parsers import with_entities

    ev = finish(ev, fields, source)
    uid = (
        kv.get("auid") if kv.get("auid") not in (None, "4294967295") else kv.get("uid")
    )
    with_entities(ev, user=f"uid:{uid}" if uid else None, host=kv.get("node"))
    if fields.get("CommandLine"):  # EXECVE / PROCTITLE: Sigma's Linux process rules
        extra = [
            r.techniques
            for r in sigma.match_source(fields, "linux", "process_creation", None)
        ]
        ev["techniques"] += [
            t for ts_ in extra for t in ts_ if t not in ev["techniques"]
        ]
    return ev


# --- Zeek ---------------------------------------------------------------------------


def parse_zeek(lines: list[str]) -> Iterator[Event]:
    """Any Zeek TSV log (#path tells which); IoT-23 '.labeled' extra columns kept."""
    fields_: list[str] = []
    path = "conn"
    for raw in lines:
        if raw.startswith("#path"):
            path = raw.split("\t", 1)[-1].strip() or path
            continue
        if raw.startswith("#fields"):
            fields_ = [
                n
                for c in raw.rstrip("\n").split("\t")[1:]
                for n in re.split(r"\s{2,}", c.strip())
            ]
            if len(fields_) == 1:  # IoT-23 joins the last columns with spaces
                fields_ = raw.split()[1:]
            continue
        if not raw.strip() or raw.startswith("#") or not fields_:
            continue
        vals = raw.rstrip("\n").split("\t")
        if len(vals) < len(fields_) and "   " in vals[-1]:  # IoT-23 label columns
            vals = vals[:-1] + re.split(r"\s{2,}", vals[-1])
        row = dict(zip(fields_, vals, strict=False))
        yield zeek_event(row, path)


def zeek_path_of(row: dict[str, Any]) -> str:
    """The Zeek log a JSON record comes from when it has no ``_path`` (2026-10-23: every
    such record was read as conn.log, so DNS / HTTP / DCE-RPC fields and the Zeek Sigma
    rules for them were lost)."""
    if "endpoint" in row and "operation" in row:
        return "dce_rpc"
    if "query" in row and ("qtype_name" in row or "rcode_name" in row or "answers" in row):
        return "dns"
    if "method" in row and "uri" in row:
        return "http"
    if "server_name" in row or ("version" in row and "cipher" in row):
        return "ssl"
    if "conn_state" in row or "proto" in row:
        return "conn"
    return "conn"


def zeek_event(row: dict[str, Any], path: str) -> Event:
    port = row.get("id.resp_p")
    state = row.get("conn_state", "")
    if path == "conn":
        kind = "conn_rejected" if state in ("REJ", "S0", "RSTOS0") else "conn"
        detail = f"{row.get('proto', '')} {row.get('service', '-')} {state}"
    elif path == "dns":
        kind, detail = (
            "dns_query",
            f"DNS {row.get('qtype_name', '')} {row.get('query', '')}",
        )
    elif path == "http":
        kind = "http_request"
        detail = f"{row.get('method', '')} {row.get('host', '')}{row.get('uri', '')} -> {row.get('status_code', '')}"
    elif path == "ssl":
        kind, detail = (
            "tls",
            f"TLS {row.get('server_name', '')} {row.get('version', '')}",
        )
    elif path == "dce_rpc":  # remote procedure calls (DCSync, remote services, scheduled tasks)
        kind, detail = "zeek_dce_rpc", f"DCE-RPC {row.get('endpoint', '')} {row.get('operation', '')}"
    else:
        kind, detail = f"zeek_{path}", " ".join(str(v) for v in list(row.values())[2:8])
    ev = _ev(
        ts=_ts(row.get("ts")),
        src_ip=_ip(row.get("id.orig_h")),
        dst_ip=_ip(row.get("id.resp_h")),
        dst_port=int(str(port)) if str(port or "").isdigit() else None,
        source="zeek",
        kind=kind,
        detail=detail[:200],
    )
    fields = dict(row)
    ev = finish(ev, fields, ("zeek", None, path))
    if path == "dns":  # Sigma's generic DNS taxonomy
        dns = {"query": row.get("query", ""), "record_type": row.get("qtype_name", ""),
               "answer": row.get("answers", "")}  # fmt: skip
        ev = finish(ev, {**fields, **dns}, (None, "dns", None), ev["techniques"])
    if path == "http":  # and the proxy taxonomy
        web = {"c-uri": row.get("uri", ""), "cs-host": row.get("host", ""),
               "cs-method": row.get("method", ""), "c-useragent": row.get("user_agent", "")}  # fmt: skip
        ev = finish(ev, {**fields, **web}, (None, "proxy", None), ev["techniques"])
    return ev


# --- JSON: cloud, Kubernetes, SaaS, osquery ------------------------------------------


def ocsf_cloudtrail(obj: dict[str, Any]) -> dict[str, Any] | None:
    """An AWS Security Lake (OCSF) CloudTrail record -> the classic CloudTrail record,
    so the CloudTrail rules apply unchanged (2026-10-24, attack_data dev half)."""
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else None
    api = obj.get("api") if isinstance(obj.get("api"), dict) else None
    if meta is None or api is None or str((meta.get("product") or {}).get("name", "")).lower() != "cloudtrail":
        return None
    user = (obj.get("actor") or {}).get("user") or {}
    resp = api.get("response") or {}
    t = obj.get("time")
    when = datetime.fromtimestamp(float(t) / 1000).astimezone().isoformat() if isinstance(t, int | float) else t
    rec: dict[str, Any] = {
        "eventTime": when, "eventName": api.get("operation", ""),
        "eventSource": (api.get("service") or {}).get("name", ""),
        "sourceIPAddress": (obj.get("src_endpoint") or {}).get("ip", ""),
        "userAgent": (obj.get("http_request") or {}).get("user_agent", ""),
        "userIdentity": {"type": user.get("type", ""), "arn": user.get("uid", ""),
                         "userName": user.get("name", ""), "accountId": (user.get("account") or {}).get("uid", "")},
        "awsRegion": (obj.get("cloud") or {}).get("region", ""),
    }  # fmt: skip
    if resp.get("error"):
        rec["errorCode"], rec["errorMessage"] = resp.get("error"), resp.get("message", "")
    data = (api.get("request") or {}).get("data")
    if isinstance(data, str) and data.startswith("{"):
        try:
            rec["requestParameters"] = json.loads(data)
        except ValueError:
            pass
    return rec


def json_source(
    obj: dict[str, Any],
) -> tuple[str | None, str | None, str | None] | None:
    """Log source of a JSON record (None: not a known JSON log)."""
    if ocsf_cloudtrail(obj) is not None:
        return ("aws", None, "cloudtrail")
    if "eventSource" in obj and "eventName" in obj:
        return ("aws", None, "cloudtrail")
    if str(obj.get("apiVersion", "")).startswith("audit.k8s.io") or (
        "verb" in obj and "objectRef" in obj
    ):
        return ("kubernetes", "application", "audit")
    if "protoPayload" in obj:
        return ("gcp", None, "gcp.audit")
    if "Workload" in obj and "Operation" in obj:
        return ("m365", None, "audit")
    if "eventType" in obj and "displayMessage" in obj:
        return ("okta", None, "okta")
    if "operationName" in obj and ("resourceId" in obj or "category" in obj):
        cat = str(obj.get("category", "")).lower()
        svc = (
            "signinlogs"
            if "signin" in cat
            else "auditlogs"
            if "audit" in cat
            else "activitylogs"
        )
        return ("azure", None, svc)
    if "action" in obj and "actor" in obj and "org" in obj:
        return ("github", None, "audit")
    if (
        "columns" in obj
        and "name" in obj
        and ("hostIdentifier" in obj or "unixTime" in obj)
    ):
        return ("osquery", None, str(obj.get("name")))
    return None


def parse_json_record(obj: dict[str, Any]) -> Event | None:
    if (classic := ocsf_cloudtrail(obj)) is not None:
        obj = classic
    src = json_source(obj)
    if src is None:
        return None
    flat = sigma.flatten(obj)
    product = src[0]
    if product == "aws":
        who = flat.get("useridentity.arn") or flat.get("useridentity.type", "")
        ev = _ev(ts=_ts(obj.get("eventTime")), src_ip=_ip(obj.get("sourceIPAddress")),
                 source="cloudtrail", kind=str(obj.get("eventName")),
                 detail=f"{obj.get('eventSource')}:{obj.get('eventName')} by {who}"
                 + (f" ({obj.get('errorCode')})" if obj.get("errorCode") else ""))  # fmt: skip
    elif product == "kubernetes":
        ref = obj.get("objectRef") or {}
        user = (obj.get("user") or {}).get("username", "")
        ev = _ev(ts=_ts(obj.get("requestReceivedTimestamp") or obj.get("stageTimestamp")),
                 src_ip=_ip((obj.get("sourceIPs") or [None])[0]), source="k8s_audit",
                 kind=f"k8s_{obj.get('verb', '')}",
                 detail=f"{user} {obj.get('verb')} {ref.get('resource', '')}/{ref.get('name', '')} "
                 f"ns={ref.get('namespace', '')}")  # fmt: skip
    elif product == "osquery":
        cols = obj.get("columns") or {}
        cmd = cols.get("cmdline") or cols.get("command") or ""
        image = cols.get("path") or ""
        is_mac = (
            "/Applications/" in image or "/System/" in image or "/Library/" in image
        )
        product = "macos" if is_mac or "darwin" in json.dumps(obj).lower() else "linux"
        flat.update(
            {
                "CommandLine": cmd,
                "Image": image,
                "ParentImage": cols.get("parent_path", ""),
            }
        )
        ev = _ev(ts=_ts(obj.get("unixTime") or obj.get("calendarTime")), source="osquery",
                 kind=f"osquery_{obj.get('name')}", detail=f"{image} {cmd}"[:200])  # fmt: skip
        src = (
            (product, "process_creation", None)
            if cmd or image
            else (product, None, None)
        )
    else:
        name = (
            obj.get("operationName")
            or obj.get("Operation")
            or obj.get("eventType")
            or obj.get("action")
        )
        ev = _ev(ts=_ts(obj.get("time") or obj.get("CreationTime") or obj.get("published")
                        or obj.get("timestamp") or obj.get("@timestamp")),
                 src_ip=_ip(obj.get("callerIpAddress") or obj.get("ClientIP") or obj.get("ipAddress")),
                 source=product or "cloud", kind=str(name)[:60], detail=str(name)[:200])  # fmt: skip
    from src.soc.parsers import with_entities

    user = (
        flat.get("useridentity.arn") or flat.get("useridentity.principalid")
        or flat.get("user.username") or flat.get("userprincipalname")
        or flat.get("protopayload.authenticationinfo.principalemail") or flat.get("userid")
        or flat.get("actor.alternateid") or flat.get("actor")
    )  # fmt: skip
    with_entities(ev, user=user, host=flat.get("hostidentifier"))
    return finish(ev, flat, src)


def parse_json_text(text: str) -> list[Event] | None:
    """A JSON file of cloud / audit records: {"Records": [...]}, a list, or JSON lines.
    None when the content is not one of these log types."""
    s = text.lstrip()
    recs: list[Any] = []
    try:
        doc = json.loads(s)
        recs = doc.get("Records", [doc]) if isinstance(doc, dict) else list(doc)
    except (json.JSONDecodeError, RecursionError):
        for ln in s.splitlines():
            ln = ln.strip()
            if ln.startswith("{"):
                try:
                    recs.append(json.loads(ln))
                except (json.JSONDecodeError, RecursionError):
                    continue
    dicts = [r for r in recs if isinstance(r, dict)]
    if not dicts or json_source(dicts[0]) is None:
        return None
    return [ev for r in dicts if (ev := parse_json_record(r))]


# --- XML events (Sysmon for Linux, Windows XmlWinEventLog) ---------------------------

_XML_EVENT = re.compile(r"<Event[\s>].*?</Event>", re.S)


def parse_xml_events(text: str) -> list[Event]:
    """Windows / Sysmon-for-Linux events rendered as XML."""
    from src.soc.parsers import parse_winsec

    out = []
    for blob in _XML_EVENT.findall(text):
        eid_m = re.search(r"<EventID[^>]*>(\d+)</EventID>", blob)
        if not eid_m:
            continue
        prov = re.search(r"<Provider Name=['\"]([^'\"]+)", blob)
        chan = re.search(r"<Channel>([^<]*)</Channel>", blob)
        tc = re.search(r"SystemTime=['\"]([^'\"]+)", blob)
        comp = re.search(r"<Computer>([^<]*)</Computer>", blob)
        data = {
            k: _xml_unescape(v)
            for k, v in re.findall(
                r"<Data Name=['\"]([^'\"]+)['\"]>(.*?)</Data>", blob, re.S
            )
        }
        provider = prov.group(1) if prov else ""
        eid = int(eid_m.group(1))
        if "linux" in provider.lower():  # Sysmon for Linux
            cat = {1: "process_creation", 3: "network_connection", 11: "file_event",
                   23: "file_delete", 5: "process_termination"}.get(eid)  # fmt: skip
            ev = _ev(ts=_ts(tc.group(1) if tc else None), source="sysmon_linux",
                     kind=f"sysmon_linux_{eid}",
                     detail=f"EventID {eid}: {data.get('Image', '')} {data.get('CommandLine', '')}"[:200])  # fmt: skip
            out.append(finish(ev, data, ("linux", cat, None)))
            continue
        msg = "\n".join(f"{k}: {v}" for k, v in data.items())
        ev2 = parse_winsec(
            {
                "Id": eid,
                "ProviderName": provider,
                "Channel": chan.group(1) if chan else "",
                "TimeCreated": tc.group(1) if tc else None,
                "Message": msg,
                "EventData": data,
                "Computer": comp.group(1) if comp else None,
            }  # fmt: skip
        )
        if ev2:
            out.append(ev2)
    return out


def _xml_unescape(v: str) -> str:
    return (
        v.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        .replace("&apos;", "'").replace("&amp;", "&")
    )  # fmt: skip
