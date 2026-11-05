"""OWASP Core Rule Set (CRS) for web-server access logs (owner 2026-10-23).

The CRS (github.com/coreruleset/coreruleset, Apache-2.0) is ModSecurity's rule set for web
attacks. An access log keeps only the request line, user agent and referer, so this is a
small engine for the part of CRS that can be evaluated on them:

* rules: request phase (1 / 2), paranoia level <= ``PARANOIA`` (1 = CRS default), not
  chained, not control logic (TX), at least one variable available in an access log;
* variables: ARGS / ARGS_GET (+ _NAMES), REQUEST_FILENAME, REQUEST_BASENAME, REQUEST_URI(_RAW),
  QUERY_STRING, REQUEST_LINE, REQUEST_METHOD, REQUEST_PROTOCOL, REQUEST_HEADERS:User-Agent /
  Referer (cookies, other headers and bodies are not in access logs: empty, as on a server
  that sent none);
* operators: @rx, @pm, @pmFromFile, @contains, @beginsWith, @endsWith, @streq, @within
  (negation supported); @detectSQLi / @detectXSS (libinjection) are not available;
* transformations: the common ones (URL / HTML / JS / CSS decoding, lowercase, whitespace,
  comments, paths, cmdLine ...); a rule needing another one is skipped and counted.

Scoring is CRS anomaly scoring: CRITICAL 5, ERROR 4, WARNING 3, NOTICE 2 per matching rule;
a request is an attack at the inbound threshold 5 (CRS default). Rules are read from
``third_party/coreruleset/rules`` (kept current by the update agent).

Windows Defender flags CRS files (they contain attack strings to look for): exclude the
folder ``third_party/coreruleset`` in Windows Security to use CRS (docs/INSTALL.md).
"""

from __future__ import annotations

import base64
import html
import re
import urllib.parse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT

RULES_DIR = PROJECT_ROOT / "third_party" / "coreruleset" / "rules"
PARANOIA = 1
THRESHOLD = 5
SEVERITY = {"CRITICAL": 5, "ERROR": 4, "WARNING": 3, "NOTICE": 2}
# CRS file -> ATT&CK techniques of a match
FILE_TECHNIQUES = {"913": ["T1595"], "920": ["T1190"], "921": ["T1190"], "930": ["T1190"],
                   "931": ["T1190"], "932": ["T1190", "T1059"], "933": ["T1190"], "934": ["T1190"],
                   "941": ["T1190"], "942": ["T1190"], "943": ["T1190"], "944": ["T1190"]}  # fmt: skip
VARS = {"ARGS", "ARGS_GET", "ARGS_NAMES", "ARGS_GET_NAMES", "REQUEST_FILENAME", "REQUEST_BASENAME",
        "REQUEST_URI", "REQUEST_URI_RAW", "QUERY_STRING", "REQUEST_LINE", "REQUEST_METHOD",
        "REQUEST_PROTOCOL", "REQUEST_HEADERS"}  # fmt: skip
HEADERS = {"user-agent", "referer"}
_STMT = re.compile(r'^\s*SecRule\s+(\S+)\s+"((?:[^"\\]|\\.)*)"\s+"((?:[^"\\]|\\.)*)"', re.S)


# --- transformations -------------------------------------------------------------------


def _url_decode_uni(s: str) -> str:
    s = re.sub(r"%u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return urllib.parse.unquote_plus(s, errors="replace")


def _js_decode(s: str) -> str:
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    s = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), s)
    return re.sub(r"\\([0-7]{1,3})", lambda m: chr(int(m.group(1), 8) & 0xFF), s)


def _css_decode(s: str) -> str:
    return re.sub(r"\\([0-9a-fA-F]{1,6})\s?", lambda m: chr(min(int(m.group(1), 16), 0x10FFFF)), s)


def _cmd_line(s: str) -> str:
    s = re.sub(r"[\\\"'^]", "", s)
    s = re.sub(r"\s*([,;])\s*", r"\1", s)
    s = re.sub(r"\s+([/(])", r"\1", s)
    return re.sub(r"\s+", " ", s).lower()


def _norm_path(s: str, win: bool = False) -> str:
    if win:
        s = s.replace("\\", "/")
    parts: list[str] = []
    for p in s.split("/"):
        if p == "..":
            if parts:
                parts.pop()
        elif p != ".":
            parts.append(p)
    return "/".join(parts)


def _b64(s: str) -> str:
    try:
        return base64.b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", "replace")
    except ValueError:
        return s


TRANSFORMS: dict[str, Any] = {
    "none": lambda s: s, "urlDecodeUni": _url_decode_uni, "urlDecode": _url_decode_uni,
    "lowercase": str.lower, "htmlEntityDecode": html.unescape, "jsDecode": _js_decode,
    "cssDecode": _css_decode, "utf8toUnicode": lambda s: s, "removeNulls": lambda s: s.replace("\x00", ""),
    "removeWhitespace": lambda s: re.sub(r"\s+", "", s), "compressWhitespace": lambda s: re.sub(r"\s+", " ", s),
    "replaceComments": lambda s: re.sub(r"/\*.*?(\*/|$)", " ", s, flags=re.S),
    "removeCommentsChar": lambda s: re.sub(r"/\*|\*/|--|#", "", s), "normalizePath": _norm_path,
    "normalisePath": _norm_path, "normalizePathWin": lambda s: _norm_path(s, True),
    "normalisePathWin": lambda s: _norm_path(s, True), "cmdLine": _cmd_line,
    "escapeSeqDecode": lambda s: _js_decode(s), "base64Decode": _b64, "trim": str.strip,
    "removeComments": lambda s: re.sub(r"/\*.*?\*/|--.*$|#.*$", "", s, flags=re.S | re.M),
}  # fmt: skip


# --- rules ---------------------------------------------------------------------------


@dataclass
class Rule:
    id: str
    file: str
    variables: list[tuple[str, str | None]]  # (collection, key or None)
    negate: bool
    op: str
    arg: Any
    transforms: list[str]
    severity: int
    msg: str
    tags: list[str] = field(default_factory=list)

    def matches(self, req: dict[str, Any]) -> bool:
        # transformed values are shared by all rules of one request (2026-10-25: the same
        # decode chains were recomputed for every rule - 62 % of all log-parsing time)
        memo: dict[tuple[str, str | None, tuple[str, ...]], list[str]] = req.setdefault("_tc", {})
        tkey = tuple(self.transforms)
        for col, key in self.variables:
            mk = (col, key, tkey)
            vals = memo.get(mk)
            if vals is None:
                vals = []
                for value in _values(req, col, key):
                    v = value
                    for t in self.transforms:
                        v = TRANSFORMS[t](v)
                    vals.append(v)
                memo[mk] = vals
            for v in vals:
                hit = _op(self.op, self.arg, v)
                if hit != self.negate:
                    return True
        return False


def _op(op: str, arg: Any, v: str) -> bool:
    if op == "rx":
        return arg.search(v) is not None
    if op in ("pm", "pmFromFile"):
        return arg.search(v.lower()) is not None
    if op == "contains":
        return arg in v
    if op == "beginsWith":
        return v.startswith(arg)
    if op == "endsWith":
        return v.endswith(arg)
    if op == "streq":
        return v == arg
    if op == "within":
        return v in arg
    return False


def _values(req: dict[str, Any], col: str, key: str | None) -> list[str]:
    if col == "REQUEST_HEADERS":
        hs = req.get("headers", {})
        return [hs[key.lower()]] if key and key.lower() in hs else ([] if key else list(hs.values()))
    if col in ("ARGS", "ARGS_GET"):
        args = req.get("args", {})
        return [v for k, v in args.items() if key is None or k == key]
    if col in ("ARGS_NAMES", "ARGS_GET_NAMES"):
        return list(req.get("args", {}))
    v = req.get(col)
    return [v] if v else []


def _phrases(path: Path) -> list[str]:
    try:
        lines = path.read_text("utf-8", "replace").splitlines()
    except OSError:
        return []
    return [ln.strip().lower() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _pm(words: list[str]) -> re.Pattern[str]:
    words = sorted(set(w for w in words if w), key=len, reverse=True)
    return re.compile("|".join(re.escape(w) for w in words) or r"(?!x)x")


def _parse_vars(spec: str) -> list[tuple[str, str | None]] | None:
    out: list[tuple[str, str | None]] = []
    for part in spec.split("|"):
        if part.startswith("!"):
            continue  # exclusions (cookie names etc.) do not apply to access logs
        if part.startswith("&"):
            return None  # counting collections (e.g. 'no Host header'): not in access logs
        col, _, key = part.partition(":")
        if col == "TX" or col.startswith("TX"):
            return None
        if col not in VARS:
            continue
        if col == "REQUEST_HEADERS" and key and key.lower() not in HEADERS:
            continue
        out.append((col, key or None))
    return out or None


def _action(acts: str, name: str) -> list[str]:
    return [m.group(1).strip("'") for m in re.finditer(rf"(?:^|,)\s*{name}:('[^']*'|[^,]*)", acts)]


def parse_rules(rules_dir: Path, paranoia: int = PARANOIA) -> tuple[list[Rule], dict[str, int]]:
    """Rules usable on access logs + counts of what was skipped and why."""
    rules: list[Rule] = []
    skipped: dict[str, int] = {}

    def skip(why: str) -> None:
        skipped[why] = skipped.get(why, 0) + 1

    for f in sorted(rules_dir.glob("REQUEST-9*.conf")):
        num = f.name.split("-")[1]
        if num not in FILE_TECHNIQUES:
            continue
        text = f.read_text("utf-8", "replace").replace("\\\r\n", " ").replace("\\\n", " ")
        prev_chain = False
        for line in text.splitlines():
            m = _STMT.match(line)
            if not m:
                continue
            spec, oper, acts = m.group(1), m.group(2).replace('\\"', '"'), m.group(3)
            chained, prev_chain = prev_chain, "chain" in re.split(r"[,\s]", acts)
            if chained or prev_chain:
                skip("chained")
                continue
            pl = re.search(r"paranoia-level/(\d)", acts)
            if not pl:
                continue  # control rules (paranoia skips, initialisation)
            if int(pl.group(1)) > paranoia:
                skip(f"paranoia {pl.group(1)}")
                continue
            phase = (_action(acts, "phase") or ["2"])[0]
            if phase not in ("1", "2", "request"):
                skip("not request phase")
                continue
            variables = _parse_vars(spec)
            if variables is None:
                skip("needs headers / cookies / body / TX")
                continue
            if "%{" in oper:  # needs CRS settings (tx.allowed_methods, ...): not on access logs
                skip("needs CRS settings (macro)")
                continue
            negate = oper.startswith("!")
            oper = oper.lstrip("!")
            if oper.startswith("@"):
                name, _, arg_s = oper[1:].partition(" ")
            else:
                name, arg_s = "rx", oper
            ts = [t for t in _action(acts, "t") if t != "none"]
            if any(t not in TRANSFORMS for t in ts) or "length" in ts:
                skip("transformation not supported")
                continue
            try:
                arg: Any
                if name == "rx":
                    arg = re.compile(arg_s)
                elif name == "pm":
                    arg = _pm(arg_s.lower().split())
                elif name == "pmFromFile":
                    arg = _pm([w for fn in arg_s.split() for w in _phrases(rules_dir / fn)])
                elif name in ("contains", "beginsWith", "endsWith", "streq"):
                    arg = arg_s
                elif name == "within":
                    arg = arg_s
                else:
                    skip(f"operator @{name}")
                    continue
            except re.error:
                skip("regex not supported by Python")
                continue
            sev = SEVERITY.get((_action(acts, "severity") or ["NOTICE"])[0].upper(), 2)
            rules.append(Rule(id=(_action(acts, "id") or ["?"])[0], file=num, variables=variables,
                              negate=negate, op=name, arg=arg, transforms=ts, severity=sev,
                              msg=(_action(acts, "msg") or [""])[0], tags=_action(acts, "tag")))  # fmt: skip
    return rules, skipped


@lru_cache(maxsize=4)
def _rules(rules_dir: str, paranoia: int) -> tuple[Rule, ...]:
    return tuple(parse_rules(Path(rules_dir), paranoia)[0])


_ON: dict[str, Any] = {"t": 0.0, "on": False}


def enabled() -> bool:
    """CRS on (Settings ``crs.enabled``, default on since the 2026-10-23 held-out check) and
    its rules present; read at most every 30 s."""
    import time as _time

    if _time.time() - _ON["t"] > 30:
        from src.soc import sinks

        try:
            on = bool((sinks.load_settings().get("crs") or {}).get("enabled", True))
        except (OSError, ValueError):
            on = True
        _ON.update(t=_time.time(), on=on and available())
    return bool(_ON["on"])


def available(rules_dir: Path | None = None) -> bool:
    d = rules_dir or RULES_DIR
    try:
        return any(d.glob("REQUEST-94*.conf"))
    except OSError:
        return False


def request(method: str, target: str, proto: str = "", ua: str = "", referer: str = "") -> dict[str, Any]:
    """The ModSecurity variables an access-log line gives."""
    path, _, query = target.partition("?")
    args: dict[str, str] = {}
    for part in query.split("&") if query else []:
        k, _, v = part.partition("=")
        args[urllib.parse.unquote_plus(k, errors="replace")] = urllib.parse.unquote_plus(v, errors="replace")
    fname = urllib.parse.unquote(path, errors="replace")
    return {"REQUEST_METHOD": method, "REQUEST_URI": target, "REQUEST_URI_RAW": target,
            "REQUEST_FILENAME": fname, "REQUEST_BASENAME": fname.rsplit("/", 1)[-1],
            "QUERY_STRING": query, "REQUEST_PROTOCOL": proto,
            "REQUEST_LINE": f"{method} {target} {proto}".strip(), "args": args,
            "headers": {k: v for k, v in (("user-agent", ua), ("referer", referer)) if v and v != "-"}}  # fmt: skip


def check(req: dict[str, Any], rules_dir: Path | None = None, paranoia: int = PARANOIA,
          threshold: int = THRESHOLD, disabled: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:  # fmt: skip
    """Anomaly score of one request and the rules that matched (``disabled``: rule ids
    a user switched off)."""
    hits = [r for r in _rules(str(rules_dir or RULES_DIR), paranoia) if r.id not in disabled and r.matches(req)]
    score = sum(r.severity for r in hits)
    techs: list[str] = []
    for r in hits:
        techs += [t for t in FILE_TECHNIQUES.get(r.file, []) if t not in techs]
    classes = sorted({t.split("/", 1)[0] for r in hits for t in r.tags if t.startswith("attack-")})
    top = max(hits, key=lambda r: r.severity) if hits else None
    return {"score": score, "attack": score >= threshold, "rules": [r.id for r in hits],
            "techniques": techs, "classes": classes, "msg": top.msg if top else ""}  # fmt: skip
