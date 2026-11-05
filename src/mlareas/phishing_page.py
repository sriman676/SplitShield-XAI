"""Phishing page model (owner 2026-10-24): rates a web page from its HTML and URL.

The URL model (phishing.py) sees only the host name, so phishing on ordinary-looking
hosts is missed (recall 0.59 held-out). When the page itself is available (an uploaded
.html, an e-mail attachment, a proxy / sandbox capture) this model reads what the page
asks for and where it sends it: password and login forms, forms posting to another
site or to a script, links that point nowhere, resources loaded from other sites,
brand names that do not match the page's own domain, obfuscated script, redirects.

No page is ever fetched by this module; it scores HTML it is given.

Collection artefacts are removed before any feature is computed (save-tool attributes
and comments such as ``data-scrapbook-*`` or ``<!-- saved from url=...>``), so a model
cannot learn how a dataset was collected instead of what the page does.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import numpy as np

from src.mlareas.phishing import registered_domain

MAX_HTML = 2_000_000  # characters read per page
N_HASH = 1024  # hashed title / text word features

_ARTEFACTS = [
    re.compile(r"\sdata-scrapbook-[\w-]+=(\"[^\"]*\"|'[^']*'|\S+)", re.I),
    re.compile(r"<!--\s*saved from url=.*?-->", re.I | re.S),
    re.compile(r"<!--\s*(?:page saved with|singlefile|scrapbook|mirrored from|archived).*?-->", re.I | re.S),
    re.compile(r"<meta[^>]+name=[\"']?(?:savepage-|singlefile)[^>]*>", re.I),
]
_TAG = re.compile(r"<\s*([a-zA-Z][\w-]*)([^>]*)>", re.S)
_ATTR = re.compile(r"([\w:-]+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.S)
_SCRIPT = re.compile(r"<script[^>]*>(.*?)</script>", re.I | re.S)
_STYLE = re.compile(r"<style[^>]*>.*?</style>", re.I | re.S)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_WORD = re.compile(r"[a-z]{3,20}")
BRANDS = ("paypal", "apple", "microsoft", "office", "outlook", "google", "gmail", "amazon", "netflix",
          "facebook", "instagram", "whatsapp", "linkedin", "dropbox", "docusign", "adobe", "chase", "wellsfargo",
          "bankofamerica", "citibank", "hsbc", "santander", "barclays", "dhl", "fedex", "ups", "usps", "royalmail",
          "ebay", "alibaba", "yahoo", "aol", "steam", "binance", "coinbase", "metamask", "irs", "hmrc")  # fmt: skip
LURE = ("login", "log in", "sign in", "signin", "verify", "verification", "password", "account", "suspended",
        "unlock", "confirm", "update your", "billing", "invoice", "wallet", "security alert", "unusual activity",
        "limited", "expire", "ssn", "credit card", "card number", "cvv", "pin")  # fmt: skip
OBFUSCATION = ("eval(", "unescape(", "atob(", "fromcharcode", "document.write(", "escape(", "\\x", "%u")


def clean_html(html: str) -> str:
    s = html[:MAX_HTML]
    for p in _ARTEFACTS:
        s = p.sub("", s)
    return s


def _attrs(raw: str) -> dict[str, str]:
    return {k.lower(): v.strip("\"'") for k, v in _ATTR.findall(raw)}


def _host(u: str) -> str:
    try:
        return (urlsplit(u).hostname or "").lower()
    except ValueError:
        return ""


def _abs(base: str, ref: str) -> str:
    """urljoin that never raises (malformed links, e.g. a broken IPv6 literal, are common)."""
    try:
        return urljoin(base or "http://x/", ref)
    except ValueError:
        return ""


def _hash_words(text: str, tag: str) -> np.ndarray:
    import zlib

    v = np.zeros(N_HASH, dtype=np.float32)
    words = _WORD.findall(text.lower())
    for w in words:
        v[zlib.crc32(f"{tag}:{w}".encode()) % N_HASH] += 1.0
    if words:
        v /= np.sqrt(len(words))
    return v


NUMERIC = ["html_kb", "n_tags", "n_forms", "n_inputs", "n_password", "n_hidden", "n_email_input",
           "form_external", "form_empty_action", "form_php", "form_mailto", "n_links", "link_external_share",
           "link_null_share", "link_distinct_hosts", "n_scripts", "script_external_share", "n_iframes",
           "iframe_hidden", "n_images", "img_external_share", "css_external_share", "favicon_external",
           "meta_refresh", "js_redirect", "obfuscation", "right_click_block", "popup", "n_brand_mentions",
           "brand_not_own_domain", "title_brand_not_own", "lure_words", "title_len", "text_kb",
           "has_copyright", "has_privacy_link", "base64_blobs", "url_in_page_own_host_share"]  # fmt: skip


def page_features(html: str, url: str) -> np.ndarray:
    """Numeric page features + hashed title words + hashed visible-text words."""
    s = clean_html(html or "")
    low = s.lower()
    own_host = _host(url)
    own = registered_domain(own_host) if own_host else ""
    tags = _TAG.findall(s)
    f: dict[str, float] = dict.fromkeys(NUMERIC, 0.0)
    f["html_kb"] = len(s) / 1024
    f["n_tags"] = len(tags)
    links = scripts_ext = scripts = imgs = imgs_ext = css = css_ext = 0
    link_ext = link_null = 0
    hosts: set[str] = set()
    own_refs = all_refs = 0
    for name, raw in tags:
        n = name.lower()
        a = _attrs(raw)
        if n == "form":
            f["n_forms"] += 1
            act = a.get("action", "").strip()
            if not act or act in ("#", "about:blank"):
                f["form_empty_action"] += 1
            elif act.lower().startswith("mailto:"):
                f["form_mailto"] += 1
            else:
                h = _host(_abs(url, act))
                if h and own and registered_domain(h) != own:
                    f["form_external"] += 1
                if ".php" in act.lower():
                    f["form_php"] += 1
        elif n == "input":
            f["n_inputs"] += 1
            t = a.get("type", "text").lower()
            nm = (a.get("name", "") + a.get("id", "") + a.get("placeholder", "")).lower()
            if t == "password" or "pass" in nm:
                f["n_password"] += 1
            if t == "hidden":
                f["n_hidden"] += 1
            if t == "email" or "email" in nm or "user" in nm:
                f["n_email_input"] += 1
        elif n == "a":
            links += 1
            href = a.get("href", "").strip()
            if not href or href.startswith(("#", "javascript:", "about:")):
                link_null += 1
                continue
            h = _host(_abs(url, href))
            if h:
                hosts.add(h)
                all_refs += 1
                if own and registered_domain(h) != own:
                    link_ext += 1
                else:
                    own_refs += 1
            if "privacy" in href.lower():
                f["has_privacy_link"] = 1
        elif n == "script":
            scripts += 1
            src = a.get("src", "")
            if src:
                h = _host(_abs(url, src))
                if h and own and registered_domain(h) != own:
                    scripts_ext += 1
        elif n == "iframe":
            f["n_iframes"] += 1
            st = (a.get("style", "") + a.get("width", "") + a.get("height", "")).replace(" ", "").lower()
            if "display:none" in st or "visibility:hidden" in st or a.get("width") == "0" or a.get("height") == "0":
                f["iframe_hidden"] += 1
        elif n == "img":
            imgs += 1
            h = _host(_abs(url, a.get("src", "")))
            if h and own and registered_domain(h) != own:
                imgs_ext += 1
        elif n == "link":
            rel = a.get("rel", "").lower()
            h = _host(_abs(url, a.get("href", "")))
            ext = bool(h and own and registered_domain(h) != own)
            if "stylesheet" in rel:
                css += 1
                css_ext += ext
            if "icon" in rel and ext:
                f["favicon_external"] = 1
        elif n == "meta" and a.get("http-equiv", "").lower() == "refresh":
            f["meta_refresh"] = 1
    f["n_links"], f["n_scripts"], f["n_images"] = links, scripts, imgs
    f["link_external_share"] = link_ext / max(1, links)
    f["link_null_share"] = link_null / max(1, links)
    f["link_distinct_hosts"] = len(hosts)
    f["script_external_share"] = scripts_ext / max(1, scripts)
    f["img_external_share"] = imgs_ext / max(1, imgs)
    f["css_external_share"] = css_ext / max(1, css)
    f["url_in_page_own_host_share"] = own_refs / max(1, all_refs)
    js = " ".join(_SCRIPT.findall(s)).lower()
    f["js_redirect"] = float(any(k in js for k in ("window.location", "location.href", "location.replace")))
    f["obfuscation"] = float(sum(js.count(k) for k in OBFUSCATION))
    f["right_click_block"] = float("contextmenu" in js or "event.button==2" in js.replace(" ", ""))
    f["popup"] = float("window.open(" in js)
    f["base64_blobs"] = float(len(re.findall(r"[A-Za-z0-9+/]{200,}={0,2}", s)))
    tm = _TITLE.search(s)
    title = re.sub(r"\s+", " ", tm.group(1)).strip().lower() if tm else ""
    text = _TAG.sub(" ", _SCRIPT.sub(" ", _STYLE.sub(" ", s)))
    text = re.sub(r"<[^>]*>|&\w+;|\s+", " ", text).lower()[:200_000]
    f["title_len"] = len(title)
    f["text_kb"] = len(text) / 1024
    own_l = own.split(".")[0] if own else ""
    mentioned = [b for b in BRANDS if b in text or b in title]
    f["n_brand_mentions"] = len(mentioned)
    f["brand_not_own_domain"] = float(any(b != own_l and b not in own_host for b in mentioned))
    f["title_brand_not_own"] = float(any(b in title and b != own_l and b not in own_host for b in BRANDS))
    f["lure_words"] = float(sum(text.count(w) + 3 * title.count(w) for w in LURE))
    f["has_copyright"] = float("©" in text or "&copy" in low or "copyright" in text)
    num = np.array([f[k] for k in NUMERIC], dtype=np.float32)
    return np.concatenate([num, _hash_words(title, "t"), _hash_words(text, "x")])


def feature_names() -> list[str]:
    return [*NUMERIC, *[f"title_w{i}" for i in range(N_HASH)], *[f"text_w{i}" for i in range(N_HASH)]]


def featurize(pages: Iterable[tuple[str, str]]) -> np.ndarray:
    return np.vstack([page_features(h, u) for h, u in pages])


def page_facts(html: str, url: str) -> list[str]:
    """Plain-English reasons from the numeric features (for the dashboard / assistant)."""
    v = dict(zip(NUMERIC, page_features(html, url)[: len(NUMERIC)].tolist(), strict=True))
    out = []
    if v["n_password"]:
        out.append("asks for a password")
    if v["form_external"]:
        out.append("a form sends what you type to another website")
    if v["form_php"] or v["form_mailto"]:
        out.append("a form posts to a script or an e-mail address")
    if v["brand_not_own_domain"] or v["title_brand_not_own"]:
        out.append("names a well-known brand that does not own this domain")
    if v["link_null_share"] > 0.5 and v["n_links"] >= 5:
        out.append("most links lead nowhere")
    if v["meta_refresh"] or v["js_redirect"]:
        out.append("redirects the visitor")
    if v["obfuscation"] >= 3 or v["base64_blobs"]:
        out.append("contains hidden or encoded script")
    if v["iframe_hidden"]:
        out.append("has a hidden frame")
    return out


BUNDLE = Path(__file__).resolve().parents[2] / "artifacts" / "mlareas" / "phishing" / "page" / "page_model.joblib"
_CACHE: dict[str, Any] = {}


def load(path: Path | None = None) -> dict[str, Any]:
    """The deployed page model (owner 2026-10-24: uploaded / attached HTML only)."""
    import joblib

    p = path or BUNDLE
    if str(p) not in _CACHE:
        if not p.is_file():
            raise FileNotFoundError(f"no phishing page model at {p}; run python -m scripts.train_phishing_page")
        _CACHE[str(p)] = joblib.load(p)  # our own artifact
    b: dict[str, Any] = _CACHE[str(p)]
    return b


def html_from_email(raw: bytes) -> list[tuple[str, str]]:
    """(name, html) of every HTML part of an e-mail (.eml): the body and attachments."""
    import email
    from email import policy

    msg = email.message_from_bytes(raw, policy=policy.default)
    out = []
    for i, part in enumerate(msg.walk()):
        ctype = part.get_content_type()
        fname = part.get_filename() or ""
        if ctype == "text/html" or fname.lower().endswith((".html", ".htm")):
            try:
                payload = part.get_payload(decode=True) or b""
            except (ValueError, LookupError):
                continue
            if not isinstance(payload, bytes):
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except LookupError:  # unknown charset name
                text = payload.decode("utf-8", errors="replace")
            out.append((fname or f"html part {i}", text))
    return out


def check_page(html: str, url: str, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one page with a trained bundle {"model", "threshold", "keep", ...}."""
    bundle = bundle or load()
    x = page_features(html, url)[None, :]
    if bundle.get("keep") is not None:
        x = x[:, bundle["keep"]]
    p = float(bundle["model"].predict_proba(x)[0, 1])
    thr = float(bundle["threshold"])
    return {"url": url, "probability": round(p, 4), "verdict": "phishing" if p >= thr else "legitimate",
            "threshold": thr, "reasons": page_facts(html, url)}  # fmt: skip
