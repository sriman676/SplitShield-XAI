"""Phishing URL detection from the URL string alone (owner 2026-10-15).

A live SOC usually sees only the URL (proxy, mail gateway, DNS logs), not the page,
so the deployed model uses features we compute ourselves from the raw string.

Leakage found in PhiUSIIL (UCI 967) and how it is handled:
  * every legitimate URL is a bare ``https://www.<domain>`` homepage, while every
    ``http://`` URL and every URL with a path / trailing slash is phishing. A model
    on the full string can score ~1.0 just by reading the scheme / path shape.
  * the deployed ("url_host") model therefore looks ONLY at the hostname after
    stripping scheme, credentials, port, path and a leading ``www.``. Rows are
    de-duplicated by that host (conflicting labels dropped) so one host never sits
    in both train and test.
  * the full-string model and the all-dataset-features model are still trained and
    reported, as references and to show the size of the artifact.

Labels: PhiUSIIL uses ``label=1`` for legitimate, ``0`` for phishing; here
``y=1`` always means PHISHING.

SOC API: ``check_url``, ``check_urls``, ``improve`` (champion/challenger), ``load``,
and ``blind_test(urls, labels)`` which scores an unseen labelled URL set with the
frozen model (no refit, no re-threshold; the lead runs it once at the very end).
Each ``why`` is a numbered, non-expert explanation: checkable facts about the
address, the model's strongest signals, the verdict, and what to do.
"""

from __future__ import annotations

import functools
import ipaddress
import math
import os
import re
import time
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.mlareas import common

AREA = "phishing"
N_HASH = 256
FAMILIES = ("lightgbm", "logreg", "random_forest")
VARIANTS = ("url_host", "url_full", "all_features")
SUSPICIOUS_AT = 0.2  # probability band 0.2-0.5 is reported as "suspicious"

BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "onedrive",
    "google", "gmail", "amazon", "netflix", "facebook", "instagram", "whatsapp",
    "metamask", "coinbase", "binance", "blockchain", "chase", "wellsfargo",
    "bankofamerica", "citi", "hsbc", "dhl", "fedex", "usps", "ups", "adobe",
    "dropbox", "docusign", "linkedin", "steam", "roblox", "att", "verizon",
)  # fmt: skip
SUSPICIOUS_WORDS = (
    "login", "signin", "secure", "verify", "account", "update", "confirm", "bank",
    "wallet", "support", "auth", "password", "billing", "recover", "unlock",
    "webmail", "service", "invoice", "claim", "gift", "bonus", "free", "help",
)  # fmt: skip
FREE_HOSTING = (
    "firebaseapp.com", "web.app", "ipfs", "repl.co", "weebly.com", "wixsite.com",
    "godaddysites.com", "000webhostapp.com", "surge.sh", "vercel.app",
    "netlify.app", "github.io", "glitch.me", "herokuapp.com", "blogspot.com",
    "pages.dev", "workers.dev", "ngrok", "duckdns.org", "azurewebsites.net",
    "square.site", "webflow.io", "gitbook.io", "sharepoint.com", "myshopify.com",
)  # fmt: skip
SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "edu", "or", "ne", "go"}

HOST_FEATURES = (
    "host_len", "n_labels", "n_hyphens", "n_digits", "digit_ratio", "entropy",
    "is_ip", "tld_len", "punycode", "max_label_len", "vowel_ratio",
    "max_consonant_run", "brand_not_owner", "susp_words", "free_hosting",
    "digits_in_domain", "domain_len",
)  # fmt: skip
# mode "host2" (owner 2026-10-16, phishing underfitting): richer hostname features on
# top of HOST_FEATURES - still the hostname only, never the leaking path/scheme
HOST2_EXTRA = (
    "brand_edit_dist", "brand_in_label", "sub_len", "n_sub_labels", "longest_digit_run",
    "max_repeat", "tld_common", "tld_abused", "domain_entropy", "domain_hyphens",
    "domain_vowel_ratio", "domain_digit_ratio", "label_len_mean", "bigram_rarity",
)  # fmt: skip
COMMON_TLDS = frozenset("com org net edu gov mil int de uk fr it es nl jp ru br in au ca cn "
                        "ch se no fi dk pl at be io co us eu kr tw".split())  # fmt: skip
# TLDs frequently abused in phishing feeds (public abuse reports)
ABUSED_TLDS = frozenset("tk ml ga cf gq xyz top club online site live shop icu buzz cyou "
                        "rest monster quest sbs cfd bond click link work support help "
                        "zip mov su".split())  # fmt: skip
# letter bigrams common in English and brand names; a name made of rare bigrams looks
# machine-generated
_COMMON_BIGRAMS = frozenset(
    "th he in er an re on at en nd ti es or te of ed is it al ar st to nt ng se ha as ou "
    "io le ve co me de hi ri ro ic ne ea ra ce li ch ll be ma si om ur ca el ta la ns di "
    "fo ho pe ec pr no ct us ac ot il tr ly nc et ut ss so rs un lo wa ge ie wh ee wi em "
    "ad ol rt po we na ul ni ts mo ow pa im mi ai sh ir su id os iv ia am fi ci vi pl ig "
    "tu ev ld ry mp fe bl ab gh ty op wo sa ay ex ke fr oo av ag if ap gr od bo sp rd do "
    "uc bu ei ov by rm ep tt oc fa ef cu rn sc gi da yo cr cl du ga qu ue ff ba ey ls va "
    "um pp ua up lu go ht ru ug ds lt pi rc rr eg au ck ew mu br bi pt ak pu ui rg ib tl "
    "ny ki rk ys ob mm fu ph og ms ye ud mb ip ub oi rl gu dr hr cc tw ft wn nu af hu nn "
    "eo vo rv nf xp gn sm fl iz ok nl my gl aw sy oa".split())  # fmt: skip
FULL_EXTRA = (
    "url_len", "is_https", "has_www", "path_len", "path_depth", "n_query",
    "n_at", "n_pct", "n_special", "double_slash", "brand_in_path",
    "susp_words_path", "ends_slash", "script_ext",
)  # fmt: skip

PLAIN = {
    "host_len": "a long hostname",
    "n_labels": "many sub-domain levels",
    "n_hyphens": "hyphens in the hostname",
    "n_digits": "digits in the hostname",
    "digit_ratio": "a high share of digits in the hostname",
    "entropy": "a random-looking hostname (high character entropy)",
    "is_ip": "a raw IP address instead of a domain name",
    "tld_len": "an unusual top-level domain length",
    "punycode": "punycode (look-alike international characters)",
    "max_label_len": "one very long hostname label",
    "vowel_ratio": "an unusual vowel/consonant mix",
    "max_consonant_run": "long consonant runs (machine-generated name)",
    "brand_not_owner": "a well-known brand name on a domain the brand does not own",
    "susp_words": "words like login/verify/secure/account in the hostname",
    "free_hosting": "a free hosting / site-builder platform",
    "digits_in_domain": "digits in the registered domain",
    "domain_len": "the length of the registered domain",
    "brand_edit_dist": "a name very close to (but not exactly) a well-known brand",
    "brand_in_label": "a brand name inside a longer hostname label",
    "sub_len": "a long sub-domain part",
    "n_sub_labels": "many sub-domain labels",
    "longest_digit_run": "a long run of digits",
    "max_repeat": "repeated characters (e.g. 'paypaal')",
    "tld_common": "the top-level domain (common vs unusual)",
    "tld_abused": "a top-level domain often abused for phishing",
    "domain_entropy": "a random-looking registered domain",
    "domain_hyphens": "hyphens in the registered domain",
    "domain_vowel_ratio": "an unusual vowel mix in the registered domain",
    "domain_digit_ratio": "digits in the registered domain",
    "label_len_mean": "unusually long labels",
    "bigram_rarity": "letter pairs rarely seen in real words (machine-made name)",
    "url_len": "a long URL",
    "is_https": "the scheme (http vs https)",
    "has_www": "the presence of 'www.'",
    "path_len": "a long path",
    "path_depth": "a deep folder path",
    "n_query": "many query parameters",
    "n_at": "an '@' sign (can hide the real destination)",
    "n_pct": "percent-encoded characters",
    "n_special": "many special characters",
    "double_slash": "'//' inside the path (redirect trick)",
    "brand_in_path": "a brand name in the path",
    "susp_words_path": "login/verify/account words in the path",
    "ends_slash": "a trailing slash",
    "script_ext": "a .php/.html/.asp page in the path",
}
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


# ------------------------------------------------------------ URL features


def split_url(url: str) -> tuple[str, str]:
    """Return ``(host, rest)``: host lower-cased without www/port/userinfo."""
    s = _SCHEME.sub("", url.strip())
    host, _, rest = s.partition("/")
    for sep in ("?", "#"):
        if sep in host:
            host, _, tail = host.partition(sep)
            rest = sep + tail + rest
    host = host.rsplit("@", 1)[-1].split(":", 1)[0].lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host, rest


def registered_domain(host: str) -> str:
    if _is_ip(host):
        return host
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"(0x[0-9a-f]+|\d+)", host))


def _max_consonant_run(s: str) -> int:
    runs = re.findall(r"[bcdfghjklmnpqrstvwxz]+", s)
    return max((len(r) for r in runs), default=0)


def _brand_not_owner(host: str, domain: str) -> int:
    owner = domain.split(".")[0]
    return int(any(b in host and owner != b for b in BRANDS))


def host_features(host: str) -> list[float]:
    labels = host.split(".") if host else [""]
    domain = registered_domain(host)
    letters = [c for c in host if c.isalpha()]
    vowels = sum(c in "aeiou" for c in letters)
    n_dig = sum(c.isdigit() for c in host)
    return [
        len(host),
        len(labels),
        host.count("-"),
        n_dig,
        n_dig / max(len(host), 1),
        _entropy(host),
        float(_is_ip(host)),
        len(labels[-1]),
        float("xn--" in host),
        max(len(x) for x in labels),
        vowels / max(len(letters), 1),
        _max_consonant_run(host),
        _brand_not_owner(host, domain),
        sum(w in host for w in SUSPICIOUS_WORDS),
        float(any(h in host for h in FREE_HOSTING)),
        sum(c.isdigit() for c in domain),
        len(domain),
    ]


def _edit_distance(a: str, b: str, cap: int = 4) -> int:
    """Levenshtein distance, stopped early above ``cap`` (cheap for short labels)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


_DIGITS = re.compile(r"\d+")
_REPEAT = re.compile(r"(.)\1*")


def host2_extra_features(host: str) -> list[float]:
    labels = host.split(".") if host else [""]
    domain = registered_domain(host)
    owner = domain.split(".")[0]
    sub = host[: -len(domain)].rstrip(".") if host.endswith(domain) else ""
    tld = labels[-1]
    # closest brand to the whole name or any of its dash parts ("paypa1-login")
    tokens = {owner, *re.split(r"[-_]", owner)} - {""}
    dist = min((_edit_distance(t, b) for t in tokens for b in BRANDS if t != b), default=5)
    letters = [c for c in owner if c.isalpha()]
    pairs = [owner[i : i + 2] for i in range(len(owner) - 1) if owner[i : i + 2].isalpha()]
    return [
        5.0 if owner in BRANDS else float(dist),  # exact brand name = the real owner
        float(any(b in lab and lab != b for lab in labels for b in BRANDS)),
        len(sub),
        len(sub.split(".")) if sub else 0,
        max((len(r) for r in _DIGITS.findall(host)), default=0),
        max((len(m.group(0)) for m in _REPEAT.finditer(host)), default=0),
        float(tld in COMMON_TLDS),
        float(tld in ABUSED_TLDS),
        _entropy(owner),
        owner.count("-"),
        sum(c in "aeiou" for c in letters) / max(len(letters), 1),
        sum(c.isdigit() for c in owner) / max(len(owner), 1),
        sum(len(x) for x in labels) / max(len(labels), 1),
        sum(p not in _COMMON_BIGRAMS for p in pairs) / max(len(pairs), 1),
    ]


def full_extra_features(url: str, rest: str) -> list[float]:
    low = url.lower()
    path = rest.split("?", 1)[0]
    return [
        len(url),
        float(low.startswith("https://")),
        float("://www." in low or low.startswith("www.")),
        len(rest),
        path.count("/"),
        rest.count("&") + int("?" in rest),
        url.count("@"),
        url.count("%"),
        sum(not c.isalnum() for c in rest),
        float("//" in rest),
        float(any(b in rest.lower() for b in BRANDS)),
        sum(w in rest.lower() for w in SUSPICIOUS_WORDS),
        float(url.endswith("/")),
        float(bool(re.search(r"\.(php|html?|aspx?|jsp)\b", path.lower()))),
    ]


def _hashed_ngrams(text: str, n: int = 3, tag: str = "") -> np.ndarray:
    vec = np.zeros(N_HASH, dtype=np.float32)
    s = f"^{text}$"
    grams = [s[i : i + n] for i in range(max(len(s) - n + 1, 1))]
    for g in grams:
        vec[zlib.crc32((tag + g).encode("utf-8", "ignore")) % N_HASH] += 1.0
    tld = text.rsplit(".", 1)[-1]
    vec[zlib.crc32(f"tld:{tld}".encode()) % N_HASH] += 1.0
    return vec / max(len(grams), 1)


def feature_names(mode: str) -> list[str]:
    names = list(HOST_FEATURES)
    if mode == "full":
        names += list(FULL_EXTRA)
    if mode == "host2":
        return (names + list(HOST2_EXTRA) + [f"ngram_{i}" for i in range(N_HASH)]
                + [f"ngram2_{i}" for i in range(N_HASH)]
                + [f"ngram4_{i}" for i in range(N_HASH)])  # fmt: skip
    return names + [f"ngram_{i}" for i in range(N_HASH)]


def featurize(urls: Iterable[str], mode: str = "host") -> np.ndarray:
    """URL strings -> float32 matrix. ``mode='host'`` ignores scheme/path."""
    rows = []
    for url in urls:
        host, rest = split_url(str(url))
        num = host_features(host)
        if mode == "full":
            num += full_extra_features(str(url), rest)
        if mode == "host2":
            rows.append(np.concatenate([
                np.asarray(num + host2_extra_features(host), np.float32), _hashed_ngrams(host),
                _hashed_ngrams(host, 2, "b:"), _hashed_ngrams(host, 4, "q:"),
            ]))  # fmt: skip
            continue
        ngram_src = host if mode == "host" else _SCHEME.sub("", str(url).lower())
        rows.append(
            np.concatenate([np.asarray(num, np.float32), _hashed_ngrams(ngram_src)])
        )
    if not rows:
        return np.zeros((0, len(feature_names(mode))), np.float32)
    return np.vstack(rows)


# ------------------------------------------------------------ data + models


def load_phiusiil(csv_path: Path | None = None) -> pd.DataFrame:
    path = (
        csv_path
        or common.raw_data_dir("phiusiil") / "PhiUSIIL_Phishing_URL_Dataset.csv"
    )
    df = pd.read_csv(path, encoding="latin-1")
    df["y"] = (df["label"] == 0).astype(np.int8)  # 1 = phishing
    return df


def dedupe_hosts(
    urls: Sequence[str], y: Sequence[int]
) -> tuple[list[str], np.ndarray, int]:
    """One row per host; hosts seen with both labels are dropped (count returned)."""
    frame = pd.DataFrame({"host": [split_url(str(u))[0] for u in urls], "y": list(y)})
    agg = frame.groupby("host")["y"].agg(["min", "max"])
    conflicts = int((agg["min"] != agg["max"]).sum())
    agg = agg[agg["min"] == agg["max"]]
    agg = agg[agg.index != ""]
    return list(agg.index), agg["min"].to_numpy(np.int8), conflicts


def make_model(family: str, seed: int) -> Any:
    if family == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, n_jobs=4, verbose=-1,
        )  # fmt: skip
    if family == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=150, max_features="sqrt", n_jobs=4, random_state=seed
        )
    raise ValueError(f"unknown family {family}")


def binary_metrics(
    y: np.ndarray, proba: np.ndarray, thr: float = 0.5
) -> dict[str, float]:
    pred = (proba >= thr).astype(int)
    tn, fp, _fn, _tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) > 1
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if both else float("nan"),
        "fpr_legit": float(fp / max(fp + tn, 1)),
    }


SMALL_BATCH = 512  # below this, one thread is faster (no thread start-up, no contention)


def _threads(n: int) -> int:
    return 1 if n < SMALL_BATCH else (os.cpu_count() or 1)


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if isinstance(model, lgb.LGBMClassifier):  # exact for any thread count
        return np.asarray(model.predict_proba(X, num_threads=_threads(len(X)))[:, 1], dtype=np.float64)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def split_indices(
    y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70/15/15 split of row indices."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return tr, va, te


def calibrate(proba_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(proba_val, y_val)
    return iso


# ------------------------------------------------------------ SOC API

_CACHE: dict[str, Any] = {}


def load(root: Path | None = None, refresh: bool = False) -> dict[str, Any]:
    """Load (and cache) the deployed URL-only bundle."""
    key = str(root)
    if refresh or key not in _CACHE:
        _CACHE[key] = common.load_bundle(AREA, root)
    bundle: dict[str, Any] = _CACHE[key]
    return bundle


def _contributions(model: Any, X: np.ndarray) -> np.ndarray | None:
    if isinstance(model, lgb.LGBMClassifier):
        # TreeSHAP is the costly part (~0.1 s per URL single-threaded on the host2
        # features); it is exact for any thread count, so use them all
        threads = os.cpu_count() or 1
        contrib = np.asarray(model.predict(X, pred_contrib=True, num_threads=threads))
        return contrib[:, :-1]  # last column is the bias term
    if isinstance(model, Pipeline):
        scaler, lr = model.steps[0][1], model.steps[-1][1]
        return np.asarray(scaler.transform(X) * lr.coef_[0])
    return None


def url_facts(url: str) -> list[str]:
    """Plain-English, checkable facts about the URL string (no model involved)."""
    host, rest = split_url(url)
    domain = registered_domain(host)
    owner = domain.split(".")[0]
    facts = []
    if _is_ip(host):
        facts.append(f"The link points to a raw IP address ({host}) instead of a named "
                     "website; real companies almost always use a domain name.")  # fmt: skip
    brands = [b for b in BRANDS if b in host and owner != b]
    if brands:
        facts.append(f"The address mentions '{brands[0]}' but the site is registered under "
                     f"'{domain}', not '{brands[0]}.<ending>'. Unless '{domain}' is known to "
                     f"belong to {brands[0]}, this is a classic imitation trick.")  # fmt: skip
    hosting = next((h for h in FREE_HOSTING if h in host), None)
    if hosting:
        facts.append(f"The page lives on '{hosting}', a free hosting / site-builder service "
                     "that anyone can use to publish a page in minutes.")  # fmt: skip
    if "xn--" in host:
        facts.append(
            "The hostname uses punycode, which can disguise look-alike letters."
        )
    words = [w for w in SUSPICIOUS_WORDS if w in host]
    if words:
        facts.append(f"The hostname contains pressure words often used in phishing: "
                     f"{', '.join(repr(w) for w in words[:4])}.")  # fmt: skip
    if host.count(".") >= 3 and not _is_ip(host):
        facts.append(f"The hostname has {host.count('.') + 1} parts; long chains of "
                     "sub-domains are a common way to bury the real domain.")  # fmt: skip
    if "@" in url:
        facts.append("The URL contains '@'; browsers ignore everything before it, so the "
                     "visible start of the link is not where it goes.")  # fmt: skip
    if any(w in rest.lower() for w in ("login", "verify", "signin", "password")):
        facts.append("The path asks for a login/verification step (noted for context; the "
                     "model itself scores only the hostname).")  # fmt: skip
    return facts


def _model_signals(
    names: list[str],
    contrib_row: np.ndarray | None,
    values: np.ndarray,
    phishing: bool,
) -> list[str]:
    """Top-3 model features pushing toward the verdict, in plain English."""
    if contrib_row is None:
        return []
    agg: dict[str, float] = {}
    for name, c, v in zip(names, contrib_row, values, strict=True):
        if name.startswith(("ngram_", "ngram2_", "ngram4_")):
            key = "the overall spelling pattern of the hostname"
        elif v == 0:  # the model credits the ABSENCE of this property
            key = "no " + re.sub(r"^(an?|the) ", "", PLAIN.get(name, name))
        else:
            key = PLAIN.get(name, name)
        agg[key] = agg.get(key, 0.0) + float(c)
    sign = 1.0 if phishing else -1.0
    ranked = [
        t
        for t, v in sorted(agg.items(), key=lambda kv: -sign * kv[1])[:3]
        if sign * v > 0
    ]
    return ranked


ACTIONS = {
    "phishing": "Do not click the link or enter any password; report the message to "
                "IT/security; block '{target}' at the proxy / mail gateway.",
    "suspicious": "Treat with caution: confirm with the sender through another channel "
                  "before opening; ask security to review '{target}'.",
    "legitimate": "No action needed based on the address. This check looks at the "
                  "hostname only, not the page content, so stay alert to odd requests.",
}  # fmt: skip


def explain(
    url: str, p: float, verdict: str, signals: list[str], model: str
) -> list[str]:
    """Numbered, non-expert explanation: facts, model signals, verdict, action."""
    steps = url_facts(url) if verdict != "legitimate" else []
    if signals:
        label = "raised" if verdict != "legitimate" else "lowered"
        steps.append(f"The model's strongest signals that {label} the risk were: "
                     f"{'; '.join(signals)}.")  # fmt: skip
    if not steps:
        steps.append(
            "No single strong warning sign; the score comes from many weak ones."
        )
    lines = [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    lines.append(f"Verdict: {verdict.upper()} (phishing probability {p:.0%}, {model}).")
    host = split_url(url)[0]
    shared = _is_ip(host) or any(h in host for h in FREE_HOSTING)
    # on shared hosting the platform domain is not the attacker's: block the host
    target = host if shared else registered_domain(host)
    lines.append("What to do: " + ACTIONS[verdict].format(target=target))
    return lines


@functools.lru_cache(maxsize=1)
def top_domains() -> frozenset[str]:
    """Registered domains of the Tranco top 10,000 (src/knowledge/top_domains.txt;
    Le Pochat et al., NDSS 2019, tranco-list.eu)."""
    p = Path(__file__).resolve().parents[1] / "knowledge" / "top_domains.txt"
    try:
        return frozenset(x.strip().lower() for x in p.read_text("utf-8").splitlines() if x.strip())
    except OSError:
        return frozenset()


def _verdict(p: float, thr: float, suspicious_at: float = SUSPICIOUS_AT) -> str:
    if p >= thr:
        return "phishing"
    return "suspicious" if p >= suspicious_at else "legitimate"


def _score(urls: Sequence[str], b: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    X = featurize(urls, b["mode"])
    raw = predict_proba(b["model"], X)
    proba = b["calibrator"].predict(raw) if b.get("calibrator") is not None else raw
    return np.asarray(proba, np.float64), X


EXPLAIN_ALL_UP_TO = 100  # batches up to this size explain every URL in full


def check_urls(
    urls: Sequence[str],
    bundle: dict[str, Any] | None = None,
    explain_all: bool | None = None,
) -> list[dict[str, Any]]:
    """Score a batch of URLs with the deployed URL-only (hostname) model.

    Scoring is batched. The per-URL explanation (TreeSHAP signals) is the slow part,
    so for large batches (> ``EXPLAIN_ALL_UP_TO``, or ``explain_all=False``) it is
    computed only for URLs rated phishing or suspicious; legitimate ones get a short
    reason. Small batches (a person checking links) are explained in full.
    """
    b = bundle or load()
    proba, X = _score(urls, b)
    sus = float(b.get("suspicious_at", SUSPICIOUS_AT))  # chosen on validation (v2+)
    verdicts = [_verdict(float(p), b["threshold"], sus) for p in proba]
    # well-known sites (Tranco top 10,000 registered domains, released 2026-10-24 by the
    # owner) never get the "suspicious" warning; a phishing verdict still stands (phishing
    # hosted on a big platform, e.g. a sites / drive page, must be caught)
    top = top_domains()
    for i, v in enumerate(verdicts):
        if v == "suspicious" and registered_domain(split_url(urls[i])[0]) in top:
            verdicts[i] = "legitimate"
    if explain_all is None:
        explain_all = len(urls) <= EXPLAIN_ALL_UP_TO
    rows = [i for i, v in enumerate(verdicts) if explain_all or v != "legitimate"]
    contrib = _contributions(b["model"], X[rows]) if rows else None
    pos = {r: k for k, r in enumerate(rows)}
    model = f"model: {b['family']} on hostname only"
    out = []
    for i, url in enumerate(urls):
        p = float(proba[i])
        verdict = verdicts[i]
        if i in pos:
            row = contrib[pos[i]] if contrib is not None else None
            signals = _model_signals(
                b["feature_names"], row, X[i], verdict != "legitimate"
            )
            why = explain(url, p, verdict, signals, model)
        else:
            why = [f"Verdict: LEGITIMATE (phishing probability {p:.0%}, {model}).",
                   "What to do: " + ACTIONS["legitimate"]]  # fmt: skip
        out.append({
            "url": url,
            "host": split_url(url)[0],
            "phishing_probability": round(p, 4),
            "verdict": verdict,
            "why": why,
            "model": f"{b['family']} ({b['variant']})",
        })  # fmt: skip
    return out


def check_url(url: str, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    return check_urls([url], bundle)[0]


def blind_test(
    urls: Sequence[str], labels: Sequence[int], bundle: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score an unseen labelled URL set with the FROZEN model (label 1 = phishing).

    Nothing is refitted or re-thresholded. Run once, at the very end (owner rule).
    """
    b = bundle or load()
    y = np.asarray(labels, dtype=int)
    proba, _ = _score(list(urls), b)
    t0 = time.perf_counter()
    _score(list(urls)[:5000], b)
    speed = min(len(urls), 5000) / max(time.perf_counter() - t0, 1e-9)
    res: dict[str, Any] = binary_metrics(y, proba, b["threshold"])
    res.update({"n": len(y), "phishing": int(y.sum()), "urls_per_s": round(speed, 1),
                "model": f"{b['family']} ({b['variant']}) v{b.get('version', 1)}"})  # fmt: skip
    return res


# ------------------------------------------------------------ self-improvement


def _as_frame(data: Any) -> pd.DataFrame:
    df = (
        data
        if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data, columns=["url", "label"])
    )
    if not {"url", "label"} <= set(df.columns):
        raise ValueError("new data needs columns 'url' and 'label' (1 = phishing)")
    return df


def improve(new_labelled_data: Any, root: Path | None = None) -> dict[str, Any]:
    """Champion/challenger step with analyst-labelled URLs (label 1 = phishing).

    The challenger is the same family retrained on champion train hosts + the new
    hosts (any host in the fixed validation set is excluded so it stays held out).
    It replaces the champion only if validation F1 is not worse.
    """
    champ = load(root, refresh=True)
    df = _as_frame(new_labelled_data)
    new_hosts, new_y, _ = dedupe_hosts(
        df["url"].astype(str).tolist(), df["label"].astype(int).tolist()
    )
    val_set = set(champ["val_hosts"])
    keep = [i for i, h in enumerate(new_hosts) if h not in val_set]
    hosts = list(champ["train_hosts"]) + [new_hosts[i] for i in keep]
    y = np.concatenate([champ["train_y"], new_y[keep]]).astype(np.int8)
    hosts_u, y_u, _ = dedupe_hosts(hosts, y.tolist())

    X_tr = featurize(hosts_u, champ["mode"])
    X_val = featurize(champ["val_hosts"], champ["mode"])
    y_val = np.asarray(champ["val_y"])
    challenger = make_model(champ["family"], seed=champ.get("seed", 0))
    challenger.fit(X_tr, y_u)

    s_champ = binary_metrics(y_val, predict_proba(champ["model"], X_val))["f1"]
    s_chal = binary_metrics(y_val, predict_proba(challenger, X_val))["f1"]
    sample = list(champ["val_hosts"][:2000])
    sp_champ = common.items_per_second(
        lambda: check_urls(sample, champ), len(sample), 1
    )
    cand = dict(champ, model=challenger, train_hosts=hosts_u, train_y=y_u)
    cand["calibrator"] = calibrate(predict_proba(challenger, X_val), y_val)
    sp_chal = common.items_per_second(lambda: check_urls(sample, cand), len(sample), 1)

    result = common.champion_challenger(
        s_champ, s_chal, sp_champ, sp_chal, "val_f1_phishing"
    )
    result.update({"new_rows": len(df), "new_hosts_used": len(keep)})
    if result["promoted"]:
        cand["version"] = int(champ.get("version", 1)) + 1
        common.save_bundle(AREA, cand, root)
        load(root, refresh=True)
    return result


# ------------------------------------------------------------ research training

DROP_COLS = ("FILENAME", "URL", "Domain", "TLD", "Title", "label", "y")


def leakage_report(df: pd.DataFrame) -> dict[str, Any]:
    """Single-feature AUCs and URL-shape artifacts that trivially encode the label."""
    y = df["y"].to_numpy()
    url = df["URL"].astype(str)
    rest = url.map(lambda u: split_url(u)[1])
    single: dict[str, float] = {}
    for col in df.select_dtypes("number").columns:
        if col in DROP_COLS:
            continue
        auc = roc_auc_score(y, df[col].fillna(0))
        single[col] = round(max(auc, 1 - auc), 4)
    top = dict(sorted(single.items(), key=lambda kv: -kv[1])[:10])

    def rate(mask: pd.Series) -> dict[str, float]:
        m = mask.to_numpy()
        rate_ = float(y[m].mean()) if m.any() else float("nan")
        return {"n": int(m.sum()), "phishing_rate": round(rate_, 4)}

    has_rest = rest.str.len() > 0
    return {
        "single_feature_auc_top10": top,
        "features_auc_over_0.95": [k for k, v in single.items() if v > 0.95],
        "artifacts": {
            "http_scheme": rate(url.str.lower().str.startswith("http://")),
            "url_has_path_or_query": rate(has_rest),
            "ends_with_slash": rate(url.str.endswith("/")),
            "bare_https_www_homepage": rate(
                url.str.lower().str.startswith("https://www.") & ~has_rest
            ),
            "legit_urls_with_path": int(((y == 0) & has_rest.to_numpy()).sum()),
        },
    }


VariantData = tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, int]]


def _variant_data(df: pd.DataFrame, variant: str) -> VariantData:
    """Return X, y, feature names, the row keys (host or URL) and dedupe stats."""
    if variant == "url_host":
        urls = df["URL"].astype(str).tolist()
        hosts, y, conflicts = dedupe_hosts(urls, df["y"].tolist())
        stats = {"rows": len(df), "unique_hosts": len(hosts),
                 "conflicting_hosts_dropped": conflicts}  # fmt: skip
        return featurize(hosts, "host"), y, feature_names("host"), hosts, stats
    d = df.drop_duplicates("URL")
    stats = {"rows": len(df), "unique_urls": len(d)}
    keys = d["URL"].astype(str).tolist()
    y_d = d["y"].to_numpy(np.int8)
    if variant == "url_full":
        return featurize(keys, "full"), y_d, feature_names("full"), keys, stats
    cols = [c for c in d.select_dtypes("number").columns if c not in DROP_COLS]
    return d[cols].fillna(0).to_numpy(np.float32), y_d, cols, keys, stats


Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def _fit_one(
    X: np.ndarray, y: np.ndarray, family: str, seed: int
) -> tuple[Any, dict[str, Any], Split]:
    tr, va, te = split_indices(y, seed)
    model = make_model(family, seed)
    model.fit(X[tr], y[tr])
    res = {
        split: binary_metrics(y[idx], predict_proba(model, X[idx]))
        for split, idx in (("train", tr), ("val", va), ("test", te))
    }
    return model, res, (tr, va, te)


def _summarise(per_seed: list[dict[str, Any]], family: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        metrics = per_seed[0][split]
        out[split] = {
            m: common.mean_sd([r[split][m] for r in per_seed]) for m in metrics
        }
    out["fit"] = common.fit_verdict(
        out["train"]["f1"]["mean"],
        out["val"]["f1"]["mean"],
        tree_model=family != "logreg",
    )
    return out


def _deploy_bundle(
    model: Any, iso: IsotonicRegression, family: str, seed: int, names: list[str],
    keys: list[str], y: np.ndarray, split: Split, metrics: dict[str, float],
) -> dict[str, Any]:  # fmt: skip
    tr, va, _ = split
    return {
        "area": AREA, "variant": "url_host", "mode": "host", "family": family,
        "seed": seed, "version": 1, "model": model, "calibrator": iso,
        "threshold": 0.5, "feature_names": names,
        "train_hosts": [keys[i] for i in tr], "train_y": y[tr],
        "val_hosts": [keys[i] for i in va], "val_y": y[va], "metrics": metrics,
    }  # fmt: skip


def _speeds(urls: list[str], bundle: dict[str, Any]) -> dict[str, float]:
    return {
        "speed_urls_per_s": common.items_per_second(
            lambda: check_urls(urls, bundle), len(urls), 1
        ),
        "speed_score_only_urls_per_s": common.items_per_second(
            lambda: _score(urls, bundle), len(urls), 1
        ),
    }


def train(
    csv_path: Path | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    families: Sequence[str] = FAMILIES,
    variants: Sequence[str] = VARIANTS,
    save: bool = True,
    root: Path | None = None,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run every variant x family x seed, select on validation F1, deploy url_host.

    Returns a JSON-able results dict; plotting arrays are under ``_plot``.
    """
    data = df if df is not None else load_phiusiil(csv_path)
    results: dict[str, Any] = {
        "dataset": "PhiUSIIL (UCI 967)", "positive_class": "phishing",
        "seeds": list(seeds), "split": "stratified 70/15/15",
        "selection": "best mean validation F1 (phishing)",
        "leakage": leakage_report(data), "variants": {},
    }  # fmt: skip
    plot: dict[str, Any] = {}
    for variant in variants:
        X, y, names, keys, stats = _variant_data(data, variant)
        vres: dict[str, Any] = {"data": stats, "n_features": len(names), "families": {}}
        seed0: dict[str, tuple[Any, Split]] = {}
        for family in families:
            per_seed = []
            for seed in seeds:
                model, res, idx = _fit_one(X, y, family, seed)
                per_seed.append(res)
                if seed == seeds[0]:
                    seed0[family] = (model, idx)
            vres["families"][family] = _summarise(per_seed, family)
        best = max(families, key=lambda f: vres["families"][f]["val"]["f1"]["mean"])
        model, split = seed0[best]
        _, va, te = split
        iso = calibrate(predict_proba(model, X[va]), y[va])
        p_te = iso.predict(predict_proba(model, X[te]))
        vres["selected"] = best
        vres["selected_calibrated_test_seed0"] = binary_metrics(y[te], p_te)
        plot[variant] = {"y_test": y[te], "p_test": p_te, "fit": vres["families"][best]}
        if variant == "url_host":
            bundle = _deploy_bundle(model, iso, best, int(seeds[0]), names, keys, y,
                                    split, vres["selected_calibrated_test_seed0"])  # fmt: skip
            vres.update(_speeds([keys[i] for i in te][:20000], bundle))
            if save:
                common.save_bundle(AREA, bundle, root)
                load(root, refresh=True)
        results["variants"][variant] = vres
        del X
    results["_plot"] = plot
    return results
