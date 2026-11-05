"""Label schema: 15 raw classes -> 9 attack families -> binary (04_DATA §3).

FAMILY_ORDER fixes the integer encoding so a class index means the same thing in
every stage. This is load-bearing for the fixed-class explanation-drift rule
(01_RESEARCH_DESIGN §4.4): comparing attributions across runs requires a stable
class index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed index order. Do not reorder — downstream class indices depend on it.
FAMILY_ORDER: tuple[str, ...] = (
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration",
    "Heartbleed",
)
FAMILY_TO_IDX: dict[str, int] = {f: i for i, f in enumerate(FAMILY_ORDER)}

# Ordered keyword rules (first match wins). Order matters: 'ddos' before 'dos',
# 'web' before 'brute' (Web Attack Brute Force is a WebAttack, not BruteForce).
_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("benign", "normal"), "BENIGN"),
    (("ddos",), "DDoS"),
    (("dos",), "DoS"),
    (("portscan", "port scan"), "PortScan"),
    (("web", "xss", "sql"), "WebAttack"),
    (("patator", "bruteforce", "brute force", "brute-force"), "BruteForce"),
    (("bot",), "Bot"),
    (("infiltration", "infilteration"), "Infiltration"),
    (("heartbleed",), "Heartbleed"),
)


def to_family(label: str) -> str:
    """Map one raw label string to its 9-class family. Fails loud on unknown."""
    norm = str(label).strip().lower()
    for keywords, family in _RULES:
        if any(k in norm for k in keywords):
            return family
    raise ValueError(f"Unrecognised label {label!r}; add a rule in labels.py")


def to_family_series(labels: pd.Series) -> pd.Series:
    """Vectorised ``to_family`` over a pandas Series (maps unique values once)."""
    mapping = {v: to_family(v) for v in labels.unique()}
    return labels.map(mapping)


def encode(families: pd.Series) -> np.ndarray:
    """Encode family strings to their fixed FAMILY_ORDER integer indices."""
    return families.map(FAMILY_TO_IDX).to_numpy(dtype=np.int64)


def to_binary(families: pd.Series) -> pd.Series:
    """Collapse families to BENIGN vs ATTACK (secondary schema)."""
    return families.where(families == "BENIGN", "ATTACK")


# --- label schemes ----------------------------------------------------------
#
# 'family' is the 9-class default (CIC). 'binary' is BENIGN vs ATTACK, used for
# cross-dataset transfer (E8) where taxonomies do NOT align: NF-UNSW-NB15 attack
# categories (Fuzzers/Exploits/Generic/…) have no CIC-family equivalent, so the
# only label space the NetFlow-v2 datasets share is benign-vs-attack (04_DATA §3).

BINARY_ORDER: tuple[str, ...] = ("BENIGN", "ATTACK")

_BENIGN_TOKENS: frozenset[str] = frozenset({"0", "0.0", "benign", "normal"})


def resolve_labels(labels: pd.Series, scheme: str = "family") -> pd.Series:
    """Map raw labels to the active scheme's class strings ('family'|'binary').

    Binary accepts either a 0/1 indicator column or a category string; anything
    not a benign token becomes ATTACK. Family delegates to ``to_family_series``
    (fails loud on unknown labels).
    """
    if scheme == "binary":
        s = labels.astype(str).str.strip().str.lower()
        return s.map(lambda v: "BENIGN" if v in _BENIGN_TOKENS else "ATTACK")
    if scheme == "ciot":
        return labels.map({v: to_ciot(v) for v in labels.unique()})
    if scheme == "family":
        return to_family_series(labels)
    if scheme in NATIVE_ORDERS:
        return labels.map({v: to_native(v, scheme) for v in labels.unique()})
    raise ValueError(f"Unknown label scheme {scheme!r}")


# CIC-IoT-2023 native taxonomy: 33 modern IoT attacks grouped into 7 classes +
# BENIGN (04_DATA §3, recency arm). Own scheme because its attacks (Mirai,
# Spoofing, IoT DDoS variants) don't fit the enterprise CIC 9-class family schema.
CIOT_ORDER: tuple[str, ...] = (
    "BENIGN",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "WebBased",
    "BruteForce",
)

# Ordered keyword rules (first match wins). 'mirai'/'ddos' before 'dos'; scan-like
# recon before web; injection/xss/backdoor/uploading -> WebBased.
_CIOT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("benign",), "BENIGN"),
    (("mirai",), "Mirai"),
    (("ddos",), "DDoS"),
    (("dos",), "DoS"),
    (("spoofing", "mitm"), "Spoofing"),
    (("recon", "scan", "hostdiscovery", "pingsweep"), "Recon"),
    (("dictionary", "bruteforce", "brute force"), "BruteForce"),
    (
        (
            "web",
            "xss",
            "sql",
            "injection",
            "backdoor",
            "malware",
            "browserhijack",
            "uploading",
        ),
        "WebBased",
    ),
)


def to_ciot(label: str) -> str:
    """Map one raw CIC-IoT-2023 label to its 8-class scheme. Fails loud on unknown."""
    norm = str(label).strip().lower()
    for keywords, cls in _CIOT_RULES:
        if any(k in norm for k in keywords):
            return cls
    raise ValueError(f"Unrecognised CIC-IoT label {label!r}; add a rule in labels.py")


# Extra datasets (owner 2026-10-14, tested outside the registered design): each
# keeps its own native taxonomy. Raw labels are matched after lower-casing and
# removing everything but letters/digits ("UDP-lag" == "UDPLag", "DrDoS_LDAP" ->
# "drdosldap" -> LDAP via the alias table).
NATIVE_ORDERS: dict[str, tuple[str, ...]] = {
    "ton": (
        "BENIGN", "Backdoor", "DDoS", "DoS", "Injection", "MITM", "Password",
        "Ransomware", "Scanning", "XSS",
    ),
    "unsw": (
        "BENIGN", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
        "Reconnaissance", "Shellcode", "Worms",
    ),
    "ddos19": (
        "BENIGN", "DNS", "LDAP", "MSSQL", "NTP", "NetBIOS", "Portmap", "SNMP", "SSDP",
        "Syn", "TFTP", "UDP", "UDPLag", "WebDDoS",
    ),
}  # fmt: skip
_NATIVE_ALIASES: dict[str, str] = {"normal": "BENIGN", "0": "BENIGN"}


def _norm(label: object) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def to_native(label: object, scheme: str) -> str:
    """Raw label -> class of a native scheme. Fails loud on unknown labels."""
    n = _norm(label)
    n = n[len("drdos") :] if n.startswith("drdos") else n  # CIC-DDoS2019 reflection
    if n in _NATIVE_ALIASES:
        return _NATIVE_ALIASES[n]
    for cls in NATIVE_ORDERS[scheme]:
        if _norm(cls) == n:
            return cls
    raise ValueError(f"Unrecognised {scheme!r} label {label!r}; add it in labels.py")


def order_for(scheme: str) -> tuple[str, ...]:
    """Class order (fixed integer encoding) for the given scheme."""
    if scheme in NATIVE_ORDERS:
        return NATIVE_ORDERS[scheme]
    return {"binary": BINARY_ORDER, "ciot": CIOT_ORDER}.get(scheme, FAMILY_ORDER)


def encode_scheme(classes: pd.Series, scheme: str) -> np.ndarray:
    """Encode class strings to their fixed integer indices for the scheme."""
    idx = {f: i for i, f in enumerate(order_for(scheme))}
    return classes.map(idx).to_numpy(dtype=np.int64)
