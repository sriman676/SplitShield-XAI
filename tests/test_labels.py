"""Label mapping: raw 15-class strings collapse to the right 9 families."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import labels as L


def test_family_mapping_edge_cases() -> None:
    assert L.to_family("BENIGN") == "BENIGN"
    assert L.to_family("DoS Hulk") == "DoS"
    assert L.to_family("DDoS") == "DDoS"  # checked before 'dos'
    assert L.to_family("Web Attack \x96 Brute Force") == "WebAttack"  # web before brute
    assert L.to_family("FTP-Patator") == "BruteForce"
    assert L.to_family("SSH-Patator") == "BruteForce"
    assert L.to_family("PortScan") == "PortScan"
    assert L.to_family("Infilteration") == "Infiltration"  # archive misspelling


def test_unknown_label_fails_loud() -> None:
    with pytest.raises(ValueError):
        L.to_family("Totally Novel Attack")


def test_encode_is_fixed_order() -> None:
    s = pd.Series(["BENIGN", "Heartbleed", "DoS"])
    assert list(L.encode(s)) == [0, 8, 1]


def test_binary_collapse() -> None:
    s = pd.Series(["BENIGN", "DoS", "Bot"])
    assert list(L.to_binary(s)) == ["BENIGN", "ATTACK", "ATTACK"]


def test_ciot_mapping_edge_cases() -> None:
    assert L.to_ciot("BENIGN") == "BENIGN"
    assert L.to_ciot("DDoS-ICMP_Flood") == "DDoS"  # checked before 'dos'
    assert L.to_ciot("DoS-SYN_Flood") == "DoS"
    assert L.to_ciot("Mirai-greeth_flood") == "Mirai"  # mirai before ddos/dos
    assert L.to_ciot("Recon-PortScan") == "Recon"
    assert L.to_ciot("VulnerabilityScan") == "Recon"  # 'scan' -> Recon
    assert L.to_ciot("DNS_Spoofing") == "Spoofing"
    assert L.to_ciot("MITM-ArpSpoofing") == "Spoofing"
    assert L.to_ciot("SqlInjection") == "WebBased"
    assert L.to_ciot("CommandInjection") == "WebBased"
    assert L.to_ciot("Backdoor_Malware") == "WebBased"
    assert L.to_ciot("BrowserHijacking") == "WebBased"
    assert L.to_ciot("Uploading_Attack") == "WebBased"
    assert L.to_ciot("XSS") == "WebBased"
    assert L.to_ciot("DictionaryBruteForce") == "BruteForce"


def test_ciot_unknown_label_fails_loud() -> None:
    with pytest.raises(ValueError):
        L.to_ciot("Totally Novel IoT Attack")


def test_ciot_encode_is_fixed_order() -> None:
    s = pd.Series(["BENIGN", "BruteForce", "DoS"])
    assert L.order_for("ciot")[0] == "BENIGN"
    assert list(L.encode_scheme(s, "ciot")) == [0, 7, 2]


def test_resolve_labels_ciot() -> None:
    s = pd.Series(["BENIGN", "DDoS-UDP_Flood", "Mirai-udpplain"])
    assert list(L.resolve_labels(s, "ciot")) == ["BENIGN", "DDoS", "Mirai"]
