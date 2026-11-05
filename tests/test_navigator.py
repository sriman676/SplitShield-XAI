"""ATT&CK Navigator layer export (src/soc/navigator.py)."""

from __future__ import annotations

from src.soc.navigator import layer


def test_layer_counts_techniques_per_incident() -> None:
    incs = [
        {"src_ip": "198.51.100.1", "severity": 6,
         "techniques": [{"id": "T1110"}, {"id": "T1078"}]},
        {"src_ip": "198.51.100.2", "severity": 9, "techniques": [{"id": "T1110"}]},
    ]  # fmt: skip
    lay = layer(incs)
    assert lay["domain"] == "enterprise-attack" and lay["versions"]["layer"] == "4.5"
    t = {x["techniqueID"]: x for x in lay["techniques"]}
    assert t["T1110"]["score"] == 2 and t["T1078"]["score"] == 1
    assert "max severity 9/10" in t["T1110"]["comment"]
    assert "198.51.100.2" in t["T1110"]["comment"]
    assert lay["gradient"]["maxValue"] == 2
    assert layer([])["techniques"] == []
