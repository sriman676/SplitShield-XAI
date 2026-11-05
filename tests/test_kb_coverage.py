"""Every class any deployed detector can output is explained and ATT&CK-mapped."""

from __future__ import annotations

import pytest

from src.data.labels import BINARY_ORDER, CIOT_ORDER, FAMILY_ORDER
from src.knowledge.attack_kb import explain
from src.soc.correlate import CLASS_TECHNIQUES, TECHNIQUES

ALL = sorted({*FAMILY_ORDER, *CIOT_ORDER, *BINARY_ORDER})


@pytest.mark.parametrize("cls", ALL)
def test_class_is_explained(cls: str) -> None:
    e = explain(cls)
    assert e is not None and e["what"] and e["defenses"]
    if cls != "BENIGN":
        assert e["attacker"] != "n/a"  # a mechanism, not a placeholder


@pytest.mark.parametrize("cls", [c for c in ALL if c not in ("BENIGN", "ATTACK")])
def test_named_attack_maps_to_known_attack_technique(cls: str) -> None:
    # the generic binary ATTACK has no family, so no technique is claimed for it
    techs = CLASS_TECHNIQUES.get(cls)
    assert techs and all(t in TECHNIQUES for t in techs)
