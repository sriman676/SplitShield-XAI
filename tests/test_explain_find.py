"""ATT&CK technique lookup from a question: whole words only (2026-10-15 regression:
"at" inside "what" mapped questions to T1053.002 "At")."""

import pytest

from src.knowledge.explain import find


@pytest.mark.parametrize(
    ("question", "technique"),
    [
        ("what are the project results", None),
        ("what is credential dumping", "T1003"),
        ("what is a brute force attack", "T1110"),
        ("explain T1059", "T1059"),
        ("what is phishing", "T1566"),
        ("explain scheduled task at", "T1053.002"),  # the real "At" still resolves
    ],
)
def test_find_matches_whole_words(question: str, technique: str | None) -> None:
    assert find(question) == technique
