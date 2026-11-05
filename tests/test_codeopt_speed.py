"""Search and chat speed are part of the nightly profile and guard code proposals
(owner 2026-10-16): measured on a temporary practice store, never the real one."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.soc import codeopt, engine


def test_search_and_chat_probes_use_a_temporary_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.soc import evolve

    if not evolve._bench():  # ships in the models zip (Releases), not in the repository
        pytest.skip("evolve benchmark missing")
    monkeypatch.setattr(codeopt, "PROBE_EVENTS", 3000)
    real = engine.EVENTS
    assert codeopt.workload_search() > 0
    assert codeopt.workload_chat() > 0
    assert engine.EVENTS == real and not Path(real).exists()  # untouched, restored
    assert {"search", "chat"} <= set(codeopt.workloads())


def test_a_patch_that_slows_search_or_chat_is_not_offered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def bench(root: Path, workload: str = "rules", repeats: int = 3) -> float:
        calls.append(workload)
        patched = (root / "x.py").is_file() and (root / "x.py").read_text() == "new"
        if workload == "rules":
            return 120.0 if patched else 100.0  # +20 % on the rules
        if workload == "chat":
            return 80.0 if patched else 100.0  # but -20 % on chat
        return 100.0

    class Box:
        def __enter__(self) -> Path:
            return tmp_path

        def __exit__(self, *a: object) -> None:
            return None

    monkeypatch.setattr(codeopt, "_sandbox", Box)
    monkeypatch.setattr(codeopt, "_bench_in", bench)
    monkeypatch.setattr(codeopt, "_tests_in", lambda root: (True, "1 passed"))
    res = codeopt.trial("x.py", "new")
    assert res["gain"] == pytest.approx(0.2) and res["tests_passed"]
    assert res["guard_ok"] is False and res["guard"]["chat"] == [100.0, 80.0]
    assert {"search", "chat"} <= set(calls)
