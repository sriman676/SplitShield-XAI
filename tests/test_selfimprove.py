"""Self-improvement loop: search space, one fixture pass, defence KB."""

from __future__ import annotations

from config.defense_kb import DEFENSE_KB, defense_for
from src.selfimprove.rl import GradientBanditController
from src.selfimprove.search import sample_archs


def test_rl_controller_learns_best_action() -> None:
    # action 2 always pays best -> preference/greedy pick should converge to it
    c = GradientBanditController(n_actions=3, seed=0, lr=0.3)
    rewards = {0: 0.1, 1: 0.3, 2: 0.9}
    for _ in range(200):
        a = c.select()
        c.update(a, rewards[a])
    assert c.best_action() == 2
    assert c.probs()[2] == max(c.probs())


def test_sample_archs_deterministic_and_bounded() -> None:
    a = sample_archs(5, seed=0, max_width=64, max_layers=3)
    b = sample_archs(5, seed=0, max_width=64, max_layers=3)
    assert a == b  # seeded
    assert all(c["hidden_width"] <= 64 and c["hidden_layers"] <= 3 for c in a)
    # sorted smallest-first by the param proxy
    proxy = [int(c["hidden_width"]) * int(c["hidden_layers"]) for c in a]
    assert proxy == sorted(proxy)


def test_defense_kb_covers_all_schemes() -> None:
    for cls in (
        "DoS",
        "DDoS",
        "PortScan",
        "BruteForce",
        "WebAttack",
        "Bot",
        "Infiltration",
        "Heartbleed",
        "Mirai",
        "Recon",
        "Spoofing",
        "WebBased",
    ):
        assert cls in DEFENSE_KB and defense_for(cls)["defenses"]
    assert "add it" in str(defense_for("NoSuchClass")["summary"])  # unknown placeholder


def test_improve_runs_on_fixture(tmp_path, monkeypatch) -> None:
    # isolate artifacts/leaderboard writes to a temp dir
    import src.selfimprove.loop as loop

    monkeypatch.setattr(loop, "ARTIFACTS", tmp_path / "best")
    monkeypatch.setattr(loop, "GENERATED", tmp_path / "generated")
    monkeypatch.setattr(loop, "LEADERBOARD", tmp_path / "leaderboard.json")
    monkeypatch.setattr(loop, "AUTO_YAML", tmp_path / "auto.yaml")
    monkeypatch.setattr(loop, "_git_commit", lambda *a, **k: None)

    row = loop.improve(dataset="fixture", n_trials=2, fast=True, device="cpu")
    assert {
        "macro_f1",
        "size_bytes",
        "promoted",
        "gate_ok",
        "valid",
        "validation",
    } <= set(row)
    assert (tmp_path / "leaderboard.json").exists()
    assert 0.0 <= row["macro_f1"] <= 1.0
    # RL -> validation sequence ran: the gate reports its individual checks
    assert "checks" in row["validation"]
    # promotion implies the output actually validated
    assert (not row["promoted"]) or row["valid"]


def test_extension_hooks_real() -> None:
    # Hooks are real (minimal) implementations now: USL returns a warm-started
    # state_dict, continual_update returns the same model object it refined.
    from types import SimpleNamespace

    import numpy as np

    from src.models.mlp import SharedMLP
    from src.selfimprove.extensions import continual_update, unsupervised_pretrain

    n, f, c = 120, 6, 3
    rng = np.random.default_rng(0)
    data = SimpleNamespace(
        X_train=rng.standard_normal((n, f)).astype(np.float32),
        y_train=rng.integers(0, c, size=n),
        label_names=[f"c{i}" for i in range(c)],
        class_weights=None,
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            hidden_width=16, hidden_layers=2, dropout=0.0, norm="none"
        ),
        training=SimpleNamespace(lr=1e-3),
        runtime=SimpleNamespace(fast=True, batch_size=64),
        seed=0,
    )
    sd = unsupervised_pretrain(data, cfg, "cpu")  # type: ignore[arg-type]
    assert sd is not None and any("blocks" in k for k in sd)
    # empty data -> None (i.e. "supervised from scratch", the old behaviour)
    assert (
        unsupervised_pretrain(SimpleNamespace(X_train=np.empty((0, f))), cfg, "cpu")  # type: ignore[arg-type]
        is None
    )

    model = SharedMLP(n_features=f, n_classes=c, hidden_layers=2, hidden_width=16)
    assert continual_update(model, data, cfg, "cpu") is model  # type: ignore[arg-type]


def test_usl_hook_is_called_when_enabled(tmp_path, monkeypatch) -> None:
    import src.selfimprove.loop as loop

    for attr, val in (
        ("ARTIFACTS", tmp_path / "best"),
        ("GENERATED", tmp_path / "generated"),
        ("LEADERBOARD", tmp_path / "leaderboard.json"),
        ("AUTO_YAML", tmp_path / "auto.yaml"),
    ):
        monkeypatch.setattr(loop, attr, val)
    monkeypatch.setattr(loop, "_git_commit", lambda *a, **k: None)

    called = {"usl": 0}
    monkeypatch.setattr(
        loop, "unsupervised_pretrain", lambda *a, **k: called.__setitem__("usl", 1)
    )
    loop.improve(dataset="fixture", n_trials=1, fast=True, device="cpu", use_usl=True)
    assert called["usl"] == 1  # the wired hook fires automatically when enabled


def test_experiment_for_dataset_uses_real_caps() -> None:
    # self-improve/deploy on real data must not inherit smoke.yaml's 200-row cap
    from config.base import build_config, experiment_for_dataset

    real = build_config(experiment_for_dataset("cicids2017-improved"), seed=0)
    assert real.dataset == "cicids2017-improved"
    assert real.subsample.class_cap == 60_000 and real.model.hidden_width == 128
    nf = build_config(experiment_for_dataset("nf-unsw-nb15-v2"), seed=0)  # no YAML
    assert nf.dataset == "nf-unsw-nb15-v2" and nf.subsample.class_cap == 60_000
    fx = build_config(experiment_for_dataset("cicids2017-improved", fast=True), seed=0)
    assert fx.subsample.class_cap == 200  # --fast keeps the smoke caps


def test_promotion_state_is_per_dataset(tmp_path, monkeypatch) -> None:
    import json

    import src.selfimprove.loop as loop

    monkeypatch.setattr(loop, "LEADERBOARD", tmp_path / "leaderboard.json")
    monkeypatch.setattr(loop, "AUTO_YAML", tmp_path / "auto.yaml")
    rows = [{"dataset": "a", "promoted": True, "val_macro_f1": 0.99}]
    (tmp_path / "leaderboard.json").write_text(json.dumps(rows), "utf-8")
    # a strong model on dataset "a" must not block promotion on dataset "b"
    assert loop._prev_best_f1("b") == -1.0
    assert loop._prev_best_f1("a") == 0.99
    best = {"hidden_layers": 2, "hidden_width": 32, "dropout": 0.1}
    best |= {"macro_f1": 0.5, "val_macro_f1": 0.5, "size_bytes": 10}
    loop._write_auto_yaml(best, "a")
    loop._write_auto_yaml({**best, "hidden_width": 64}, "b")
    assert loop.auto_model("a")["hidden_width"] == 32  # type: ignore[index]
    assert loop.auto_model("b")["hidden_width"] == 64  # type: ignore[index]
    assert loop.auto_model("c") is None
    assert loop.best_dir("a") != loop.best_dir("b")
