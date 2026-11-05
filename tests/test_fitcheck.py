"""Automatic over/underfitting check and the deploy-time fit doctor."""

from __future__ import annotations

from src.eval.fitcheck import diagnose


def _h(vals: list[float], losses: list[float] | None = None) -> list[dict[str, float]]:
    losses = losses or [0.3] * len(vals)
    return [
        {"val_f1": v, "val_loss": loss} for v, loss in zip(vals, losses, strict=True)
    ]


def test_good_fit() -> None:
    f = diagnose(_h([0.80, 0.86, 0.88, 0.885, 0.886, 0.886]), 0.89, 50, True)
    assert f["verdict"] == "good fit" and not f["tweaks"]


def test_overfitting() -> None:
    f = diagnose(_h([0.80, 0.85, 0.86]), 0.97, 50, True)
    assert f["verdict"] == "overfitting" and "regularize" in f["tweaks"]


def test_under_trained_and_noisy_stop() -> None:
    rising = [0.5 + 0.01 * i for i in range(20)]
    f = diagnose(_h(rising), 0.70, 20, False)  # still improving at max_epochs
    assert f["verdict"] == "underfitting" and "train_longer" in f["tweaks"]
    noisy = [0.85, 0.89, 0.86, 0.92, 0.87, 0.90, 0.86]  # the CIC-IDS2017 case
    g = diagnose(_h(noisy), 0.90, 50, True)
    assert g["verdict"] == "underfitting" and any("noisy" in x for x in g["findings"])


def test_overconfident() -> None:
    f = diagnose(_h([0.86, 0.88, 0.89, 0.89], [0.28, 0.30, 0.38, 0.43]), 0.90, 50, True)
    assert f["verdict"] == "overconfident" and f["tweaks"] == ["calibrate"]


def test_training_reports_fit_and_deploy_runs_fit_doctor() -> None:
    from config.base import build_config, load_experiment
    from src.data.preprocess import preprocess_frame
    from src.data.sources import get_spec, load_raw
    from src.training.centralized import train_centralized

    spec = get_spec("fixture")
    cfg = build_config(load_experiment("smoke.yaml"), seed=0, fast=True)
    data = preprocess_frame(load_raw(spec), spec, cfg, seed=0)
    _, _, met = train_centralized(data, cfg, "cpu")
    assert met["fit"]["verdict"] in {
        "good fit",
        "overfitting",
        "underfitting",
        "overconfident",
    }

    import deploy

    fit = {"verdict": "underfitting", "tweaks": ["train_longer"]}
    model, _, _ = train_centralized(data, cfg, "cpu")
    c2, _, log = deploy._fit_doctor(
        data, cfg, model, fit, "cpu", len(data.label_names), rounds=1
    )
    assert len(log) == 1 and log[0]["tweaks"] == ["train_longer"]
    if log[0]["kept"]:
        assert c2.training.patience == cfg.training.patience * 2


def test_high_bias_against_reference() -> None:
    from src.eval.fitcheck import with_reference

    f = diagnose(_h([0.58, 0.60, 0.61, 0.612, 0.612]), 0.615, 50, True)
    assert f["verdict"] == "good fit"  # no gap: the rules alone miss it (R1 case)
    g = with_reference(f, 0.688, "Random Forest")
    assert g["verdict"] == "underfitting" and "train_longer" in g["tweaks"]
    assert with_reference(f, 0.63, "Random Forest") is f  # within 0.05: fine
