"""The deployed detector (``deploy.py`` output) and the firewall response action.

Loads ``artifacts/deploy/`` (weights, scaler, columns, top SHAP features, and the
D3 energy threshold calibrated on clean validation data) and scores raw flow
features. Used by the SOC engine, the monitor and the dashboard.
"""

from __future__ import annotations

import ipaddress
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from src.knowledge.attack_kb import explain
from src.models.mlp import SharedMLP
from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
_blocked: set[str] = set()  # IPs with a firewall rule, so a burst adds one

DEPLOY = PROJECT_ROOT / "artifacts" / "deploy"


class StackCascade(torch.nn.Module):
    """Flow detector across networks (owner-approved exception, 2026-10-24).

    A small LightGBM ("stack") reads the deployed scorer's probabilities and those of
    a multi-dataset LightGBM ("generalist": CIC-IDS2017, CIC-DDoS2019, CIC-IDS2018,
    CIC IoT-DIAD 2024) and gives the final class probabilities
    (scripts/improve_flows_v2.py, variant ``stack``). The output is
    ``temperature * log(q)``, so the usual softmax with the deploy temperature returns
    q: BENIGN = 1 - attack score, the attack families share the rest. The detector's
    confirmed-alert threshold becomes the stack's validated threshold. Attributes
    the explanations read come from the wrapped scorer.
    """

    def __init__(self, base: torch.nn.Module, gen: object, stack: object, benign: int, n_classes: int,
                 temperature: float = 1.0) -> None:  # fmt: skip
        super().__init__()
        self.base = base
        self.gen = gen
        self.stack = stack
        self.benign, self.n = int(benign), int(n_classes)
        self.temperature = float(temperature or 1.0)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - delegates to the wrapped scorer
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["base"], name)

    def _full(self, model: object, X: Any) -> Any:  # noqa: ANN401 - numpy in, numpy out
        import numpy as np

        p = model.predict_proba(X)  # type: ignore[attr-defined]
        out = np.zeros((len(X), self.n), dtype=np.float64)
        out[:, np.asarray(model.classes_)] = p  # type: ignore[attr-defined]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import numpy as np

        out = self.base(x)
        pc = torch.softmax(out, 1).numpy().astype(np.float64)
        X = x.numpy()
        pg = self._full(self.gen, X)
        meta = np.hstack([np.log(pc + 1e-6), np.log(pg + 1e-6)]).astype(np.float32)
        q = self._full(self.stack, meta)
        q = np.clip(q, 1e-9, 1.0)
        q = q / q.sum(1, keepdims=True)
        return torch.as_tensor(self.temperature * np.log(q), dtype=out.dtype)


class SpecialistCascade(torch.nn.Module):
    """Champion OR DDoS specialist (owner-approved exception, 2026-10-17).

    The deployed scorer is unchanged for every flow except those it calls BENIGN
    while a small LightGBM trained on CIC-IDS2017 + CIC-DDoS2019 (attack vs benign,
    scripts/improve_flows.py ``cascade``) scores them >= ``threshold``. For those rows
    the output is rewritten so that, after the deploy temperature, DDoS gets the
    specialist's score as its probability and the other classes share the rest in
    their original proportions - the alert format, tiers, calibration and the D3
    energy check (MLP logits) stay as before. Attributes the explanations read
    (``trees``, ``weights``, ``mlp``, ...) come from the wrapped scorer.
    """

    def __init__(self, base: torch.nn.Module, spec: object, threshold: float,
                 benign: int, ddos: int, temperature: float = 1.0) -> None:  # fmt: skip
        super().__init__()
        self.base = base
        self.spec = spec
        self.threshold = float(threshold)
        self.benign, self.ddos = int(benign), int(ddos)
        self.temperature = float(temperature or 1.0)
        self.last_specialist: object = None  # rows the specialist added (bool mask)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - delegates to the wrapped scorer
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["base"], name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import numpy as np

        out = self.base(x)
        cand = (out.argmax(1) == self.benign).numpy()
        fired = np.zeros(len(cand), dtype=bool)
        if cand.any():
            s = self.spec.predict_proba(x[torch.as_tensor(cand)].numpy())[:, 1]  # type: ignore[attr-defined]
            hit = s >= self.threshold
            if hit.any():
                rows = np.flatnonzero(cand)[hit]
                fired[rows] = True
                t = self.temperature
                p = torch.softmax(out[rows] / t, 1)
                q = p.clone()
                q[:, self.ddos] = 0.0
                q = q / q.sum(1, keepdim=True).clamp_min(1e-12) * torch.as_tensor(1.0 - s[hit])[:, None]
                q[:, self.ddos] = torch.as_tensor(s[hit], dtype=q.dtype)
                out = out.clone()
                out[rows] = t * torch.log(q.clamp_min(1e-12)).to(out.dtype)
        self.last_specialist = fired
        return out


class Detector:
    """Frozen detector over the deployment artifacts."""

    def __init__(self, deploy_dir: Path = DEPLOY) -> None:
        cols = json.loads((deploy_dir / "columns.json").read_text(encoding="utf-8"))
        self.feature_names: list[str] = cols["feature_names"]
        self.label_names: list[str] = cols["label_names"]
        # D3 runtime detector: energy above this means "unlike anything seen in
        # training" (possible evasion / manipulated input). None on old artifacts.
        self.energy_threshold: float | None = cols.get("energy_threshold")
        # validation-fitted confidence temperature (1.0 = uncalibrated, old artifacts)
        self.temperature: float = float(cols.get("temperature", 1.0))
        # tier-2 alert threshold on attack probability (~1% false alarms on
        # validation); None on older artifacts -> every alert is "confirmed"
        self.alert_threshold_high: float | None = cols.get("alert_threshold_high")
        # live-input preprocessing (src/data/robust.py): training medians fill missing
        # values; ranges flag input in other units. None on older artifacts.
        self.input_stats: dict[str, list[float]] | None = cols.get("input_stats")
        self.last_input_check: dict[str, object] | None = None
        with open(deploy_dir / "scaler.pkl", "rb") as fh:
            self.scaler = pickle.load(fh)  # noqa: S301 - our own artifact, path-guarded
        self.top_features: dict[str, list[str]] = json.loads(
            (deploy_dir / "top_features.json").read_text(encoding="utf-8")
        )
        self.model = SharedMLP(
            n_features=len(self.feature_names),
            n_classes=len(self.label_names),
            hidden_layers=cols["hidden_layers"],
            hidden_width=cols["hidden_width"],
            dropout=cols["dropout"],
            norm=cols.get("norm", "none"),  # older artifacts: no norm layers
        )
        self.model.load_state_dict(
            torch.load(deploy_dir / "model.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()
        # per-class confirmed-alert thresholds (rare classes stricter); None = global
        self.alert_threshold_by_class: dict[str, float] | None = cols.get(
            "alert_threshold_by_class"
        )
        # split-conformal threshold: 2+ plausible classes -> "model unsure"
        self.conformal_qhat: float | None = cols.get("conformal_qhat")
        # ensemble deployment: MLP + tree members averaged (src/models/ensemble.py);
        # the MLP alone still drives the D3 energy detector and SHAP indicators
        self.ensemble: torch.nn.Module | None = None
        if cols.get("ensemble"):
            import joblib

            from src.models.ensemble import EnsembleModel

            trees = joblib.load(deploy_dir / "ensemble.pkl")  # our own artifact
            self.ensemble = EnsembleModel(
                self.model, trees, cols["ensemble"], cols.get("class_log_bias")
            ).eval()
        # owner-approved exception 2026-10-17: champion OR CIC-DDoS2019 specialist
        # (adds DDoS alerts on flows the champion calls benign; see SpecialistCascade)
        spec = cols.get("ddos_specialist")
        if spec and (deploy_dir / spec["file"]).is_file():
            import joblib

            b = joblib.load(deploy_dir / spec["file"])  # our own artifact
            self.ensemble = SpecialistCascade(
                self.ensemble if self.ensemble is not None else self.model, b["model"],
                float(spec["threshold"]), self.label_names.index("BENIGN"),
                self.label_names.index("DDoS"), self.temperature,
            ).eval()  # fmt: skip
        # owner-approved exception 2026-10-24: stack over the scorer + a multi-dataset
        # generalist (flows from other networks; see StackCascade)
        st = cols.get("flow_stack")
        if st and (deploy_dir / st["stack_file"]).is_file() and (deploy_dir / st["generalist_file"]).is_file():
            import joblib

            g = joblib.load(deploy_dir / st["generalist_file"])  # our own artifacts
            s = joblib.load(deploy_dir / st["stack_file"])
            self.ensemble = StackCascade(
                self.ensemble if self.ensemble is not None else self.model, g["model"], s["stack"],
                self.label_names.index("BENIGN"), len(self.label_names), self.temperature,
            ).eval()  # fmt: skip
            self.alert_threshold_high = float(st["threshold"])
            self.alert_threshold_by_class = None
        self.last_mlp_logits = None
        self.last_X = None
        self.last_Xs = None

    def reason(self, cls: str) -> str:
        """One-line 'why flagged' from the attack KB + the model's indicators."""
        exp = explain(cls)
        what = exp["what"] if exp else cls
        feats = ", ".join(self.top_features.get(cls, [])) or "n/a"
        return f"{what}  [key indicators: {feats}]"


def firewall_block(ip: str) -> bool:
    """Block ``ip`` inbound: Windows Firewall (Administrator), pf on macOS or
    nftables / iptables on Linux (root). Public addresses only."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        _log.warning("firewall: refusing invalid IP %r", ip)
        return False
    # never cut this machine off its own LAN, gateway or DNS on a misclassification
    if not addr.is_global:
        _log.warning("firewall: refusing to block non-public address %s", addr)
        return False
    if str(addr) in _blocked:
        return True
    if sys.platform != "win32":  # macOS (pf) / Linux (nftables / iptables), needs root
        from src.soc import osops

        ok, why = osops.block_ip(str(addr))
        if ok:
            _log.info("firewall: blocked %s", addr)
            _blocked.add(str(addr))
        else:
            _log.warning("firewall: could not block %s (%s)", addr, why)
        return ok
    name = f"name=SplitShield-block-{addr}"
    show = ["netsh", "advfirewall", "firewall", "show", "rule", name]
    if subprocess.run(show, capture_output=True, timeout=30).returncode == 0:
        _blocked.add(str(addr))  # rule left by an earlier run
        return True
    cmd = [
        "netsh",
        "advfirewall",
        "firewall",
        "add",
        "rule",
        name,
        "dir=in",
        "action=block",
        f"remoteip={addr}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        _log.info("firewall: blocked %s", addr)
        _blocked.add(str(addr))
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as exc:
        _log.warning(
            "firewall: block failed for %s (%s); run as Administrator", addr, exc
        )
        return False
