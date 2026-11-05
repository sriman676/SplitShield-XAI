"""T4 — client-attribution probe (02_THREAT_MODEL T4, Stage 4c).

Which adversary: T4, the honest-but-curious server, membership variant. It wants
to know *which client* an intercepted cut-layer activation came from.

What it knows and where: exactly what T3 knows — ``ServerNet``, the ``ClientNet``
architecture, and the activations it legitimately receives. Crucially, in
sequential split learning the server processes one client per turn, so it
observes (activation, source-client) pairs *for free* — the training labels for
this probe are not stolen, they are an inherent side effect of the protocol. The
knowledge enters where ``activations_by_client`` is built from each client's own
data below. It does NOT need ClientNet weights beyond running the forward pass it
already runs to serve the client.

Control (§3.3): the majority-client constant predictor, reported as **balanced
accuracy**. A server that always guesses the busiest client scores high plain
accuracy while learning nothing; balanced accuracy against the prior is the
honest floor.

Strongest version NOT implemented: an active server that perturbs returned
gradients to make clients' activations more separable. This passive probe is a
lower bound (11_LIMITATIONS §1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn

from src.eval.baselines import class_prior_balanced_accuracy
from src.eval.metrics import metric_entry
from utils.logging_config import get_logger

_log = get_logger(__name__)


@torch.no_grad()
def cut_activations(client: nn.Module, X_scaled: np.ndarray, device: str) -> np.ndarray:
    """Cut-layer activations the server observes for ``X`` (flattened per sample)."""
    client.eval()
    a = client(torch.as_tensor(X_scaled, dtype=torch.float32, device=device))
    return a.reshape(len(X_scaled), -1).cpu().numpy()


MAX_ITER_START = 1000
MAX_ITER_CAP = 64_000


def fit_converged(
    A: np.ndarray, y: np.ndarray, seed: int, start: int = MAX_ITER_START
) -> Any:
    """Standardized logistic probe whose iteration limit grows automatically:
    doubled from ``start`` while lbfgs reports non-convergence, up to
    ``MAX_ITER_CAP``. The limit used is kept on the model (``max_iter_used``)."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = start
    while True:
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=n, random_state=seed)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                clf.fit(A, y)
                clf.max_iter_used = n
                return clf
            except ConvergenceWarning:
                if n >= MAX_ITER_CAP:
                    break
        n *= 2
    _log.warning("attribution probe not converged at max_iter=%d", n)
    clf.fit(A, y)  # best effort at the cap, warning logged once
    clf.max_iter_used = n
    return clf


def client_attribution(
    client: nn.Module,
    X_by_client: list[np.ndarray],
    device: str,
    seed: int,
    converge: bool = False,
) -> dict[str, Any]:
    """Train an activation->client probe and score it against the prior (§3.3).

    Args:
        client: The trained ``ClientNet`` (produces the observed activations).
        X_by_client: Each client's scaled feature matrix.
        device: Torch device.
        seed: Attack randomness (§5).

    Returns:
        A metrics entry: balanced accuracy with its class-prior control, plus the
        per-client support.
    """
    acts, labels = [], []
    for cid, Xc in enumerate(X_by_client):
        if len(Xc) == 0:
            continue
        acts.append(cut_activations(client, Xc, device))
        labels.append(np.full(len(Xc), cid))
    A = np.concatenate(acts)
    y = np.concatenate(labels)

    n_clients = len(X_by_client)
    prior = class_prior_balanced_accuracy(y, n_clients)

    # Degenerate case (one client has all rows): probe is undefined; report prior.
    if len(np.unique(y)) < 2:
        return metric_entry(
            "attack_type_balanced_acc",
            prior,
            control={"name": "class_prior", "balanced_accuracy": prior},
            per_client_support={int(c): int(np.sum(y == c)) for c in np.unique(y)},
            note="single-client activations; probe undefined",
        )

    A_tr, A_te, y_tr, y_te = train_test_split(
        A, y, test_size=0.3, random_state=seed, stratify=y
    )
    if converge:  # corrected follow-up: standardized probe, run to convergence
        clf: Any = fit_converged(A_tr, y_tr, seed)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(A_tr, y_tr)
    from sklearn.metrics import balanced_accuracy_score

    bacc = float(balanced_accuracy_score(y_te, clf.predict(A_te)))

    extra = {"probe_max_iter": int(clf.max_iter_used)} if converge else {}
    return metric_entry(
        "attack_type_balanced_acc",
        bacc,
        control={"name": "class_prior", "balanced_accuracy": prior},
        per_client_support={int(c): int(np.sum(y == c)) for c in range(n_clients)},
        **extra,
    )
