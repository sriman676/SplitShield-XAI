"""BLOCKING gate: chain(split(d))(x) == mlp(x) for every d (08_ENGINEERING §6).

If the split is not an exact parameter view, every distributed result is silently
invalid. This is written in Task 3 and must stay green forever.
"""

from __future__ import annotations

import torch

from src.models.chained import ChainedModel
from src.models.splittable import SplittableMLP


def test_chain_equals_unsplit_for_all_depths() -> None:
    torch.manual_seed(0)
    model = SplittableMLP(n_features=12, hidden_layers=4, hidden_width=32)
    model.eval()  # freeze dropout so the comparison is deterministic
    x = torch.randn(16, 12)
    reference = model(x)
    for d in range(1, 5):
        client, server = model.split(d)
        chained = ChainedModel(client, server)
        chained.eval()
        assert torch.allclose(chained(x), reference, atol=1e-6), f"mismatch at d={d}"


def test_split_shares_parameters_not_copies() -> None:
    model = SplittableMLP(n_features=8, hidden_layers=4, hidden_width=16)
    client, _server = model.split(2)
    # mutating a parent parameter must be visible through the client view
    with torch.no_grad():
        model.blocks[0][0].weight.add_(1.0)  # type: ignore[index]
    assert (
        client.blocks[0][0].weight.detach().sum()  # type: ignore[index]
        == model.blocks[0][0].weight.detach().sum()  # type: ignore[index]
    )


def test_invalid_depth_raises() -> None:
    model = SplittableMLP(n_features=8, hidden_layers=4, hidden_width=16)
    for bad in (0, 5):
        try:
            model.split(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
