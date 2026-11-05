"""Client-side cost, computed analytically from layer shapes (01_RESEARCH_DESIGN §4.6).

DV3 for H1: as the cut depth ``d`` grows, more of the network sits on the client,
so client forward FLOPs and parameter count rise. No wall-clock timing — it is
machine-dependent and not comparable across the two machines this is built and
trained on (07_COMPUTE §4.6). The cut-layer activation size is reported too; in a
uniform-width MLP it is ``hidden_width`` regardless of ``d`` (stated plainly, not
hidden), so only FLOPs and params carry the depth trend.
"""

from __future__ import annotations

from typing import Any


def client_cost(
    n_features: int,
    hidden_width: int,
    cut_depth: int,
    bytes_per_element: int = 4,
) -> dict[str, Any]:
    """Analytic client cost for a cut after ``cut_depth`` hidden blocks.

    Client = hidden blocks ``1..d`` (each ``Linear`` + ReLU + Dropout); the head
    stays server-side. FLOPs count multiply-adds as 2 per MAC; ReLU/Dropout are
    negligible and omitted.

    Args:
        n_features: Input dimension (first block's in-features).
        hidden_width: Hidden layer width (constant across the MLP).
        cut_depth: Number of hidden blocks on the client (``d``).
        bytes_per_element: 4 for float32 activations.

    Returns:
        ``client_forward_flops``, ``client_params``, and
        ``cut_activation_bytes_per_sample``.
    """
    if cut_depth < 1:
        raise ValueError(f"cut_depth must be >= 1, got {cut_depth}")
    flops = 0
    params = 0
    in_dim = n_features
    for _ in range(cut_depth):
        out_dim = hidden_width
        flops += 2 * in_dim * out_dim  # Linear MACs x2
        params += in_dim * out_dim + out_dim  # weights + bias
        in_dim = out_dim
    return {
        "client_forward_flops": int(flops),
        "client_params": int(params),
        "cut_activation_bytes_per_sample": int(hidden_width * bytes_per_element),
    }
