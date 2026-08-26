"""Parameter initialisation — Phase 4 ablation axis 1.

TO IMPLEMENT (Phase 2). Scale factors follow the course:

    zeros    W = 0                       (included to demonstrate why it fails)
    random   W = randn * 0.01
    xavier   W = randn * sqrt(1 / n_prev)
    he       W = randn * sqrt(2 / n_prev)

Biases are always zeros — symmetry is broken by W alone.
"""
from __future__ import annotations

import numpy as np

METHODS = ("zeros", "random", "xavier", "he")


def initialize_parameters(layer_dims: list[int], method: str = "he",
                          seed: int | None = None) -> dict[str, np.ndarray]:
    """Build the parameter dict for an L-layer network.

    layer_dims : e.g. [254, 64, 32, 4] -> 3 weight matrices
    method     : one of METHODS
    returns    : {"W1": (n_1, n_0), "b1": (n_1, 1), "W2": ..., ...}

    Shapes follow C1W4: W_l is (layer_dims[l], layer_dims[l-1]), b_l is (layer_dims[l], 1).
    """
    if method not in METHODS:
        raise ValueError(f"unknown init {method!r}, expected one of {METHODS}")
    raise NotImplementedError("Phase 2")


def scale_for(method: str, n_prev: int, n_curr: int) -> float:
    """The multiplier applied to randn for one layer. Kept separate so it is testable."""
    raise NotImplementedError("Phase 2")
