"""Activation functions and their derivatives.

TO IMPLEMENT (Phase 2).

Numerical stability is not optional here:
  - sigmoid must not overflow for large |z|
  - softmax must subtract the row/column max before exponentiating
"""
from __future__ import annotations

import numpy as np

HIDDEN = ("relu", "tanh")
OUTPUT = ("sigmoid", "softmax")


def activation(Z: np.ndarray, name: str) -> np.ndarray:
    """Apply an activation elementwise (softmax: over the class axis).

    Z    : (n_l, m) pre-activations
    name : "relu" | "tanh" | "sigmoid" | "softmax"
    """
    raise NotImplementedError("Phase 2")


def activation_backward(dA: np.ndarray, Z: np.ndarray, name: str) -> np.ndarray:
    """dZ = dA * g'(Z) for the hidden activations.

    Note: for "softmax" paired with cross-entropy the combined gradient is computed
    in losses.loss_backward, not here — that cancellation is the whole point.
    """
    raise NotImplementedError("Phase 2")


def softmax(Z: np.ndarray, axis: int = 0) -> np.ndarray:
    """Stable softmax: subtract the max along `axis` before exponentiating."""
    raise NotImplementedError("Phase 2")


def log_softmax(Z: np.ndarray, axis: int = 0) -> np.ndarray:
    """log(softmax(Z)) via log-sum-exp. Use this inside the loss, never log(softmax(Z))."""
    raise NotImplementedError("Phase 2")
