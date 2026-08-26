"""Forward and backward propagation for an L-layer network.

TO IMPLEMENT (Phase 2, extended in Phases 3 and 4).

Carries Phase 4 axes 2 (l2), 3 (keep_prob) and 7 (batch_norm). Build it in stages:
  Phase 2  forward/backward with relu/tanh hidden + sigmoid output
  Phase 3  softmax output
  Phase 4  dropout, L2, batch norm

Convention: columns-first, X is (n_x, m), as in C1W4.
"""
from __future__ import annotations

import numpy as np


def forward(X: np.ndarray, parameters: dict, hidden_activation: str = "relu",
            output_activation: str = "softmax", keep_prob: float = 1.0,
            batch_norm: bool = False, training: bool = True, seed: int | None = None):
    """Run the network forward.

    X       : (n_x, m)
    returns : (AL, cache) where AL is (n_y, m) and cache holds everything backward needs

    keep_prob < 1 applies inverted dropout to HIDDEN layers only, and only when
    training=True. The dropout masks must be stored in the cache — backward has to
    reuse the exact same ones.
    """
    raise NotImplementedError("Phase 2")


def backward(AL: np.ndarray, Y: np.ndarray, cache: dict, parameters: dict,
             hidden_activation: str = "relu", output_activation: str = "softmax",
             l2: float = 0.0, keep_prob: float = 1.0, batch_norm: bool = False,
             class_weights: np.ndarray | None = None) -> dict:
    """Backpropagate.

    returns : {"dW1": ..., "db1": ..., "dW2": ..., ...} matching `parameters`

    Start from losses.loss_backward, which hands you dZL directly.
    """
    raise NotImplementedError("Phase 2")


def predict_proba(X: np.ndarray, parameters: dict, **kwargs) -> np.ndarray:
    """Forward pass with training=False (dropout off, batch-norm running stats)."""
    raise NotImplementedError("Phase 2")


def predict(X: np.ndarray, parameters: dict, threshold: float = 0.5, **kwargs) -> np.ndarray:
    """Hard class predictions, shape (m,).

    Binary: AL > threshold. Multi-class: argmax over the class axis.
    """
    raise NotImplementedError("Phase 2")
