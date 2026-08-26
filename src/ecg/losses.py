"""Loss functions — L2 term (axis 2) and class weighting (axis 8) live here.

TO IMPLEMENT (Phase 2 for BCE, Phase 3 for cross-entropy).

Stability:
  - binary_crossentropy: clip AL away from 0 and 1 before log
  - categorical_crossentropy: use activations.log_softmax, never log(softmax(Z))
"""
from __future__ import annotations

import numpy as np

NAMES = ("binary_crossentropy", "categorical_crossentropy")


def compute_loss(AL: np.ndarray, Y: np.ndarray, name: str,
                 class_weights: np.ndarray | None = None,
                 parameters: dict | None = None, l2: float = 0.0) -> float:
    """Average loss over the batch, plus the L2 penalty if l2 > 0.

    AL            : (1, m) probabilities for binary, (n_y, m) for multi-class
    Y             : (1, m) 0/1 labels for binary, (n_y, m) one-hot for multi-class
    class_weights : (n_y,) per-class multipliers, or None. Phase 4 axis 8.
    parameters    : needed only when l2 > 0 (the penalty sums over every W, not b)
    l2            : lambda. The penalty is l2 / (2 * m) * sum(W**2).
    """
    if name not in NAMES:
        raise ValueError(f"unknown loss {name!r}, expected one of {NAMES}")
    raise NotImplementedError("Phase 2 / Phase 3")


def loss_backward(AL: np.ndarray, Y: np.ndarray, name: str,
                  class_weights: np.ndarray | None = None) -> np.ndarray:
    """Gradient of the loss w.r.t. the OUTPUT PRE-ACTIVATION, i.e. dZL, not dAL.

    For both sigmoid+BCE and softmax+cross-entropy this simplifies to (AL - Y)
    (scaled by 1/m and by class weights). Returning dZL directly avoids ever
    dividing by AL, which is where the numerical trouble lives.
    """
    raise NotImplementedError("Phase 2 / Phase 3")


def l2_penalty(parameters: dict, l2: float, m: int) -> float:
    """l2 / (2m) * sum over all W of sum(W**2). Biases are not regularised."""
    raise NotImplementedError("Phase 4")


def l2_gradient(W: np.ndarray, l2: float, m: int) -> np.ndarray:
    """The term added to dW when L2 is on: (l2 / m) * W."""
    raise NotImplementedError("Phase 4")
