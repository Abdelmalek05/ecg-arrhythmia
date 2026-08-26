"""Numerical gradient checking.

TO IMPLEMENT (Phase 2) — and run it the MOMENT the L-layer backward pass exists,
before training anything or reading any curve. A training curve produced by wrong
gradients looks perfectly reasonable.

Method: flatten every parameter into one vector, perturb each entry by +/- epsilon,
recompute the loss, and compare the finite difference to the analytic gradient:

    relative_error = ||grad_approx - grad|| / (||grad_approx|| + ||grad||)

Run it on a TINY network (a few units, a handful of examples) or it will take
forever — the cost is two forward passes per parameter.

Caveats that cause false failures:
  - turn dropout OFF (keep_prob=1.0); it makes the loss stochastic
  - relu is not differentiable at 0; use tanh, or expect the odd outlier
"""
from __future__ import annotations

import numpy as np

PASS_THRESHOLD = 1e-7


def dictionary_to_vector(parameters: dict) -> tuple[np.ndarray, list]:
    """Flatten {"W1":..., "b1":...} into one column vector plus the shape info."""
    raise NotImplementedError("Phase 2")


def vector_to_dictionary(theta: np.ndarray, shapes: list) -> dict:
    """Inverse of dictionary_to_vector."""
    raise NotImplementedError("Phase 2")


def gradients_to_vector(grads: dict, shapes: list) -> np.ndarray:
    """Flatten the gradient dict in the SAME order as dictionary_to_vector."""
    raise NotImplementedError("Phase 2")


def gradient_check(parameters: dict, grads: dict, X: np.ndarray, Y: np.ndarray,
                   loss_fn, epsilon: float = 1e-7, verbose: bool = True) -> float:
    """Return the relative error. Below PASS_THRESHOLD means backward is correct.

    loss_fn : callable(parameters) -> float, closing over everything else
    """
    raise NotImplementedError("Phase 2")
