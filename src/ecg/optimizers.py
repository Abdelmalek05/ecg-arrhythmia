"""Parameter update rules — Phase 4 ablation axis 5.

TO IMPLEMENT (Phase 4).

    gd        W -= lr * dW
    momentum  v = beta1*v + (1-beta1)*dW           ; W -= lr * v
    rmsprop   s = beta2*s + (1-beta2)*dW**2        ; W -= lr * dW / (sqrt(s)+eps)
    adam      both, with bias correction using the step counter t

State lives in a dict so the same `update_parameters` call works for all four.
"""
from __future__ import annotations

import numpy as np

NAMES = ("gd", "momentum", "rmsprop", "adam")

DEFAULTS = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}


def init_optimizer_state(parameters: dict, name: str) -> dict:
    """Zero-initialised moment buffers matching `parameters`, plus a step counter.

    Returns {} for "gd"; {"v": {...}, "t": 0} for momentum; {"s": {...}} for rmsprop;
    {"v": {...}, "s": {...}, "t": 0} for adam.
    """
    if name not in NAMES:
        raise ValueError(f"unknown optimizer {name!r}, expected one of {NAMES}")
    raise NotImplementedError("Phase 4")


def update_parameters(parameters: dict, grads: dict, state: dict, name: str,
                      learning_rate: float, **hp) -> tuple[dict, dict]:
    """Apply one update. Returns (parameters, state), both updated in place.

    hp overrides DEFAULTS (beta1, beta2, epsilon).
    """
    raise NotImplementedError("Phase 4")


def learning_rate_at(initial_lr: float, epoch: int, schedule: str = "none",
                     decay_rate: float = 0.01) -> float:
    """Phase 4 axis 6.

        none         initial_lr
        inverse      initial_lr / (1 + decay_rate * epoch)
        exponential  initial_lr * (1 - decay_rate) ** epoch
    """
    raise NotImplementedError("Phase 4")
