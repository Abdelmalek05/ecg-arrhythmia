"""One dataclass holding every knob, so an ablation is `replace(cfg, axis=value)`.

Each field maps to one Phase 4 ablation axis. See PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

SEED = 42

# The three seeds every ablation is averaged over.
SEEDS = (1, 2, 3)


@dataclass
class Config:
    # --- data -------------------------------------------------------- axis 9
    features: str = "waveform+rr"       # "waveform" (250) | "waveform+rr" (254)
    classes: str = "all"                # "all" (N/S/V/F) | "NV" (binary, Phase 2)

    # --- architecture -------------------------------------------------------
    hidden_dims: tuple[int, ...] = (64, 32)
    hidden_activation: str = "relu"     # "relu" | "tanh"
    output_activation: str = "softmax"  # "softmax" | "sigmoid"

    # --- initialisation ---------------------------------------------- axis 1
    init: str = "he"                    # "zeros" | "random" | "xavier" | "he"

    # --- regularisation ------------------------------------------- axes 2,3,7
    l2: float = 0.0                     # lambda; 0 disables
    keep_prob: float = 1.0              # 1.0 disables dropout
    batch_norm: bool = False

    # --- optimisation --------------------------------------------- axes 4,5,6
    optimizer: str = "adam"             # "gd" | "momentum" | "rmsprop" | "adam"
    learning_rate: float = 0.01
    lr_decay: str = "none"              # "none" | "inverse" | "exponential"
    decay_rate: float = 0.01
    batch_size: int = 64                # 0 means full batch
    epochs: int = 30

    # --- class imbalance ---------------------------------------------- axis 8
    class_weights: str = "none"         # "none" | "balanced"

    # --- bookkeeping --------------------------------------------------------
    seed: int = SEED
    phase: str = ""
    note: str = ""

    def layer_dims(self, n_x: int, n_y: int) -> list[int]:
        """Full layer sizes, e.g. [254, 64, 32, 4]."""
        return [n_x, *self.hidden_dims, n_y]

    def as_row(self) -> dict:
        d = asdict(self)
        d["hidden_dims"] = "-".join(str(h) for h in self.hidden_dims)
        return d
