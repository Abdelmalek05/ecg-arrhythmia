# this file loads the beats we built in phase 0
# it also picks the features, picks the classes, and cuts the data into small batches

from __future__ import annotations

import json

import numpy as np

from .paths import BUILD

CLASSES = ["N", "S", "V", "F"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def splits() -> dict:
    """Which patient records went into train / dev / test."""
    return json.loads((BUILD / "splits.json").read_text())


def load_split(name: str, features: str = "waveform+rr", classes: str = "all",
               columns: bool = False):
    """Return (X, y, rec) for one split.

    name      : "train" | "dev" | "test"
    features  : "waveform" -> 250 inputs; "waveform+rr" -> 254 inputs
    classes   : "all" -> labels 0..3 (N,S,V,F); "NV" -> binary, N=0 V=1
    columns   : if True return X as (n_x, m) and y as (1, m) / (n_y, m)

    `rec` is the patient id per beat. It is NEVER a model input — it exists so
    splits stay honest and so error analysis can group by patient.
    """
    if name not in ("train", "dev", "test"):
        raise ValueError(f"unknown split {name!r}")

    X = np.load(BUILD / f"{name}_X.npy")
    rr = np.load(BUILD / f"{name}_rr.npy")
    y = np.load(BUILD / f"{name}_y.npy")
    rec = np.load(BUILD / f"{name}_rec.npy")

    if features == "waveform":
        pass
    elif features == "waveform+rr":
        X = np.concatenate([X, rr], axis=1)
    elif features == "rr":
        # only the 4 timing numbers, no shape at all
        # this tells us if the timing alone carries any signal
        X = rr
    else:
        raise ValueError(f"unknown features {features!r}")

    if classes == "NV":
        m = np.isin(y, [CLASS_INDEX["N"], CLASS_INDEX["V"]])
        X, y, rec = X[m], y[m], rec[m]
        y = (y == CLASS_INDEX["V"]).astype(np.int64)
    elif classes != "all":
        raise ValueError(f"unknown classes {classes!r}")

    X = X.astype(np.float32, copy=False)
    if columns:
        X = X.T
        y = y.reshape(1, -1)
    return X, y, rec


def class_names(classes: str = "all") -> list[str]:
    return ["N", "V"] if classes == "NV" else list(CLASSES)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """(m,) integer labels -> (m, n_classes) one-hot."""
    out = np.zeros((y.size, n_classes), dtype=np.float32)
    out[np.arange(y.size), y.ravel()] = 1.0
    return out


def balanced_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Weights inversely proportional to class frequency, mean 1.

    Used by the `class_weights="balanced"` arm of Phase 4 axis 8.
    """
    counts = np.bincount(y.ravel(), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)


def iterate_minibatches(X, Y, batch_size: int, seed: int, columns: bool = False):
    """Yield shuffled mini-batches. batch_size=0 means one full-batch pass.

    X, Y are rows-first unless columns=True.
    """
    m = X.shape[1] if columns else X.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(m)
    step = m if batch_size in (0, None) else batch_size
    for start in range(0, m, step):
        idx = order[start:start + step]
        if columns:
            yield X[:, idx], Y[:, idx]
        else:
            yield X[idx], Y[idx]


def fit_standardizer(X, columns=True):
    """Learn the mean and the std of every feature, from the TRAIN split only.

    In Phase 3 we found that the 250 waveform numbers carry about 1000 times more
    variance than the 4 timing numbers, so the model almost cannot hear the timing.
    Making every feature the same size is one way to fix that.
    """
    axis = 1 if columns else 0
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True) + 1e-8
    return mu, sd


def apply_standardizer(X, mu, sd):
    # dev and test must use the numbers learned on train, never their own
    return (X - mu) / sd
