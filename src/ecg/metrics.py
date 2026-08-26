# this file measures how good a model is
# accuracy lies here because most beats are normal, so we use macro f1

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """rows = true class, cols = predicted class."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class(y_true, y_pred, n_classes: int) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support for every class."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "support": actual.astype(np.int64)}


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """Unweighted mean F1 over classes — the project's single number.

    Classes with zero support are excluded so an absent class cannot drag the
    score to zero (relevant for F, which is thin in dev).
    """
    p = per_class(y_true, y_pred, n_classes)
    present = p["support"] > 0
    return float(p["f1"][present].mean()) if present.any() else 0.0


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()).mean())


def majority_baseline(y_true, n_classes: int) -> float:
    """Accuracy of always predicting the most common class. The floor to beat."""
    counts = np.bincount(np.asarray(y_true).ravel(), minlength=n_classes)
    return float(counts.max() / counts.sum())


def report(y_true, y_pred, class_names: list[str]) -> str:
    """Printable per-class table plus the headline numbers."""
    n = len(class_names)
    p = per_class(y_true, y_pred, n)
    lines = [f"{'class':>6}{'prec':>9}{'recall':>9}{'f1':>9}{'support':>9}"]
    for i, name in enumerate(class_names):
        lines.append(f"{name:>6}{p['precision'][i]:>9.3f}{p['recall'][i]:>9.3f}"
                     f"{p['f1'][i]:>9.3f}{p['support'][i]:>9d}")
    lines.append("")
    lines.append(f"  accuracy  {accuracy(y_true, y_pred):.4f}"
                 f"   (baseline {majority_baseline(y_true, n):.4f})")
    lines.append(f"  macro-F1  {macro_f1(y_true, y_pred, n):.4f}")
    return "\n".join(lines)


def per_patient(y_true, y_pred, rec: np.ndarray, n_classes: int) -> dict[str, float]:
    """Macro-F1 per patient.

    Patients differ enormously here (0% to 78% abnormal beats), so a single pooled
    number hides which hearts the model fails on. Used in Phase 5 error analysis.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {r: macro_f1(y_true[rec == r], y_pred[rec == r], n_classes)
            for r in sorted(set(np.asarray(rec).ravel().tolist()))}
