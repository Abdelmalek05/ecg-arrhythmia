# this file draws the figures we use in the reports
# every figure is saved inside results/figures

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .paths import FIGURES, ensure_dirs

CLASS_COLOURS = {"N": "#2c6fbb", "S": "#e07b39", "V": "#c0392b", "F": "#7d3c98"}
FS = 360
PRE = 90


def beat_time_axis(n: int = 250) -> np.ndarray:
    """Milliseconds relative to the R-peak, for plotting a beat window."""
    return (np.arange(n) - PRE) / FS * 1000.0


def save(fig, name: str, dpi: int = 130):
    ensure_dirs()
    path = FIGURES / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    return path


def learning_curves(history: dict, title: str = "", name: str | None = None):
    """Train/dev loss and metric against epoch, from a train() history dict."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax, label in (("loss", axes[0], "loss"), ("macro_f1", axes[1], "macro-F1")):
        for split, style in (("train", "-"), ("dev", "--")):
            k = f"{split}_{key}"
            if k in history:
                ax.plot(history[k], style, label=split)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if name:
        save(fig, name)
    return fig


def confusion(cm: np.ndarray, class_names: list[str], title: str = "",
              normalise: bool = True, name: str | None = None):
    """Confusion matrix heatmap. Row-normalised by default — with an 89% majority
    class, raw counts show one bright cell and nothing else."""
    m = cm.astype(float)
    if normalise:
        rows = m.sum(axis=1, keepdims=True)
        m = np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{m[i, j]:.2f}" if normalise else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if m[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title(title or "Confusion matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if name:
        save(fig, name)
    return fig


def ablation(df, axis: str, metric: str = "dev_macro_f1", title: str = "",
             name: str | None = None):
    """Bar chart with error bars over seeds, from train.summarise output."""
    g = df.groupby(axis)[metric].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(g.index.astype(str), g["mean"], xerr=g["std"].fillna(0),
            color="#2c6fbb", alpha=0.85, capsize=4)
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} by {axis}")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    if name:
        save(fig, name)
    return fig


def beats(X: np.ndarray, labels: list[str], title: str = "", name: str | None = None,
          ncols: int = 5):
    """Grid of individual beat waveforms — the workhorse of Phase 5 error analysis."""
    n = len(X)
    nrows = int(np.ceil(n / ncols))
    t = beat_time_axis(X.shape[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.1 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k < n:
            lab = labels[k] if k < len(labels) else ""
            colour = CLASS_COLOURS.get(str(lab)[0], "#333333")
            ax.plot(t, X[k], color=colour, lw=1.1)
            ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_title(str(lab), fontsize=8, color=colour)
            ax.grid(alpha=0.15)
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if name:
        save(fig, name)
    return fig
