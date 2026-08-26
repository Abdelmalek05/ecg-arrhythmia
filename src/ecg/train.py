"""The training loop and the results log.

`train(config)` is the single entry point every notebook calls, so an ablation is:

    from dataclasses import replace
    for opt in ["gd", "momentum", "rmsprop", "adam"]:
        for seed in [1, 2, 3]:
            train(replace(cfg, optimizer=opt, seed=seed))

The loop itself is TO IMPLEMENT (Phase 2, extended through Phase 4).
`log_result` and `load_results` below are plumbing and already work.
"""
from __future__ import annotations

import csv
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from .config import Config
from .paths import RESULTS_CSV, ensure_dirs

RESULT_FIELDS = [
    "run_id", "timestamp", "phase", "note",
    "features", "classes", "hidden_dims", "hidden_activation", "output_activation",
    "init", "l2", "keep_prob", "batch_norm",
    "optimizer", "learning_rate", "lr_decay", "decay_rate", "batch_size", "epochs",
    "class_weights", "seed",
    "train_loss", "train_acc", "train_macro_f1",
    "dev_loss", "dev_acc", "dev_macro_f1", "dev_f1_per_class",
    "baseline_acc", "wall_clock",
]


def train(config: Config, verbose: bool = True, log: bool = True) -> dict:
    """Train one model end to end and return a results dict.

    Steps:
      1. data.load_split for train and dev, honouring config.features / config.classes
      2. init.initialize_parameters(config.layer_dims(n_x, n_y), config.init, config.seed)
      3. optimizers.init_optimizer_state
      4. for each epoch: optimizers.learning_rate_at, then iterate_minibatches ->
         model.forward -> losses.compute_loss -> model.backward ->
         optimizers.update_parameters
      5. evaluate on train and dev with metrics.report
      6. log_result(...)

    Returns {"config":..., "parameters":..., "history":..., "metrics":...}.
    Never touches the test split — that is read once, in Phase 5.
    """
    raise NotImplementedError("Phase 2")


# --------------------------------------------------------------------- plumbing
def log_result(config: Config, metrics: dict, wall_clock: float,
               run_id: str | None = None) -> str:
    """Append one row to results/results.csv. Creates the file and header if needed."""
    ensure_dirs()
    row = {k: "" for k in RESULT_FIELDS}
    row.update(config.as_row())
    row.update({k: v for k, v in metrics.items() if k in RESULT_FIELDS})
    row["run_id"] = run_id or uuid.uuid4().hex[:8]
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["wall_clock"] = round(wall_clock, 2)
    row = {k: row.get(k, "") for k in RESULT_FIELDS}

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return row["run_id"]


def load_results() -> pd.DataFrame:
    """Every run ever logged. Phase 4's plots are built from this, not from memory."""
    if not RESULTS_CSV.exists():
        return pd.DataFrame(columns=RESULT_FIELDS)
    return pd.read_csv(RESULTS_CSV)


def summarise(axis: str, metric: str = "dev_macro_f1") -> pd.DataFrame:
    """Mean and std of `metric` grouped by one ablation axis, across seeds.

    A 0.3% gap on a single seed is noise; this is what makes that visible.
    """
    df = load_results()
    if df.empty or axis not in df.columns:
        return pd.DataFrame()
    g = df.groupby(axis)[metric].agg(["mean", "std", "count"])
    return g.sort_values("mean", ascending=False)
