# this file runs one training from start to end, and writes the result in a csv
# every notebook calls train(config), so an experiment is just a different config

import csv
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from .config import Config
from .paths import RESULTS_CSV, ensure_dirs
from .data import (load_split, class_names, one_hot, balanced_class_weights,
                   iterate_minibatches, fit_standardizer, apply_standardizer)
from .init import initialize_parameters
from .model import forward, backward, predict, new_bn_state
from .init import number_of_layers
from .losses import compute_loss
from .metrics import accuracy, macro_f1, per_class, majority_baseline
from .optimizers import init_optimizer_state, update_parameters, learning_rate_at

RESULT_FIELDS = [
    "run_id", "timestamp", "phase", "note",
    "features", "classes", "standardize", "hidden_dims", "hidden_activation", "output_activation",
    "init", "l2", "keep_prob", "batch_norm",
    "optimizer", "learning_rate", "lr_decay", "decay_rate", "batch_size", "epochs",
    "class_weights", "seed",
    "train_loss", "train_acc", "train_macro_f1",
    "dev_loss", "dev_acc", "dev_macro_f1", "dev_f1_per_class",
    "baseline_acc", "wall_clock",
]


def prepare_labels(y, n_classes):
    # binary wants shape (1, m) with 0 and 1
    # many classes want one hot, shape (n_classes, m)
    if n_classes == 1:
        return y.reshape(1, -1).astype(np.float64)
    return one_hot(y.ravel(), n_classes).T.astype(np.float64)


def evaluate(X, y_true, parameters, config, n_classes, class_weights, bn_state=None):
    """Loss and metrics on one split."""
    if n_classes == 1:
        loss_name = "binary_crossentropy"
        n_report = 2
    else:
        loss_name = "categorical_crossentropy"
        n_report = n_classes

    AL, cache = forward(X, parameters, config.hidden_activation,
                        config.output_activation, keep_prob=1.0,
                        batch_norm=config.batch_norm, training=False,
                        bn_state=bn_state)
    Y = prepare_labels(y_true, n_classes)
    # categorical cross entropy wants the scores BEFORE softmax, they are in the cache
    ZL = cache["Z" + str(number_of_layers(parameters))]
    loss = compute_loss(AL, Y, loss_name, class_weights=class_weights,
                        parameters=parameters, l2=config.l2, ZL=ZL)

    y_pred = predict(X, parameters, config.hidden_activation, config.output_activation,
                     batch_norm=config.batch_norm, bn_state=bn_state)
    y_flat = y_true.ravel()
    return {
        "loss": loss,
        "acc": accuracy(y_flat, y_pred),
        "macro_f1": macro_f1(y_flat, y_pred, n_report),
        "f1_per_class": per_class(y_flat, y_pred, n_report)["f1"],
        "y_pred": y_pred,
    }


def train(config, verbose=True, log=True):
    """Train one model and give back the parameters, the history and the metrics.

    We never touch the test split here. Test is read once, in Phase 5.
    """
    start = time.time()
    np.random.seed(config.seed)

    # 1. data
    Xtr, ytr, _ = load_split("train", config.features, config.classes, columns=True)
    Xdv, ydv, _ = load_split("dev", config.features, config.classes, columns=True)

    if config.standardize:
        mu, sd = fit_standardizer(Xtr, columns=True)
        Xtr = apply_standardizer(Xtr, mu, sd)
        Xdv = apply_standardizer(Xdv, mu, sd)   # train statistics, not dev's own

    names = class_names(config.classes)
    if config.output_activation == "sigmoid":
        n_classes = 1          # one output unit is enough for two classes
        n_report = 2
    else:
        n_classes = len(names)
        n_report = n_classes

    Ytr = prepare_labels(ytr, n_classes)

    if config.class_weights == "balanced":
        weights = balanced_class_weights(ytr.ravel(), n_report)
    else:
        weights = None

    # 2. parameters
    n_x = Xtr.shape[0]
    layer_dims = config.layer_dims(n_x, n_classes)
    parameters = initialize_parameters(layer_dims, config.init, config.seed,
                                       batch_norm=config.batch_norm)
    bn_state = new_bn_state(parameters) if config.batch_norm else None

    if n_classes == 1:
        loss_name = "binary_crossentropy"
    else:
        loss_name = "categorical_crossentropy"

    opt_state = init_optimizer_state(parameters, config.optimizer)

    history = {"train_loss": [], "train_macro_f1": [], "dev_loss": [],
               "dev_macro_f1": [], "learning_rate": []}

    # 3. the training loop
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        lr_now = learning_rate_at(config.learning_rate, epoch,
                                  config.lr_decay, config.decay_rate)
        history["learning_rate"].append(lr_now)

        for Xb, Yb in iterate_minibatches(Xtr, Ytr, config.batch_size,
                                          seed=config.seed + epoch, columns=True):
            AL, cache = forward(Xb, parameters, config.hidden_activation,
                                config.output_activation, keep_prob=config.keep_prob,
                                batch_norm=config.batch_norm, training=True,
                                seed=config.seed + epoch + n_batches, bn_state=bn_state)
            ZLb = cache["Z" + str(len(layer_dims) - 1)]
            batch_loss = compute_loss(AL, Yb, loss_name, class_weights=weights,
                                      parameters=parameters, l2=config.l2, ZL=ZLb)
            grads = backward(AL, Yb, cache, parameters, config.hidden_activation,
                             config.output_activation, l2=config.l2,
                             keep_prob=config.keep_prob, batch_norm=config.batch_norm,
                             class_weights=weights, loss_name=loss_name)

            parameters, opt_state = update_parameters(
                parameters, grads, opt_state, config.optimizer, lr_now)

            epoch_loss = epoch_loss + batch_loss
            n_batches = n_batches + 1

        tr = evaluate(Xtr, ytr, parameters, config, n_classes, weights, bn_state)
        dv = evaluate(Xdv, ydv, parameters, config, n_classes, weights, bn_state)
        history["train_loss"].append(tr["loss"])
        history["train_macro_f1"].append(tr["macro_f1"])
        history["dev_loss"].append(dv["loss"])
        history["dev_macro_f1"].append(dv["macro_f1"])

        if verbose and (epoch % max(1, config.epochs // 10) == 0 or epoch == config.epochs - 1):
            print("epoch " + str(epoch + 1) + "/" + str(config.epochs)
                  + "  train loss " + str(round(tr["loss"], 5))
                  + "  train mF1 " + str(round(tr["macro_f1"], 4))
                  + "  dev mF1 " + str(round(dv["macro_f1"], 4)))

    # 4. final numbers
    wall_clock = time.time() - start
    metrics = {
        "train_loss": round(tr["loss"], 6),
        "train_acc": round(tr["acc"], 6),
        "train_macro_f1": round(tr["macro_f1"], 6),
        "dev_loss": round(dv["loss"], 6),
        "dev_acc": round(dv["acc"], 6),
        "dev_macro_f1": round(dv["macro_f1"], 6),
        "dev_f1_per_class": "|".join(str(round(v, 4)) for v in dv["f1_per_class"]),
        "baseline_acc": round(majority_baseline(ydv.ravel(), n_report), 6),
    }

    run_id = None
    if log:
        run_id = log_result(config, metrics, wall_clock)

    return {"run_id": run_id, "config": config, "parameters": parameters,
            "history": history, "metrics": metrics, "class_names": names,
            "dev_pred": dv["y_pred"], "wall_clock": wall_clock}


# ------------------------------------------------------------------ the results file
def log_result(config, metrics, wall_clock, run_id=None):
    # add one line to results/results.csv, make the file if it is not there
    ensure_dirs()
    row = {}
    for field in RESULT_FIELDS:
        row[field] = ""
    row.update(config.as_row())
    for key in metrics:
        if key in RESULT_FIELDS:
            row[key] = metrics[key]
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]
    row["run_id"] = run_id
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["wall_clock"] = round(wall_clock, 2)

    clean = {}
    for field in RESULT_FIELDS:
        clean[field] = row.get(field, "")

    _migrate_if_header_changed()

    write_header = not RESULTS_CSV.exists()
    f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
    if write_header:
        writer.writeheader()
    writer.writerow(clean)
    f.close()
    return run_id


def _migrate_if_header_changed():
    """If we added a column since the file was made, rewrite the old rows.

    Without this the file gets rows of two different widths and pandas cannot
    read it at all. That happened once already, in Phase 4a.
    """
    if not RESULTS_CSV.exists():
        return
    reader = csv.reader(open(RESULTS_CSV, newline="", encoding="utf-8"))
    rows = list(reader)
    if not rows:
        return
    header = rows[0]
    if header == RESULT_FIELDS:
        return

    fixed = []
    for r in rows[1:]:
        if len(r) == len(header):
            d = dict(zip(header, r))
        elif len(r) == len(RESULT_FIELDS):
            d = dict(zip(RESULT_FIELDS, r))
        else:
            continue
        fixed.append({field: d.get(field, "") for field in RESULT_FIELDS})

    f = open(RESULTS_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
    writer.writeheader()
    writer.writerows(fixed)
    f.close()
    print("results.csv: header changed, rewrote " + str(len(fixed)) + " old rows")


def load_results():
    # every run we ever did, the Phase 4 plots are built from this file
    if not RESULTS_CSV.exists():
        return pd.DataFrame(columns=RESULT_FIELDS)
    return pd.read_csv(RESULTS_CSV)


def summarise(axis, metric="dev_macro_f1"):
    """Mean and std of one metric, grouped by one setting, over the seeds.

    A gap of 0.003 on a single seed is noise. This is what shows it.
    """
    df = load_results()
    if df.empty or axis not in df.columns:
        return pd.DataFrame()
    return df.groupby(axis)[metric].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
