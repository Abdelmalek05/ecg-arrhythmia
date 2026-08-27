# this script reads the TEST set. it is meant to be run once.
#
# the rule of the project: every decision was made on dev. the three configs below
# were already chosen before this file was written, and nothing is tuned after it.
# we evaluate them once, write the numbers down, and stop.

import json
import numpy as np

from dataclasses import replace

from ecg.config import Config
from ecg.data import load_split, class_names, fit_standardizer, apply_standardizer
from ecg.init import initialize_parameters
from ecg.model import forward, backward, predict, new_bn_state
from ecg.losses import compute_loss
from ecg.data import one_hot, iterate_minibatches
from ecg.metrics import (accuracy, macro_f1, per_class, confusion_matrix,
                         majority_baseline, per_patient, report)
from ecg.optimizers import init_optimizer_state, update_parameters
from ecg.paths import RESULTS

SEEDS = [1, 2, 3]

# the winner on dev, plus the two comparisons, all decided before opening test
CONFIGS = {
    "timing only (4)": Config(classes="all", output_activation="softmax",
                              features="rr", standardize=True, hidden_dims=(16,),
                              hidden_activation="tanh", init="he", optimizer="gd",
                              learning_rate=0.1, batch_size=64, epochs=30),
    "shape + timing (254)": Config(classes="all", output_activation="softmax",
                                   features="waveform+rr", standardize=True,
                                   hidden_dims=(16,), hidden_activation="tanh",
                                   init="he", optimizer="gd", learning_rate=0.1,
                                   batch_size=64, epochs=30),
    "shape only (250)": Config(classes="all", output_activation="softmax",
                               features="waveform", standardize=True,
                               hidden_dims=(16,), hidden_activation="tanh",
                               init="he", optimizer="gd", learning_rate=0.1,
                               batch_size=64, epochs=30),
}


def train_once(config):
    """Train on TRAIN only. Returns the parameters and the standardiser."""
    Xtr, ytr, _ = load_split("train", config.features, config.classes, columns=True)
    mu, sd = fit_standardizer(Xtr, columns=True)
    if config.standardize:
        Xtr = apply_standardizer(Xtr, mu, sd)

    Ytr = one_hot(ytr.ravel(), 4).T.astype(np.float64)
    layer_dims = config.layer_dims(Xtr.shape[0], 4)
    parameters = initialize_parameters(layer_dims, config.init, config.seed)
    state = init_optimizer_state(parameters, config.optimizer)

    for epoch in range(config.epochs):
        for Xb, Yb in iterate_minibatches(Xtr, Ytr, config.batch_size,
                                          seed=config.seed + epoch, columns=True):
            AL, cache = forward(Xb, parameters, config.hidden_activation,
                                config.output_activation, training=True)
            grads = backward(AL, Yb, cache, parameters, config.hidden_activation,
                             config.output_activation,
                             loss_name="categorical_crossentropy")
            parameters, state = update_parameters(parameters, grads, state,
                                                  config.optimizer,
                                                  config.learning_rate)
    return parameters, (mu, sd)


def evaluate_on(split, config, parameters, scaler):
    X, y, rec = load_split(split, config.features, config.classes, columns=True)
    if config.standardize:
        X = apply_standardizer(X, scaler[0], scaler[1])
    pred = predict(X, parameters, config.hidden_activation, config.output_activation)
    return y.ravel(), pred, rec


def main():
    names = class_names("all")
    out = {}

    print("=" * 74)
    print("READING THE TEST SET. THIS HAPPENS ONCE.")
    print("=" * 74)
    _, yte, _ = load_split("test", classes="all", columns=True)
    print("test beats: %d   |   always-predict-N accuracy %.4f, macro-F1 %.4f"
          % (yte.size, majority_baseline(yte.ravel(), 4),
             macro_f1(yte.ravel(), np.zeros_like(yte).ravel(), 4)))
    print()

    for label, base in CONFIGS.items():
        dev_f1, test_f1, dev_acc, test_acc = [], [], [], []
        keep = None
        for seed in SEEDS:
            cfg = replace(base, seed=seed)
            params, scaler = train_once(cfg)
            yd, pd_, _ = evaluate_on("dev", cfg, params, scaler)
            yt, pt, rt = evaluate_on("test", cfg, params, scaler)
            dev_f1.append(macro_f1(yd, pd_, 4)); test_f1.append(macro_f1(yt, pt, 4))
            dev_acc.append(accuracy(yd, pd_)); test_acc.append(accuracy(yt, pt))
            if seed == SEEDS[0]:
                keep = (yt, pt, rt, params, cfg, scaler)
        print("%-22s dev mF1 %.4f +/- %.4f    TEST mF1 %.4f +/- %.4f    test acc %.4f"
              % (label, np.mean(dev_f1), np.std(dev_f1),
                 np.mean(test_f1), np.std(test_f1), np.mean(test_acc)))
        out[label] = {"dev_macro_f1": float(np.mean(dev_f1)),
                      "dev_std": float(np.std(dev_f1)),
                      "test_macro_f1": float(np.mean(test_f1)),
                      "test_std": float(np.std(test_f1)),
                      "test_acc": float(np.mean(test_acc)),
                      "drop": float(np.mean(dev_f1) - np.mean(test_f1))}
        if label == "timing only (4)":
            yt, pt, rt = keep[0], keep[1], keep[2]
            out["_best_true"] = yt.tolist()
            out["_best_pred"] = pt.tolist()
            out["_best_rec"] = rt.tolist()

    print("\n" + "=" * 74)
    print("THE CHOSEN MODEL ON TEST: timing only, 4 features")
    print("=" * 74)
    yt = np.array(out["_best_true"]); pt = np.array(out["_best_pred"])
    rt = np.array(out["_best_rec"])
    print(report(yt, pt, names))
    print("\nconfusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(yt, pt, 4)
    print("      " + "".join("%8s" % c for c in names))
    for i, c in enumerate(names):
        print("%6s" % c + "".join("%8d" % v for v in cm[i]))

    pp = per_patient(yt, pt, rt, 4)
    vals = np.array(list(pp.values()))
    print("\nper patient macro-F1 over the 22 test patients:")
    print("   min %.3f   median %.3f   max %.3f" % (vals.min(), np.median(vals), vals.max()))
    worst = sorted(pp.items(), key=lambda kv: kv[1])[:5]
    best = sorted(pp.items(), key=lambda kv: kv[1])[-3:]
    print("   worst: " + ", ".join("%s %.3f" % (k, v) for k, v in worst))
    print("   best : " + ", ".join("%s %.3f" % (k, v) for k, v in best))
    out["_per_patient"] = {k: float(v) for k, v in pp.items()}

    (RESULTS / "phase5_test_results.json").write_text(json.dumps(out, indent=1))
    print("\nsaved results/phase5_test_results.json")


if __name__ == "__main__":
    main()
