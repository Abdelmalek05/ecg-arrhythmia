# this script runs every phase 4a experiment and writes them all to results/results.csv
# it takes about an hour, so we run it in the background
#
# the base config is exactly the phase 3 config, so the numbers can be compared

import time
from dataclasses import replace

from ecg.config import Config
from ecg.train import train

SEEDS = [1, 2, 3]

BASE = Config(classes="all", output_activation="softmax",
              features="waveform+rr", standardize=False,
              hidden_dims=(16,), hidden_activation="tanh", init="he",
              l2=0.0, keep_prob=1.0,
              optimizer="gd", learning_rate=0.1, lr_decay="none",
              batch_size=64, epochs=30, class_weights="none",
              phase="4a")

done = 0
start = time.time()


def run(cfg, label):
    global done
    out = train(cfg, verbose=False)
    done = done + 1
    m = out["metrics"]
    print("[%3d] %-44s dev mF1 %.4f  | %s  (%.0fs)"
          % (done, label, m["dev_macro_f1"], m["dev_f1_per_class"], out["wall_clock"]),
          flush=True)


# ---------------------------------------------------------------- one at a time
print("=== axis 1: initialisation ===", flush=True)
for value in ["zeros", "random", "xavier", "he"]:
    for seed in SEEDS:
        run(replace(BASE, init=value, seed=seed, note="axis1 init"),
            "init=" + value + " seed=" + str(seed))

print("\n=== axis 2: l2 ===", flush=True)
for value in [0.0, 0.01, 0.1, 1.0]:
    for seed in SEEDS:
        run(replace(BASE, l2=value, seed=seed, note="axis2 l2"),
            "l2=" + str(value) + " seed=" + str(seed))

print("\n=== axis 3: dropout ===", flush=True)
for value in [1.0, 0.8, 0.5]:
    for seed in SEEDS:
        run(replace(BASE, keep_prob=value, seed=seed, note="axis3 dropout"),
            "keep_prob=" + str(value) + " seed=" + str(seed))

print("\n=== axis 6: learning rate decay ===", flush=True)
for value in ["none", "inverse", "exponential"]:
    for seed in SEEDS:
        run(replace(BASE, lr_decay=value, decay_rate=0.05, seed=seed, note="axis6 lrdecay"),
            "lr_decay=" + value + " seed=" + str(seed))

print("\n=== axis 10: standardisation ===", flush=True)
for value in [False, True]:
    for seed in SEEDS:
        run(replace(BASE, standardize=value, seed=seed, note="axis10 standardize"),
            "standardize=" + str(value) + " seed=" + str(seed))

print("\n=== axis 4: batch size (batch_size=8 is slow) ===", flush=True)
for value in [0, 512, 64, 8]:
    for seed in SEEDS:
        run(replace(BASE, batch_size=value, seed=seed, note="axis4 batchsize"),
            "batch_size=" + str(value) + " seed=" + str(seed))

# ---------------------------------------------------------------- optimizer x lr
# comparing optimizers at one learning rate is not fair, they do not want the same
# one. so we give each optimizer two sensible values and keep the better.
print("\n=== axis 5: optimizer x learning rate ===", flush=True)
LR_FOR = {"gd": [0.03, 0.1], "momentum": [0.03, 0.1],
          "rmsprop": [0.0003, 0.001], "adam": [0.0003, 0.001]}
for opt in ["gd", "momentum", "rmsprop", "adam"]:
    for lr in LR_FOR[opt]:
        for seed in SEEDS:
            run(replace(BASE, optimizer=opt, learning_rate=lr, seed=seed,
                        note="axis5 optimizer"),
                "opt=" + opt + " lr=" + str(lr) + " seed=" + str(seed))

# ---------------------------------------------------------------- the grids
# phase 3 showed axes are not independent, so these are grids, not sweeps.
print("\n=== grid A: features x optimizer x standardise ===", flush=True)
print("    (this is the phase 3 hypothesis: can adam hear the quiet features?)", flush=True)
for feats in ["waveform", "waveform+rr", "rr"]:
    for opt, lr in [("gd", 0.1), ("adam", 0.001)]:
        for std in [False, True]:
            for seed in SEEDS:
                run(replace(BASE, features=feats, optimizer=opt, learning_rate=lr,
                            standardize=std, seed=seed, note="gridA feat-opt-std"),
                    "feat=" + feats + " opt=" + opt + " std=" + str(std) + " seed=" + str(seed))

print("\n=== grid B: class weights x features (with adam + standardise) ===", flush=True)
for weights in ["none", "balanced"]:
    for feats in ["waveform", "waveform+rr", "rr"]:
        for seed in SEEDS:
            run(replace(BASE, class_weights=weights, features=feats,
                        optimizer="adam", learning_rate=0.001, standardize=True,
                        seed=seed, note="gridB weights-features"),
                "weights=" + weights + " feat=" + feats + " seed=" + str(seed))

print("\n\nDONE: %d runs in %.0f minutes" % (done, (time.time() - start) / 60.0), flush=True)
