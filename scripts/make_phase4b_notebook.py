# this script writes the phase 4b notebook
# we keep it so the notebook can be made again if it gets broken

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"

HEADER = """%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ecg.init import initialize_parameters, parameter_keys
from ecg.model import forward, backward, new_bn_state
from ecg.gradcheck import gradient_check
from ecg.train import load_results

df = load_results().query("phase == '4b'").copy()
f1 = df["dev_f1_per_class"].str.split("|", expand=True).astype(float)
f1.columns = ["f1_N", "f1_S", "f1_V", "f1_F"]
df = pd.concat([df, f1], axis=1)
print("phase 4b runs:", len(df))
"""


def md(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


cells = [
    md("""
# 07 - Batch normalisation (Phase 4b)

Batch norm takes the numbers inside the network and makes them mean 0 and std 1,
for every hidden unit, over the batch. Then it lets the network scale and shift them
back with two learned numbers, **gamma** and **beta**.

In Phase 4a we found something important: the model could not use the 4 timing
features, because the 250 shape features were about 1000 times louder. Scaling the
**input** fixed part of that.

Batch norm scales **inside** the network. So this notebook asks one question:

> **Can batch norm do the same job as scaling the input?**
"""),
    code(HEADER),
    md("""
## 1. The new parameters

Batch norm adds gamma and beta to every hidden layer. Not to the output layer.

gamma starts at 1 and beta starts at 0, so at the beginning batch norm only
normalises and does not change anything else.
"""),
    code("""
p_plain = initialize_parameters([8, 6, 5, 3], "he", seed=1)
p_bn = initialize_parameters([8, 6, 5, 3], "he", seed=1, batch_norm=True)

print("without batch norm:", parameter_keys(p_plain))
print("with batch norm   :", parameter_keys(p_bn))
print()
print("gamma1 is all ones :", np.all(p_bn["gamma1"] == 1))
print("beta1 is all zeros :", np.all(p_bn["beta1"] == 0))
print("no gamma on the last layer:", "gamma3" not in p_bn)
"""),
    md("""
## 2. Does it really normalise?
"""),
    code("""
rng = np.random.default_rng(0)
X = rng.standard_normal((8, 200)) * 5 + 3        # on purpose: not mean 0, not std 1

state = new_bn_state(p_bn)
AL, cache = forward(X, p_bn, "tanh", "softmax", batch_norm=True,
                    training=True, bn_state=state)

for l in [1, 2]:
    z = cache["bn"][l]["Z_hat"]
    print("layer %d after batch norm: mean %+.2e   std %.6f" % (l, z.mean(), z.std()))
"""),
    md("""
## 3. Checking the backward pass

This is the hardest derivative in the whole project.

The reason is that batch norm uses the **mean and the variance of the batch**. So if
you change one number, the mean changes, and that changes the normalised value of
**every other number in the batch** as well. The gradient therefore has three parts,
not one.

We must check it before we trust anything.

One important detail: with batch norm we check in **training mode**. In prediction
mode batch norm uses a saved running average instead of the batch, so it is a
different function and the check would not mean anything.
"""),
    code("""
X_small = rng.standard_normal((8, 40))
Y_small = np.zeros((3, 40))
Y_small[rng.integers(0, 3, 40), np.arange(40)] = 1.0

for dims in [[8, 6, 3], [8, 6, 5, 3]]:
    p = initialize_parameters(dims, "he", seed=2, batch_norm=True)
    AL, cache = forward(X_small, p, "tanh", "softmax", batch_norm=True, training=True)
    grads = backward(AL, Y_small, cache, p, "tanh", "softmax", batch_norm=True)
    print(str(dims), "->", end=" ")
    gradient_check(p, grads, X_small, Y_small, "tanh", "softmax", batch_norm=True)
"""),
    code("""
# and it still catches a real mistake
p = initialize_parameters([8, 6, 5, 3], "he", seed=3, batch_norm=True)
AL, cache = forward(X_small, p, "tanh", "softmax", batch_norm=True, training=True)
grads = backward(AL, Y_small, cache, p, "tanh", "softmax", batch_norm=True)
grads["dgamma1"] = grads["dgamma1"] * 1.01

err = gradient_check(p, grads, X_small, Y_small, "tanh", "softmax",
                     batch_norm=True, verbose=False)
print("with a 1 percent mistake in dgamma1: %.3e   caught: %s" % (err, err > 1e-7))
"""),
    md("""
### A small thing that proves the wiring is right
"""),
    code("""
print("max |db1| = %.2e" % abs(grads["db1"]).max())
"""),
    md("""
The gradient of `b` is zero.

That is correct, not a bug. Batch norm takes the mean away, so adding `b` to `Z`
changes nothing at all. The bias has no effect and so it has no gradient. Real
libraries turn the bias off when batch norm is on, for exactly this reason.

## 4. The experiment

36 runs: batch norm on/off, input scaling on/off, three feature sets, 3 seeds.
"""),
    code("""
table = (df.groupby(["features", "batch_norm", "standardize"])[["dev_macro_f1", "f1_S"]]
           .agg(["mean", "std"]).round(4))
print(table.to_string())
"""),
    code("""
fig = plt.imread("../results/figures/phase4b_batchnorm.png")
plt.figure(figsize=(14, 5)); plt.imshow(fig); plt.axis("off"); plt.show()
"""),
    md("""
## 5. The answer is no

Look at the 254 feature case, class S, which is where the problem was:
"""),
    code("""
d = df[df.features == "waveform+rr"]
for bn, std, label in [(False, False, "neither"),
                       (True, False, "batch norm only (inside)"),
                       (False, True, "input scaling only (before)"),
                       (True, True, "both")]:
    s = d[(d.batch_norm == bn) & (d.standardize == std)].sort_values("seed")
    print("%-28s S-F1 %.4f   seeds %s"
          % (label, s["f1_S"].mean(), np.round(np.sort(s["f1_S"].values), 4)))
"""),
    md("""
**Batch norm alone gives 0.023. Scaling the input gives 0.133.**

And it is not a lucky seed: the **worst** input scaling run (0.043) is still better
than the **best** batch norm run (0.035).

### Why not? It is about *when*, not *how much*

Batch norm works on `Z = W * A + b`. But `Z` is already a **mix** of all 254 inputs.
By the time the numbers reach `Z`, the 4 quiet features have already been added
together with the 250 loud ones, and their part of the mix is tiny.

Normalising `Z` then makes the *mixture* mean 0 and std 1. It cannot go back and give
the quiet features a bigger share, because that information is already gone.

Scaling the input happens **before** the mixing, so every feature enters the sum with
the same size.

> **Normalising has to happen before the mixing, not after.**

This is a nice thing to have measured rather than read.

## 6. What we learned

1. **Batch norm is not a replacement for scaling your input.** They fix different
   problems, in different places.
2. **Overall it changes almost nothing here** (0.4014 against 0.4016 on 254
   features). It is not harmful, it is just not the tool for this problem.
3. **The bias becomes useless** when batch norm is on. We saw its gradient go to
   1e-17 with our own eyes.
4. **The gradient check earned its keep again.** Three terms, easy to get wrong,
   and we would not have known.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
path = OUT / "07_batchnorm.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote " + str(path))
