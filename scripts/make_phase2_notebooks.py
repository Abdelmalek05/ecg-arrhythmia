# this script writes the three phase 2 notebooks
# we keep it so the notebooks can be made again if they get broken

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"

HEADER = """%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt

from ecg.config import Config
from ecg.data import load_split, class_names
from ecg.metrics import report, macro_f1, accuracy, majority_baseline, confusion_matrix
from ecg import plots
"""


def md(text):
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(text):
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def write(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUT / (name + ".ipynb")
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote " + str(path))


# ---------------------------------------------------------------- notebook 02
nb02 = [
    md("""
# 02 - Logistic regression from zero

Phase 2, part 1. Here we build the simplest model: **one neuron**.

The task is binary. We only keep two classes:

- **N** = normal beat
- **V** = ventricular beat, it starts in the bottom of the heart and has a wide shape

The goal of this notebook is not a high score. The goal is to make the maths work
and to check it against a library that we trust.
"""),
    code(HEADER),
    md("""
## 1. Load the data and find the floor

Before any model, we need to know what "doing nothing" gives us.
Most beats are normal, so a lazy model can just say "normal" every time and still
look good. That number is our floor.
"""),
    code("""
Xtr, ytr, rec_tr = load_split("train", classes="NV", columns=True)
Xdv, ydv, rec_dv = load_split("dev", classes="NV", columns=True)

print("train:", Xtr.shape, " dev:", Xdv.shape)
print("train N =", int((ytr == 0).sum()), " V =", int((ytr == 1).sum()))
print("dev   N =", int((ydv == 0).sum()), " V =", int((ydv == 1).sum()))
"""),
    code("""
always_normal = np.zeros_like(ytr)
print("IF WE ALWAYS SAY NORMAL:")
print(report(ytr, always_normal, class_names("NV")))
"""),
    md("""
Look at the two numbers at the bottom.

- **Accuracy is 0.92**, that looks very good, but the model learned nothing.
- **Macro-F1 is 0.48**, and this one tells the truth.

Macro-F1 takes the F1 of each class and makes a simple average. Class V gets an F1
of 0, so the average drops. This is why we use macro-F1 in this project and not
accuracy.

**We must beat 0.48.**
"""),
    md("""
## 2. Train the neuron

One neuron does three things:

1. multiply the input by weights `w` and add a bias `b`
2. push the answer through a sigmoid, so it becomes a probability between 0 and 1
3. compare with the true label and move `w` and `b` a little in the better direction

We repeat step 3 many times. This is gradient descent.

The code is in `src/ecg/logistic.py`.
"""),
    code("""
from ecg.logistic import fit, predict

result = fit(Xtr, ytr, num_iterations=2000, learning_rate=0.5, print_every=400)

pred_tr = predict(result["w"], result["b"], Xtr)
pred_dv = predict(result["w"], result["b"], Xdv)
"""),
    code("""
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(result["costs"])
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
ax.set_title("The loss goes down, so the model is learning")
ax.grid(alpha=0.3)
plots.save(fig, "phase2_logistic_loss")
plt.show()
"""),
    md("""
## 3. Check against scikit-learn

This is the important part of the notebook.

We wrote the maths ourselves, so we can easily make a mistake. scikit-learn solves
the same problem with a better method. If our answer is close to theirs, our maths
is probably right.

We ask scikit-learn for `penalty=None`. By default it adds regularisation, and then
it would be solving a different problem, so the comparison would not be fair.
"""),
    code("""
from sklearn.linear_model import LogisticRegression

sk = LogisticRegression(penalty=None, max_iter=2000)
sk.fit(Xtr.T, ytr.ravel())
sk_tr = sk.predict(Xtr.T)
sk_dv = sk.predict(Xdv.T)

gap = abs(accuracy(ytr, pred_tr) - accuracy(ytr, sk_tr))

print("               ours    sklearn     gap")
print("train acc   %.4f     %.4f   %.4f" % (accuracy(ytr, pred_tr), accuracy(ytr, sk_tr), gap))
print("dev acc     %.4f     %.4f" % (accuracy(ydv, pred_dv), accuracy(ydv, sk_dv)))
print("dev mF1     %.4f     %.4f" % (macro_f1(ydv, pred_dv, 2), macro_f1(ydv, sk_dv, 2)))
print()
print("the two models agree on %.2f%% of training beats" % (100 * (pred_tr == sk_tr).mean()))
print("GATE (gap must be under 0.01):", "PASS" if gap < 0.01 else "FAIL")
"""),
    md("""
## 4. What we learned

Look at the dev numbers, not the train numbers.
"""),
    code("""
print("OUR MODEL ON DEV:")
print(report(ydv, pred_dv, class_names("NV")))
"""),
    md("""
Two things are surprising here, and both are important.

**First, dev accuracy went DOWN.** The lazy model had 0.92 and our trained model has
about 0.88. But macro-F1 went **up**, from 0.48 to about 0.59. The model now finds
some V beats, and it pays for that with a few wrong answers on normal beats. This is
a better model, even if accuracy says the opposite. This is exactly why we chose
macro-F1 at the start.

**Second, train is much better than dev.** On train we get about 0.96 macro-F1, on
dev only about 0.59. The reason is our split: train and dev contain **different
patients**. The model learns the hearts it saw, and a new heart is harder. If we had
split the beats randomly, both numbers would be near 0.99 and we would learn nothing
from them.
"""),
]

# ---------------------------------------------------------------- notebook 03
nb03 = [
    md("""
# 03 - A network with one hidden layer

Phase 2, part 2.

One neuron can only draw a straight line between the two classes. A hidden layer
lets the model bend that line. Here we ask a simple question:

**does a hidden layer help, and how big should it be?**

We try 3 sizes (4, 16, 64 units) and 2 activations (tanh and relu).
We run every combination **3 times with different seeds**. You will see why.
"""),
    code(HEADER + """
from dataclasses import replace
from ecg.train import train, load_results
"""),
    md("""
## 1. One training, to see the shape of it

`train(config)` does everything: load, build the weights, run the epochs, measure,
and write one line into `results/results.csv`.
"""),
    code("""
cfg = Config(classes="NV", output_activation="sigmoid", optimizer="gd",
             hidden_dims=(16,), hidden_activation="tanh", init="he",
             learning_rate=0.1, batch_size=64, epochs=10, seed=1,
             phase="2-demo", note="one shallow run, shown in the notebook")

out = train(cfg, verbose=True, log=False)
"""),
    code("""
fig = plots.learning_curves(out["history"], title="One hidden layer, 16 units, tanh")
plt.show()
"""),
    md("""
Look at the gap between the two lines on the right plot. Train keeps going up,
dev stops early and stays flat. The model is learning the training patients by
heart. More epochs will not fix that.

## 2. The sweep

The full sweep is 3 sizes x 2 activations x 3 seeds = 18 runs, about 6 minutes.
It was already run, and every result is inside `results/results.csv`.
So here we only read the file.

If you want to run it again, use the loop in the next cell.
"""),
    code("""
# HOW THE SWEEP WAS RUN (it takes about 6 minutes, so it is turned off here)
#
# base = Config(classes="NV", output_activation="sigmoid", optimizer="gd",
#               init="he", learning_rate=0.1, batch_size=64, epochs=30,
#               phase="2", note="shallow sweep")
# for size in [4, 16, 64]:
#     for act in ["tanh", "relu"]:
#         for seed in [1, 2, 3]:
#             train(replace(base, hidden_dims=(size,), hidden_activation=act,
#                           seed=seed), verbose=False)

df = load_results()
sweep = df[df["note"] == "shallow sweep"]
print("runs in the sweep:", len(sweep))
"""),
    code("""
table = (sweep.groupby(["hidden_dims", "hidden_activation"])["dev_macro_f1"]
              .agg(["mean", "std", "min", "max"])
              .round(4)
              .sort_values("mean", ascending=False))
print(table.to_string())
"""),
    md("""
## 3. The seeds matter more than the settings

Look at the `min` and `max` columns.

For 4 units with tanh, the worst seed gives about 0.64 and the best gives about
0.79. That is a very big difference, and **nothing changed except the starting
random weights**.

Now compare the `mean` column. The distance between the best setting and the worst
setting is smaller than the distance between two seeds of the same setting.

This is the reason we always run 3 seeds. With only one run per setting we could
"prove" almost anything we want.
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 4.5))
for act, colour in [("tanh", "#2c6fbb"), ("relu", "#c0392b")]:
    part = sweep[sweep["hidden_activation"] == act]
    g = part.groupby("hidden_dims")["dev_macro_f1"].agg(["mean", "std"])
    g = g.reindex([4, 16, 64])
    ax.errorbar(range(3), g["mean"], yerr=g["std"], marker="o", capsize=5,
                label=act, color=colour, lw=2)
    ax.scatter(np.repeat(range(3), 3) + (0.06 if act == "relu" else -0.06),
               part.sort_values("hidden_dims")["dev_macro_f1"],
               color=colour, alpha=0.35, s=22)
ax.set_xticks(range(3), ["4", "16", "64"])
ax.set_xlabel("units in the hidden layer")
ax.set_ylabel("dev macro-F1")
ax.set_title("Bars are 3 seeds. The spread is bigger than the difference.")
ax.legend()
ax.grid(alpha=0.25)
plots.save(fig, "phase2_shallow_sweep")
plt.show()
"""),
    md("""
## 4. What we learned

- **tanh is better than relu here**, at every size, by about 0.05. This is small but
  it happens every time, so we believe it.
- **The size of the hidden layer changes almost nothing.** 4 units are as good as 64.
- Train macro-F1 is near 0.99 for every single run. So the model is big enough
  already. The problem is not the model, the problem is that we only have **17
  patients** in train. A bigger brain does not help you if you never met enough
  people.
"""),
]

# ---------------------------------------------------------------- notebook 04
nb04 = [
    md("""
# 04 - Deep network, and checking the gradient

Phase 2, part 3.

Two jobs in this notebook:

1. **Check that our backward pass is correct.** This must be done before we believe
   any result.
2. Train a deeper network and compare all three models.
"""),
    code(HEADER + """
from dataclasses import replace
from ecg.train import train, load_results
from ecg.init import initialize_parameters
from ecg.model import forward, backward
from ecg.gradcheck import gradient_check
"""),
    md("""
## 1. Gradient checking

This is the most important cell of Phase 2.

The idea is simple. We take one weight, we push it up by a very small number, and we
look at the loss. Then we push it down and look again. The difference tells us the
slope. If our backward pass computed the same slope, our maths is correct.

Why it matters: **a wrong gradient still gives a nice loss curve**. The model still
trains, the loss still goes down, the plot still looks fine. You can lose a week
before you notice.

Three rules when you run it:

- use a **tiny** network, because we need two forward passes for every weight
- turn dropout **off**, because it makes the loss random
- use **tanh**, not relu, because relu has a corner at 0 and gives false alarms
"""),
    code("""
rng = np.random.default_rng(0)
X_small = rng.standard_normal((8, 6))
Y_small = (rng.random((1, 6)) > 0.5).astype(float)

for dims in [[8, 1], [8, 5, 1], [8, 5, 4, 1]]:
    p = initialize_parameters(dims, "he", seed=1)
    AL, cache = forward(X_small, p, "tanh", "sigmoid")
    grads = backward(AL, Y_small, cache, p, "tanh", "sigmoid")
    print(str(dims) + " ->", end=" ")
    gradient_check(p, grads, X_small, Y_small, "tanh", "sigmoid")
"""),
    md("""
### Does the check really work?

A test that always says PASS is useless. So we break the gradient on purpose, only
by 1%, which is the size of a real bug like a wrong sign or a missing transpose.
The check must catch it.
"""),
    code("""
p = initialize_parameters([8, 5, 1], "he", seed=5)
AL, cache = forward(X_small, p, "tanh", "sigmoid")
grads = backward(AL, Y_small, cache, p, "tanh", "sigmoid")

grads["dW1"] = grads["dW1"] * 1.01          # a small bug on purpose

err = gradient_check(p, grads, X_small, Y_small, "tanh", "sigmoid", verbose=False)
print("relative error with a 1%% bug: %.3e" % err)
print("threshold is 1e-07, so the check catches it:", err > 1e-7)
"""),
    md("""
Good. A 1% mistake gives an error about 10000 times bigger than the limit.
Now we can trust the numbers.

## 2. The three models together

All the runs are already in `results/results.csv`. We read them and compare.
"""),
    code("""
df = load_results()
df["hidden_dims"] = df["hidden_dims"].fillna("").astype(str)

# we keep only tanh, and for the shallow model only the 16 unit version
is_logistic = df["note"] == "model 1 logistic regression"
is_deep = df["note"] == "model 3 deep network"
is_shallow = (df["note"] == "shallow sweep") & (df["hidden_dims"] == "16")
three = df[(is_logistic | is_deep | is_shallow) &
           (df["hidden_activation"] == "tanh")].copy()

def label(note):
    if note == "model 1 logistic regression":
        return "1. logistic [254, 1]"
    if note == "model 3 deep network":
        return "3. deep [254, 64, 32, 1]"
    return "2. shallow [254, 16, 1]"

three["model"] = three["note"].apply(label)
print("runs used:", len(three))

summary = (three.groupby("model")[["train_macro_f1", "dev_macro_f1", "dev_acc"]]
                .agg(["mean", "std"]).round(4))
print(summary.to_string())
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 4.5))
order = sorted(three["model"].unique())
means_tr = [three[three["model"] == m]["train_macro_f1"].mean() for m in order]
means_dv = [three[three["model"] == m]["dev_macro_f1"].mean() for m in order]
stds_dv = [three[three["model"] == m]["dev_macro_f1"].std() for m in order]
x = np.arange(len(order))
ax.bar(x - 0.2, means_tr, 0.4, label="train", color="#9dc3e6")
ax.bar(x + 0.2, means_dv, 0.4, yerr=stds_dv, capsize=5, label="dev", color="#2c6fbb")
ax.axhline(0.48, color="#c0392b", ls="--", lw=1.5, label="always say normal (0.48)")
ax.set_xticks(x, [m[3:] for m in order], fontsize=8)
ax.set_ylabel("macro-F1")
ax.set_title("Train is almost perfect. Dev is not. That gap is the real story.")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
plots.save(fig, "phase2_three_models")
plt.show()
"""),
    md("""
## 3. What we learned

**Every model beats the floor of 0.48.** So all of them learned something real.

**The deep model is not the best.** The order is:

1. shallow (one hidden layer) - best
2. deep - middle
3. logistic regression - worst

This is not what people expect. Deeper is supposed to be better. But look at the
train column: the deep model gets 0.9996 on train. It knows the training patients
almost perfectly and it still fails on new ones. Adding layers only helped it to
memorise faster.

**The reason is the data, not the model.** We have 40496 training beats, but they
come from only **17 people**. A bigger model cannot invent new patients.

**One more thing about the seeds.** Logistic regression gives almost the same answer
every time (std 0.005). The networks jump around a lot more (std 0.03 to 0.065). The
reason is that logistic regression has only one best answer, so the starting point
does not matter. A network has many, and the random start decides which one you
find.
"""),
]

write("02_logistic_regression", nb02)
write("03_shallow_nn", nb03)
write("04_deep_nn_gradcheck", nb04)
