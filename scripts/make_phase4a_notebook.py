# this script writes the phase 4a notebook
# we keep it so the notebook can be made again if it gets broken

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"

HEADER = """%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ecg.train import load_results

FLOOR = 0.2366          # what "always say normal" gets on dev

df = load_results().query("phase == '4a'").copy()
f1 = df["dev_f1_per_class"].str.split("|", expand=True).astype(float)
f1.columns = ["f1_N", "f1_S", "f1_V", "f1_F"]
df = pd.concat([df, f1], axis=1)
print("phase 4a runs:", len(df))
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
# 06 - The ablation lab (Phase 4a)

Now we have a model that works. In this notebook we change **one thing at a time**
and see what helps.

138 runs, 3 seeds each. Everything is already in `results/results.csv`, so here we
only read and compare.

Two numbers to keep in your head:

- **0.2366** is the floor. A model that always says "normal" gets this.
- **0.4254** was the best we had at the end of Phase 3.
"""),
    code(HEADER),
    md("""
## 1. One thing at a time

Every run below starts from the same base config and changes only one setting.
"""),
    code("""
def axis(note, column, order):
    d = df[df.note == note]
    g = d.groupby(column)["dev_macro_f1"].agg(["mean", "std"]).reindex(order).round(4)
    return g

print("INITIALISATION")
print(axis("axis1 init", "init", ["he", "xavier", "random", "zeros"]).to_string())
print()
print("L2")
print(axis("axis2 l2", "l2", [0.01, 0.0, 0.1, 1.0]).to_string())
print()
print("DROPOUT (keep_prob)")
print(axis("axis3 dropout", "keep_prob", [1.0, 0.8, 0.5]).to_string())
print()
print("BATCH SIZE (0 means the whole training set at once)")
print(axis("axis4 batchsize", "batch_size", [64, 512, 8, 0]).to_string())
print()
print("LEARNING RATE DECAY")
print(axis("axis6 lrdecay", "lr_decay", ["exponential", "none", "inverse"]).to_string())
"""),
    code("""
fig = plt.imread("../results/figures/phase4a_axes.png")
plt.figure(figsize=(14, 7)); plt.imshow(fig); plt.axis("off"); plt.show()
"""),
    md("""
### What these say

**Zeros initialisation gives exactly 0.2366, the floor, with a standard deviation of
zero.** Every hidden unit starts the same, so every gradient is the same, so they
stay the same forever. The network can only ever say "normal". This is the clearest
result in the whole project: it is not "a bit worse", it is *nothing at all*.

**The full batch also fails** (0.256). With the whole training set in one batch and
30 epochs, the model only gets **30 updates**. With batches of 64 it gets about
19000. Small batches are not only about memory, they are about how many steps you
take.

**L2 and dropout do not help.** L2 of 0.01 is the same as no L2. Bigger values only
make it worse, and L2 of 1.0 pushes it back to the floor. Dropout hurts every time.

That is not what people expect from regularisation, but it makes sense here. Our
problem is not that the model is too confident about the classes. Our problem is
that it almost never predicts the rare classes at all. Regularisation makes a model
*less* sure, and that is the wrong direction.

**Learning rate decay changes nothing.** 0.4024 against 0.4016 is inside the noise.

## 2. Does the optimiser rescue the quiet features?

At the end of Phase 3 we found the problem: the 4 timing numbers are about 1000
times quieter than the 250 shape numbers, so the model does not hear them.

Our guess was that **Adam would fix it**, because Adam gives every weight its own
step size.

Let us look.
"""),
    code("""
g = df[df.note == "gridA feat-opt-std"]
table = (g.groupby(["features", "optimizer", "standardize"])[["dev_macro_f1", "f1_S"]]
          .agg(["mean", "std"]).round(4))
print(table.to_string())
"""),
    code("""
fig = plt.imread("../results/figures/phase4a_gridA.png")
plt.figure(figsize=(14, 5)); plt.imshow(fig); plt.axis("off"); plt.show()
"""),
    md("""
## 3. Our guess was wrong

Look at the "shape + timing" bars for class S.

- gd, not scaled: **0.017**
- adam, not scaled: **0.019**
- gd, scaled: **0.133**
- adam, scaled: **0.105**

**Adam did not help. Scaling the features did.** And plain gd with scaling is even a
little better than Adam with scaling.

So the fix was never the optimiser. It was that the numbers had different sizes.

### Why we were wrong

Adam does give each weight its own step size, and we showed that on a simple bowl
in `optimizers.py`. But here the problem is not that the *gradients* are badly
scaled. The problem is that the *inputs* are. A quiet input makes a small
contribution no matter how big a step the weight takes, so a better optimiser
cannot bring it back.

## 4. The bigger surprise

Now look at the third group of bars: **timing only, 4 features.**
"""),
    code("""
for feats, label in [("rr", "timing only (4)"),
                     ("waveform+rr", "shape + timing (254)"),
                     ("waveform", "shape only (250)")]:
    d = g[(g.features == feats) & (g.optimizer == "gd") & (g.standardize == True)]
    print("%-22s macro-F1 %.4f +/- %.4f   S-F1 %.4f   V-F1 %.4f   F-F1 %.4f"
          % (label, d["dev_macro_f1"].mean(), d["dev_macro_f1"].std(),
             d["f1_S"].mean(), d["f1_V"].mean(), d["f1_F"].mean()))
"""),
    md("""
**Four numbers beat two hundred and fifty four.**

And it is not close: 0.557 against 0.386. The gap is much bigger than the seed
noise, which is only about 0.003 for the timing model.

Even more strange: the timing features are also better at class **V** (0.64 against
0.45). We thought V was a *shape* class, because a V beat is wide and easy to see.
But V beats also come early, and it turns out that being early is easier for the
model to use than being wide.

**So the 250 waveform numbers are not just useless next to the timing ones, they
actively hurt.** Adding them drops macro-F1 by about 0.17.

## 5. Class weights

We also tried giving the rare classes a bigger weight in the loss.
"""),
    code("""
b = df[df.note == "gridB weights-features"]
print(b.groupby(["class_weights", "features"])[["dev_macro_f1", "f1_S", "f1_F"]]
       .mean().round(4).to_string())
"""),
    md("""
Class weights made things **worse**, for every feature set.

The likely reason is class F. It gets the biggest weight of all, because it is the
rarest. But F is not learnable here: 382 training beats, and 372 of them come from
one single patient. So the loss spends a lot of effort on a class it cannot get
right, and the other classes pay for it.

## 6. Where we are now
"""),
    code("""
best = df.sort_values("dev_macro_f1", ascending=False).head(5)
print(best[["features", "optimizer", "learning_rate", "standardize",
            "class_weights", "seed", "dev_macro_f1", "f1_S", "f1_V", "f1_F"]]
      .to_string(index=False))
print()
print("floor           0.2366")
print("phase 3 best    0.4254")
print("phase 4a best   %.4f" % best["dev_macro_f1"].max())
"""),
    md("""
## 7. What we learned

1. **The input matters much more than the training tricks.** Changing the features
   moved the score by 0.24. Every optimiser, every regulariser, every learning rate
   moved it by less than 0.05, and most by less than the seed noise.

2. **Our Adam guess was wrong, and being wrong was useful.** It told us the problem
   is the size of the inputs, not the size of the steps.

3. **Two settings really do break the model**: zeros initialisation (no learning at
   all) and full batch (only 30 updates). Both are C2 lessons that we can now say we
   have seen with our own eyes.

4. **Regularisation did not help.** The model is not too confident, it is too shy.

5. **Class F is still not learnable.** No setting fixed it, and we should stop trying.

6. **The 250 raw samples are a weak way to describe a beat.** A flat network reading
   raw samples one by one is simply not a good tool for shapes. This is exactly why
   the papers on this dataset use convolutional networks. We chose not to use them,
   and now we can measure what that choice costs.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
path = OUT / "06_ablations.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote " + str(path))
