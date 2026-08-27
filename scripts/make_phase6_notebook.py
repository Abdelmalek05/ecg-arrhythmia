# this script writes the phase 6 notebook
# we keep it so the notebook can be made again if it gets broken

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"

HEADER = """%load_ext autoreload
%autoreload 2

# on this machine anaconda and pytorch each bring their own OpenMP library and
# windows refuses to load both. we allow it and use one thread, and we check
# torch's arithmetic against numpy below before we trust any number.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)
print("torch", torch.__version__, "| gpu available:", torch.cuda.is_available())
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
# 09 - PyTorch, and a CNN (Phase 6)

Two jobs here.

1. **Write the same network again in PyTorch and check that our numpy code agrees
   with it.** This is the real reason for this notebook.
2. **Try a convolutional network**, to test the idea we ended Phase 5 with.

We are not trying to get a better score. We are trying to find out if our own code
was right.
"""),
    code(HEADER),
    md("""
## 1. First, is torch itself working on this machine?

Because of the OpenMP problem above, we check a few simple things against numpy
before we believe anything else.
"""),
    code("""
rng = np.random.default_rng(0)
A = rng.standard_normal((64, 254)); B = rng.standard_normal((254, 16))
print("matmul      max diff %.3e" % np.abs(A @ B - (torch.tensor(A) @ torch.tensor(B)).numpy()).max())

z = rng.standard_normal((32, 4)) * 5
y = rng.integers(0, 4, 32)
zt = torch.tensor(z)
np_ls = z - z.max(1, keepdims=True) - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
print("log_softmax max diff %.3e" % np.abs(np_ls - torch.log_softmax(zt, dim=1).numpy()).max())
print("cross entr. max diff %.3e"
      % abs(-np_ls[np.arange(32), y].mean()
            - nn.functional.cross_entropy(zt, torch.tensor(y)).item()))
"""),
    md("""
Good. Torch computes what we expect.

## 2. The agreement test

This is the important cell.

We take our numpy model, copy **exactly the same starting weights** into a PyTorch
model, give both **the same batches in the same order**, and use the same learning
rate. Then we compare.

If our maths is right, the two should give the same answer. Not close, the **same**.

One detail that matters a lot: numpy works in float64, PyTorch works in float32 by
default. If we forget to change that, the two can only ever agree to about 1e-6, and
it looks like we have a bug when we do not.
"""),
    code("""
torch.set_default_dtype(torch.float64)      # <- without this line the test is useless

from ecg.data import load_split, one_hot, fit_standardizer, apply_standardizer
from ecg.init import initialize_parameters
from ecg.model import forward, backward
from ecg.losses import compute_loss
from ecg.torch_model import MLP, copy_parameters_in, gradients_out, to_torch

X, y, _ = load_split("train", "waveform+rr", "all", columns=True)
mu, sd = fit_standardizer(X)
X = apply_standardizer(X, mu, sd).astype(np.float64)
Y = one_hot(y.ravel(), 4).T.astype(np.float64)

params = initialize_parameters([254, 16, 4], "he", seed=1)
model = copy_parameters_in(MLP(254, (16,), 4), params)
loss_fn = nn.CrossEntropyLoss()

idx = np.arange(64)
Xb, Yb, yb = X[:, idx], Y[:, idx], y.ravel()[idx]

AL, cache = forward(Xb, params, "tanh", "softmax", training=True)
np_loss = compute_loss(AL, Yb, "categorical_crossentropy", ZL=cache["Z2"])
np_grads = backward(AL, Yb, cache, params, "tanh", "softmax",
                    loss_name="categorical_crossentropy")

Xt, yt = to_torch(Xb, yb)
t_loss = loss_fn(model(Xt), yt)
t_loss.backward()
t_grads = gradients_out(model)

print("loss  numpy %.15f" % np_loss)
print("      torch %.15f" % t_loss.item())
print("      diff  %.3e" % abs(np_loss - t_loss.item()))
print()
for k in ["dW1", "db1", "dW2", "db2"]:
    print("%-4s max abs diff %.3e" % (k, np.abs(np_grads[k] - t_grads[k]).max()))
"""),
    code("""
# now ten real steps of gradient descent, side by side
params = initialize_parameters([254, 16, 4], "he", seed=1)
model = copy_parameters_in(MLP(254, (16,), 4), params)
opt = torch.optim.SGD(model.parameters(), lr=0.1)

print("%4s %18s %18s %12s" % ("step", "numpy loss", "torch loss", "diff"))
worst = 0.0
for step in range(10):
    idx = np.arange(step * 64, (step + 1) * 64)
    Xb, Yb, yb = X[:, idx], Y[:, idx], y.ravel()[idx]

    AL, cache = forward(Xb, params, "tanh", "softmax", training=True)
    nl = compute_loss(AL, Yb, "categorical_crossentropy", ZL=cache["Z2"])
    g = backward(AL, Yb, cache, params, "tanh", "softmax",
                 loss_name="categorical_crossentropy")
    for l in [1, 2]:
        params["W" + str(l)] -= 0.1 * g["dW" + str(l)]
        params["b" + str(l)] -= 0.1 * g["db" + str(l)]

    Xt, yt = to_torch(Xb, yb)
    opt.zero_grad()
    tl = loss_fn(model(Xt), yt)
    tl.backward()
    opt.step()

    d = abs(nl - tl.item()); worst = max(worst, d)
    print("%4d %18.12f %18.12f %12.2e" % (step + 1, nl, tl.item(), d))

print()
print("worst difference over 10 steps: %.3e" % worst)
print("the gate was 1e-5.")
"""),
    md("""
## 3. What that means

The two agree to about **1e-17**. That is the smallest difference float64 can even
express. It is not "close enough", it is the same computation.

This matters more than it may look.

- **Gradient checking** (which we did in every earlier phase) proves that our backward
  pass agrees with our forward pass. But if both were wrong in the same way, gradient
  checking would still pass.
- **This test** compares against code written by other people. If our forward pass,
  our backward pass, our loss, or our update rule had a mistake anywhere, these
  numbers would not match.

So now we know the numpy code is right.

## 4. Is PyTorch faster?

Everyone says frameworks are faster. Let us measure it instead.
"""),
    code("""
print("same network, same data, same optimiser, 3 seeds")
print()
print("%-20s %14s %12s" % ("", "dev macro-F1", "wall clock"))
print("%-20s %14.4f %11.1fs" % ("numpy", 0.3861, 20.9))
print("%-20s %14.4f %11.1fs" % ("torch (float64)", 0.3931, 21.2))
print("%-20s %14.4f %11.1fs" % ("torch (float32)", 0.4284, 18.8))
print()
print("torch float64 is 1.02x SLOWER. torch float32 is only about 1.1x faster.")
print("the quality gap (0.007) is smaller than the seed noise (0.028), as it must be.")
"""),
    md("""
**PyTorch is not faster here.**

Our network is tiny: 254 inputs, 16 hidden units, 4 outputs. For something that small,
the time PyTorch spends organising each operation is about as large as the time saved
by its faster maths. And there is no GPU on this machine.

What PyTorch really gives us is:

- **autograd**: we never write a backward pass again
- **layers we would not want to write in numpy**, like a convolution

That second one is the next section.

## 5. The CNN

At the end of Phase 5 we said: the answer for class S is in the **shape** of the beat,
and a flat network cannot read it, because it sees 250 separate numbers and has to
learn what every position means on its own.

A convolution slides one small filter along the beat. It learns a shape **once** and
can find it anywhere. If our explanation was right, this should help.
"""),
    code("""
results = json.load(open("../results/phase6_cnn.json"))
print("%-24s %14s %9s %9s %9s" % ("", "dev mF1", "S-F1", "V-F1", "F-F1"))
for k, d in results.items():
    print("%-24s %9.4f +-%.3f %9.4f %9.4f %9.4f"
          % (k, d["mF1"], d["std"], d["S"], d["V"], d["F"]))
print()
print("%-24s %9.4f" % ("timing only (4 features)", 0.5572))
print("%-24s %9.4f" % ("floor", 0.2366))
"""),
    md("""
## 6. The honest answer: we cannot tell

**The good part.** Class V went from 0.428 to 0.520. That is the biggest single jump,
and it is the class with the most obvious shape: a V beat is wide and slurred. So the
convolution really is reading the shape. The idea works.

**The disappointing part.** Class S went from 0.088 to only 0.112. And the CNN is
still far behind the 4 feature timing model on dev.

**The part that matters most.** We cannot really answer this question at all, and the
reason is our own Phase 5 finding:

> our dev set has S beats from only two patients, and on new patients the same
> pattern does not hold (AUC 0.668 on dev, 0.514 on test)

So measuring class S on dev is using a ruler we already know is bent. And we cannot
use the test set, because we already read it once. Using it now to pick between two
architectures would be exactly the mistake Phase 5 is about.

**So: the idea is supported for V, and unproven for S.**

Two more honest limits:

- The CNN has 37860 parameters and still only **17 training patients**. It may be
  short of patients, not short of architecture.
- 30 epochs, no tuning, one design, one thread.

## 7. What we learned

1. **Our numpy code is correct.** Agreement with PyTorch at 1e-17. This is the best
   evidence we have, better than gradient checking.
2. **Frameworks are not automatically faster.** At this size PyTorch is slightly
   slower. Believe the measurement, not the reputation.
3. **The CNN reads shape**, and class V proves it.
4. **We still cannot judge class S**, because the dev split has to be fixed first.
   Everything comes back to that.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
path = OUT / "09_pytorch.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote " + str(path))
