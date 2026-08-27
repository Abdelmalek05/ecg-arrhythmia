# this script writes the phase 5 notebook
# we keep it so the notebook can be made again if it gets broken

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"

HEADER = """%load_ext autoreload
%autoreload 2

import json
import numpy as np
import matplotlib.pyplot as plt

from ecg.data import load_split, class_names
from ecg.metrics import report, macro_f1, accuracy, confusion_matrix, per_class, majority_baseline
from ecg import plots

results = json.load(open("../results/phase5_test_results.json"))
y_true = np.array(results["_best_true"])
y_pred = np.array(results["_best_pred"])
records = np.array(results["_best_rec"])
names = class_names("all")
print("test beats:", y_true.size)
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
# 08 - Error analysis and the test set (Phase 5)

This is the last notebook with numbers in it. Here we finally open the **test set**.

We only do this once. Every choice we made until now was made by looking at the dev
set. If we now look at the test set and then change our mind, the test number stops
meaning anything.

The full write up is in `reports/strategy_memo.md`.
"""),
    code(HEADER),
    md("""
## 1. The result

We took three models to the test set. All three were chosen before we looked.
"""),
    code("""
print("%-24s %10s %12s %10s" % ("model", "dev mF1", "TEST mF1", "change"))
for label in ["timing only (4)", "shape + timing (254)", "shape only (250)"]:
    r = results[label]
    print("%-24s %10.4f %12.4f %+10.4f"
          % (label, r["dev_macro_f1"], r["test_macro_f1"],
             r["test_macro_f1"] - r["dev_macro_f1"]))
print()
print("floor on test: accuracy %.4f, macro-F1 %.4f"
      % (majority_baseline(y_true, 4), macro_f1(y_true, np.zeros_like(y_true), 4)))
"""),
    md("""
## 2. Something went wrong

Look again at that table.

**The model we chose is the worst one on the test set.** The 254 feature model, which
looked clearly worse on dev, is better on test by about 0.06.

We are **not** going to change our answer now. If we pick the winner after seeing the
test set, then the test set becomes a second dev set and we have no honest number
left. So we keep our choice, we report 0.4193, and we spend the rest of this notebook
finding out **why we were wrong**.

That is more useful than the score anyway.

## 3. What the chosen model actually does
"""),
    code("""
print(report(y_true, y_pred, names))
"""),
    code("""
cm = confusion_matrix(y_true, y_pred, 4)
fig = plots.confusion(cm, names, title="Chosen model (timing only) on the TEST set")
plt.show()

print("of %d true S beats:" % cm[1].sum())
for i, n in enumerate(names):
    print("   called %s: %5d  (%.1f%%)" % (n, cm[1][i], 100 * cm[1][i] / cm[1].sum()))
"""),
    md("""
**65% of all S beats were called V.**

On dev, class S had an F1 of 0.628. On test it is 0.053. Everything else stayed the
same:

| class | dev | test |
|---|---|---|
| N | 0.9737 | 0.9702 |
| **S** | **0.6280** | **0.0535** |
| V | 0.6413 | 0.6647 |
| F | 0.0000 | 0.0000 |

So the whole drop is one class.

## 4. Why: timing cannot tell S from V

Our model only sees **when** a beat arrives, not what it looks like.

The problem is that **S beats and V beats are both early**. Timing can say "this beat
came too soon". It cannot say which of the two kinds it is. For that you need the
shape.

Let us measure how well the timing feature separates S from V, on each split.
"""),
    code("""
def auc(pos, neg):
    all_v = np.concatenate([pos, neg])
    order = all_v.argsort()
    ranks = np.empty(len(all_v)); ranks[order] = np.arange(1, len(all_v) + 1)
    a = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return max(a, 1 - a)

print("AUC of pre_rr/local_rr, S against V.  0.5 = useless, 1.0 = perfect")
for split in ["train", "dev", "test"]:
    X, y, rec = load_split(split, features="rr", classes="all", columns=True)
    ratio = X[3]
    s, v = ratio[y.ravel() == 1], ratio[y.ravel() == 2]
    print("   %-6s S median %.3f   V median %.3f   AUC %.3f   (n_S = %d)"
          % (split, np.median(s), np.median(v), auc(s, v), len(s)))
"""),
    md("""
On dev the feature works (0.668). On test it is **almost a coin flip (0.514)**.

Same feature, same code. Different patients.

## 5. The real mistake: our dev set had S beats from only two people
"""),
    code("""
for split in ["dev", "test"]:
    X, y, rec = load_split(split, features="rr", classes="all", columns=True)
    ratio = X[3]
    print(split.upper() + "  (patients with 20 or more S beats)")
    rows = []
    for r in sorted(set(rec.tolist())):
        m = (rec == r) & (y.ravel() == 1)
        if m.sum() >= 20:
            rows.append((int(m.sum()), float(np.median(ratio[m])), r))
    for n, med, r in sorted(rows, reverse=True):
        print("   patient %s: %5d S beats, median ratio %.3f" % (r, n, med))
    print()
"""),
    md("""
Read the numbers.

In **dev**, most S beats come from patient 207, and that patient's S beats have a
ratio of **0.992**, which means they arrive almost exactly on time. So on dev the rule
"S is a bit early, V is very early" works.

In **test**, most S beats come from patient 232, and that patient's S beats have a
ratio of **0.738**. V beats sit at 0.726. They are the same.

Our model learned a rule that was true for **two people**. Dev could not tell us it
was wrong, because dev *was* those two people.

> **A dev set needs enough patients per class, not enough beats per class.**

218 S beats sounds like enough. Two S patients is not.

## 6. Looking at the mistakes

Let us plot some of the S beats that were called V, and put the average S beat and the
average V beat behind them.
"""),
    code("""
Xw, _, _ = load_split("test", features="waveform", classes="all", columns=True)
Xw = Xw.T
Xtr, ytr, _ = load_split("train", features="waveform", classes="all", columns=True)
Xtr = Xtr.T
mean_S = Xtr[ytr.ravel() == 1].mean(axis=0)
mean_V = Xtr[ytr.ravel() == 2].mean(axis=0)

mistakes = np.where((y_true == 1) & (y_pred == 2))[0]
d_s = np.linalg.norm(Xw[mistakes] - mean_S, axis=1)
d_v = np.linalg.norm(Xw[mistakes] - mean_V, axis=1)

print("of the %d S beats called V:" % len(mistakes))
print("   %.1f%% are closer in SHAPE to the average S beat than to the average V beat"
      % (100 * (d_s < d_v).mean()))
print("   average distance to the S shape: %.2f" % d_s.mean())
print("   average distance to the V shape: %.2f" % d_v.mean())
"""),
    code("""
fig = plt.imread("../results/figures/phase5_errors.png")
plt.figure(figsize=(15, 8)); plt.imshow(fig); plt.axis("off"); plt.show()
"""),
    md("""
Almost every one of them looks like an S beat: a narrow, sharp spike. The wide
rounded V shape is not there.

**The answer was in the signal. We threw it away when we chose the features.**

## 7. Bias and variance
"""),
    code("""
print("%-24s %8s %8s %8s %12s" % ("model", "train", "dev", "TEST", "train - dev"))
for label, tr in [("timing only (4)", 0.5634),
                  ("shape + timing (254)", 0.9567),
                  ("shape only (250)", 0.9310)]:
    r = results[label]
    print("%-24s %8.4f %8.4f %8.4f %12.4f"
          % (label, tr, r["dev_macro_f1"], r["test_macro_f1"], tr - r["dev_macro_f1"]))
"""),
    md("""
The two models are at **opposite ends**.

**Timing only**: train 0.5634, dev 0.5572. The gap is only 0.006, so there is almost
no variance. But it only reaches 0.56 **on the data it was trained on**. It cannot fit
even that. This is **high bias**: 4 numbers are not enough to describe the problem.
More data will not help. More regularisation will not help.

**254 features**: train 0.9567, dev 0.3861. A gap of 0.57. It learns the 17 training
patients almost perfectly and then fails on new people. This is **high variance**. The
cure is more data, and here that means **more patients**, not more beats.

In Phase 4 we ran 174 experiments on regularisation and optimisers. We were tuning the
**high bias** model, where none of those tools do anything. That is exactly what we
saw: nothing moved the score by more than 0.05.

## 8. Per patient
"""),
    code("""
pp = results["_per_patient"]
vals = np.array(list(pp.values()))
print("macro-F1 for each of the 22 test patients:")
print("   min %.3f   median %.3f   max %.3f" % (vals.min(), np.median(vals), vals.max()))
print()
for r, v in sorted(pp.items(), key=lambda kv: kv[1]):
    bar = "#" * int(v * 40)
    print("   %s  %.3f  %s" % (r, v, bar))
"""),
    md("""
One number hides a lot. Patients 212 and 105 get 1.000, but that is because almost all
of their beats are normal, so there is nothing to get wrong. Patient 232, who has 75%
of all the S beats in the test set, gets 0.312.

If this were a real medical tool, the **worst** patient would matter more than the
average one.

## 9. What we learned

1. **Our dev set chose the wrong model.** Not by a small amount, and not by bad luck.
2. **The reason was patients, not beats.** Dev had 218 S beats but only two people who
   really had S beats.
3. **The fix comes first, before anything else.** Choose the dev split by patients per
   class, or rotate the patients (cross validation).
4. **We tuned the wrong model.** All the Phase 4 work went into a high bias model.
5. **The shape of the beat carries the answer**, and a flat network on raw samples
   cannot read it. That is an architecture problem, and it points straight at Course 4.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
path = OUT / "08_error_analysis.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote " + str(path))
