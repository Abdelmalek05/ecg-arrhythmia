# ECG Arrhythmia Classification — from scratch in NumPy

Classifying individual heartbeats into four arrhythmia types, with the neural network
written from first principles: forward and backward propagation, initialisation
schemes, optimisers and regularisation all implemented directly in NumPy rather than
called from a framework.

A rehearsal of Courses 1–3 of the Deep Learning Specialization, on real clinical data.

---

## The problem

Each example is one heartbeat, taken from a 30-minute ECG recording and labelled by a
cardiologist. The task is to identify what kind of beat it is:

| class | what it is | what gives it away |
|---|---|---|
| **N** | normal | narrow, sharp QRS complex; arrives on schedule |
| **S** | supraventricular ectopic | **shape is nearly identical to N** — only the early arrival betrays it |
| **V** | ventricular ectopic | wide, slurred QRS; the signal never used the fast conduction path |
| **F** | fusion | a normal and a ventricular beat collided; morphology sits between the two |

Class S is the interesting one. Because its waveform looks like a normal beat, a model
given only the signal shape is structurally unable to find it — no amount of training
fixes a feature set that does not contain the answer. Four R-R interval features
(measuring how early the beat arrived relative to the patient's own rhythm) are
extracted alongside the waveform for exactly this reason, and the effect of adding them
is one of the ablations.

**Input:** 250 waveform samples (−0.25 s to +0.44 s around the R-peak, 360 Hz, lead MLII)
+ 4 R-R timing features = **254 features**.

---

## The dataset

Built from the raw [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
on PhysioNet — 48 half-hour recordings, ~109,000 expert-annotated beats.

**The splits share no patients.** Each recording contributes ~2,300 near-identical beats
from one person, so splitting beats at random would put ~1,800 near-twins of every test
beat into training. A model can then score ~99% by recognising individuals rather than
arrhythmias, and both train and dev error collapse to ~1%, leaving nothing to diagnose.
Splitting by *patient* — the standard inter-patient protocol — keeps the evaluation
honest: at test time the model sees hearts it has never encountered.

| split | patients | N | S | V | F | total | majority baseline |
|---|---|---|---|---|---|---|---|
| train | 17 | 36,418 | 726 | 2,970 | 382 | 40,496 | 89.9% |
| dev | 5 | 9,430 | 218 | 818 | 32 | 10,498 | 89.8% |
| test | 22 | 44,240 | 1,837 | 3,220 | 388 | 49,685 | 89.0% |

Two consequences worth stating up front:

- **The floor is ~89%.** Always predicting "normal" scores 89.9%. Accuracy is therefore
  not a useful metric here; **macro-F1** is the project's single number.
- **Class S is 1.79% of train but 3.70% of test.** A real train/test distribution shift,
  inherent to the benchmark rather than manufactured.

Full documentation, including per-patient class distributions and the verification
plots, is in [`reports/data_card.md`](reports/data_card.md).

---

## Reproducing it

No data files are committed — the dataset rebuilds itself from code:

```bash
git clone <this-repo> && cd ECG
pip install -e .
python -m ecg.build_dataset
```

That downloads the 48 records (~90 MB, a few minutes) and writes
`data/build/*.npy`. It is gated at five points — the AAMI class mapping must reproduce
the published beat counts exactly, the splits must be provably patient-disjoint, and the
extracted waveforms must be R-peak aligned — so a silent failure is caught rather than
trained on.

---

## Layout

```
src/ecg/            algorithms — every variant is an argument, not a copy-pasted file
  build_dataset.py    the Phase 0 pipeline, gated end to end
  config.py           one dataclass holding every knob
  data.py             loaders, feature selection, mini-batching
  init.py             zeros | random | xavier | he
  activations.py      relu | tanh | sigmoid | softmax, and derivatives
  losses.py           cross-entropy, L2 penalty, class weighting
  model.py            forward / backward, dropout, batch norm
  optimizers.py       gd | momentum | rmsprop | adam, LR schedules
  metrics.py          confusion matrix, per-class P/R/F1, macro-F1
  gradcheck.py        numerical verification of backprop
  train.py            train(config) -> one row in results/results.csv
  plots.py            shared figures

notebooks/          experiments — configure, call, plot, interpret
reports/            written findings
results/            results.csv and figures
```

**The rule:** no algorithm logic in a notebook. If a cell contains a loop over layers or
a derivative, it belongs in a module. Notebooks call `train(config)` and interpret what
comes back — which also means every ablation is a loop rather than a copied script:

```python
from dataclasses import replace
for opt in ["gd", "momentum", "rmsprop", "adam"]:
    for seed in [1, 2, 3]:
        train(replace(cfg, optimizer=opt, seed=seed))
```

---

## Plan

Phased, with each phase gated on a verifiable result. See [`PLAN.md`](PLAN.md).

| phase | | status |
|---|---|---|
| 0 | Build the patient-disjoint dataset | ✅ complete |
| 1 | Repository setup | in progress |
| 2 | C1 mechanics — logistic regression → shallow → deep, with gradient checking | |
| 3 | Softmax over 4 classes | |
| 4 | C2 ablation lab — initialisation, regularisation, optimisers, class imbalance, features | |
| 5 | C3 strategy memo — metric choice, bias/variance, error analysis, distribution shift | |
| 6 | PyTorch port, verified against the NumPy implementation step by step | |
| 7 | Additional PhysioNet databases (optional) | deferred |

The test split is read once, in Phase 5. Every decision before that is made on dev.
