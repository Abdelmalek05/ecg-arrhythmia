# ECG Arrhythmia Classification — from scratch in NumPy

Classifying individual heartbeats into four arrhythmia types, with the neural network
written from first principles: forward and backward propagation, initialisation
schemes, optimisers and regularisation implemented directly in NumPy rather than called
from a framework.

A rehearsal of Courses 1–3 of the Deep Learning Specialization, on real clinical data.
**216 logged experiments.** The most useful result is a negative one — the model
selection procedure failed, and the notebooks work out exactly why.

---

## Result

Held-out test set: **49,685 beats from 22 patients never seen in training.** Read once.

| | accuracy | macro-F1 |
|---|---|---|
| always predict "normal" (the floor) | 0.8904 | 0.2355 |
| **reported model** | **0.9205** | **0.4193** |

Per class:

```
 class     prec   recall       f1  support        confusion (rows = true)
     N    0.968    0.972    0.970    44240              N     S     V     F
     S    0.148    0.033    0.053     1837        N  43021   247   914    58
     V    0.557    0.825    0.665     3220        S    579    60  1197     1
     F    0.000    0.000    0.000      388        V    463    99  2655     3
                                                  F    385     0     3     0
```

Accuracy moves 3 points above the floor and looks like a marginal model. Macro-F1
nearly doubles. With 89% of beats normal, **accuracy is close to useless here**, which
is why macro-F1 is the project's single number.

Per patient, macro-F1 ranges **0.263 to 1.000** (median 0.528). A pooled number hides a
4× spread across hearts.

---

## The problem

Each example is one heartbeat, taken from a 30-minute ECG recording and labelled by a
cardiologist:

| class | what it is | what gives it away |
|---|---|---|
| **N** | normal | narrow, sharp QRS; arrives on schedule |
| **S** | supraventricular ectopic | **shape is nearly identical to N** — only the early arrival betrays it |
| **V** | ventricular ectopic | wide, slurred QRS; the signal never used the fast conduction path |
| **F** | fusion | a normal and a ventricular beat collided; morphology sits between the two |

Class S is the hard one, and the reason is structural: its waveform looks like a normal
beat, so a model given only the signal shape cannot find it — no amount of training
fixes a feature set that does not contain the answer. Four R-R interval features
(how early the beat arrived relative to that patient's own rhythm) are extracted
alongside the waveform for exactly this reason.

**Input:** 250 waveform samples (−0.25 s to +0.44 s around the R-peak, 360 Hz, lead
MLII) + 4 R-R timing features = **254 features**.

---

## The dataset

Built from the raw [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
— 48 half-hour recordings, ~109,000 expert-annotated beats.

**The splits share no patients.** Each recording contributes ~2,300 near-identical beats
from one person, so splitting beats at random puts ~1,800 near-twins of every test beat
into training. A model then scores ~99% by recognising individuals rather than
arrhythmias, and train and dev error both collapse to ~1%, leaving nothing to diagnose.
Splitting by *patient* keeps the evaluation honest.

| split | patients | N | S | V | F | total | majority baseline |
|---|---|---|---|---|---|---|---|
| train | 17 | 36,418 | 726 | 2,970 | 382 | 40,496 | 89.9% |
| dev | 5 | 9,430 | 218 | 818 | 32 | 10,498 | 89.8% |
| test | 22 | 44,240 | 1,837 | 3,220 | 388 | 49,685 | 89.0% |

Class S is 1.79% of train but 3.70% of test — a real distribution shift, inherent to
the benchmark rather than manufactured.

Full documentation in [`reports/data_card.md`](reports/data_card.md).

---

## What was learned

### 1. The model chosen on dev was the worst of three on test

| model | dev | **test** | |
|---|---|---|---|
| timing only (4 features) | 0.5572 | **0.4193** | ← selected, and reported |
| shape + timing (254) | 0.3861 | **0.4794** | |
| shape only (250) | 0.3232 | 0.4040 | |

The better model was rejected by the selection procedure. It was **not** swapped in
afterwards — choosing on the test set would have destroyed the only unbiased estimate
in the project. `0.4193` stands.

### 2. The cause was patients per class, not beats per class

The entire drop is one class: S falls from 0.628 on dev to 0.053 on test. N and V
barely move. Since macro-F1 ignores class sizes, the known S distribution shift
explains none of it.

The timing features detect **prematurity** — but *both* S and V beats are premature.
Timing says "this beat came early", not which kind of early beat it is. Separability of
S from V using the timing feature alone:

| split | AUC | S carrier patients |
|---|---|---|
| dev | 0.668 | **2** |
| test | 0.514 | 16 |

Dev's 218 S beats came from effectively two patients, whose S beats happen to arrive
nearly on time (median ratio 0.992). Test's dominant S patient sits at 0.738 —
indistinguishable from V's 0.726. The model learned a rule true of two hearts, and dev
*was* those two hearts.

> **A dev set needs enough patients per class, not enough beats per class.**

Worked through in [`notebooks/08_error_analysis.ipynb`](notebooks/08_error_analysis.ipynb).

### 3. The information was there and the feature choice discarded it

Of the 1,197 S beats wrongly called V, **94.6% are closer in shape to the average S beat
than to the average V beat.** The waveform would have separated them.

### 4. Two models at opposite ends of bias/variance

| model | train | dev | train − dev | |
|---|---|---|---|---|
| timing only (4) | 0.5634 | 0.5572 | **0.006** | high bias — cannot fit its own training data |
| shape + timing (254) | 0.9567 | 0.3861 | **0.571** | high variance — memorises 17 patients |

174 ablation runs of regularisation and optimisation were spent on the **high-bias**
model, where none of those levers apply. That is visible in the results: no C2 technique
moved macro-F1 by more than ~0.05.

### 5. Normalisation has to happen before the mixing

The 250 waveform features carry ~1000× more total variance than the 4 timing features,
so a hidden unit barely hears the timing block. Standardising the **inputs** raised S-F1
from 0.017 to 0.133. Adam (0.019) and batch norm (0.023) both failed — they act on
`W·A + b`, which is already a *mixture*; rescaling it cannot restore who contributed
what.

### 6. Two settings genuinely break the model

Zero initialisation returns **exactly** the floor (0.2366, std 0.0000) — symmetry never
breaks, so the network can only emit the majority class. Full-batch descent collapses to
0.2564, because 30 epochs × 1 batch is **30 updates** against ~19,000 at batch 64.

### 7. The NumPy implementation is correct

Verified against PyTorch with identical initial weights, batches and learning rate:

```
loss   numpy 2.374121473870872
       torch 2.374121473870872
worst difference over 10 descent steps:  4.163e-17
```

Float64 rounding error. Gradient checking proves backward matches forward; it cannot
catch both being wrong together. This can.

Also measured: **PyTorch is not faster at this scale** — 20.9 s NumPy vs 21.2 s torch
(float64), 18.8 s at float32. Framework overhead cancels the optimised kernels for a
254→16→4 network with no GPU.

### 8. Class F is not learnable from this data

F1 = 0.000 on dev and test across 192 four-class runs. 382 training beats, **372 of them
from a single patient.** A property of the dataset, not the model. Reported, not chased.

---

## Reproducing it

No data files are committed — the dataset rebuilds itself from code:

```bash
git clone <this-repo> && cd ECG
pip install -e .
python -m ecg.build_dataset
```

That downloads the 48 records (~90 MB) and writes `data/build/*.npy`. It is gated at
five points — the AAMI class mapping must reproduce the published beat counts exactly,
the splits must be provably patient-disjoint, and the extracted waveforms must be
R-peak aligned — so a silent failure is caught rather than trained on.

For the PyTorch parts: `pip install -e ".[torch]"`.

---

## Layout

```
src/ecg/            algorithms — every variant is an argument, not a copy-pasted file
  build_dataset.py    the data pipeline, gated end to end
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
  torch_model.py      the same network in PyTorch, plus a small 1D CNN
  plots.py            shared figures

notebooks/          the experiments and the written analysis
scripts/            reproducible sweeps and notebook generators
results/            results.csv (every run) and figures
reports/            data card
```

**The rule: no algorithm logic in a notebook.** If a cell contains a loop over layers or
a derivative, it belongs in a module. Notebooks call `train(config)` and interpret what
comes back — which also means an ablation is a loop, not a copied script:

```python
from dataclasses import replace
for opt in ["gd", "momentum", "rmsprop", "adam"]:
    for seed in [1, 2, 3]:
        train(replace(cfg, optimizer=opt, seed=seed))
```

Every run appends a row to `results/results.csv`, and every figure is rebuilt from that
file rather than from memory.

---

## Notebooks

| | |
|---|---|
| `02_logistic_regression` | one neuron from scratch, checked against scikit-learn |
| `03_shallow_nn` | one hidden layer; why three seeds are mandatory |
| `04_deep_nn_gradcheck` | L-layer network and numerical gradient checking |
| `05_softmax_multiclass` | four classes; the feature-scale problem |
| `06_ablations` | init, L2, dropout, batch size, optimisers, LR decay |
| `07_batchnorm` | batch norm, and why it cannot replace input scaling |
| `08_error_analysis` | the test set, and why dev chose the wrong model |
| `09_pytorch` | agreement with PyTorch to 1e-17, and a 1D CNN |

[`PLAN.md`](PLAN.md) records how the work was staged and what each step had to prove
before moving on.

---

## Data and citation

No data is stored in this repository. `python -m ecg.build_dataset` downloads the
records directly from PhysioNet at build time.

PhysioNet asks that both of the following be cited by anyone using the MIT-BIH
Arrhythmia Database:

> Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.*
> IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). PMID: 11446209.

> Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE,
> Moody GB, Peng C-K, Stanley HE. *PhysioBank, PhysioToolkit, and PhysioNet: Components
> of a New Research Resource for Complex Physiologic Signals.*
> Circulation 101(23):e215-e220 (2000).

Database: https://physionet.org/content/mitdb/1.0.0/

The inter-patient evaluation protocol (the DS1 / DS2 record split used here) is from:

> de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG
> morphology and heartbeat interval features.* IEEE Trans Biomed Eng 51(7):1196-1206
> (2004).

## Licence

Code is MIT licensed — see [LICENSE](LICENSE). The licence covers this repository's code
only, not the MIT-BIH data, which remains under PhysioNet's terms.

## Scope and limitations

This is a learning project, built to practise Courses 1–3 of the Deep Learning
Specialization. **It is not a medical device and its results are not a clinical claim.**

The limitations are real and were measured, not assumed:

- **One lead, one database, 44 usable patients** — 17 of them for training. Patient
  diversity, not beat count, is the binding constraint on everything here.
- **Class F cannot be learned from this data** (one patient carries it).
- **The reported 0.4193 is a genuinely held-out number, but it comes from a model chosen
  by a procedure now known to be flawed.** The honest reading is *this is what the
  selected model achieves*, not *this is the best achievable here*.
- **A fully-connected network on 250 raw samples is a weak representation of a
  waveform** — it has no notion that neighbouring samples are related. This is why the
  literature uses convolutional networks on this dataset, and it is the measured cost of
  the decision to exclude them.
