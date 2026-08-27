# How this project was built

*C1–C3 rehearsal project — Andrew Ng Deep Learning Specialization*

This is the working plan the project was built to. It is kept as a record of how the
work was staged and, more importantly, **what each step had to prove before the next
one was allowed to start.** The results themselves are in [README.md](README.md) and in
the notebooks.

**Status: complete.** 216 runs logged. Final held-out result: **test macro-F1 0.4193**
against a floor of 0.2355. The test set was read once, in Phase 5, and no decision was
made after reading it.

## The idea behind the staging

Every phase ends on a **gate** — a number that has to match something known, not a
feeling that the step is done. A few examples of what that caught:

- the AAMI class mapping had to reproduce the published beat counts exactly (109,494)
- the from-scratch logistic regression had to match scikit-learn within 1%
- the L-layer network had to reproduce the standalone logistic regression **bit for bit**
- gradient checking had to stay under 1e-7 *and* had to fail on a deliberate 1% bug
- the mean beat waveforms had to look like plausible heartbeats before any model ran

The gates are the reason the leakage problem, the feature-scale problem and the
dev-selection failure were found rather than shipped.

---

## The dataset (built, verified)

MIT-BIH Arrhythmia Database from PhysioNet, rebuilt from raw records so the split can be **patient-disjoint**. Not the Kaggle CSVs — those pool all 48 recordings and shuffle, discarding patient identity, which inflates accuracy to ~99% and flattens every diagnostic this project depends on.

**One example = one heartbeat:** 250 waveform samples (−0.25 s to +0.44 s around the R-peak, 360 Hz, MLII lead) + 4 R-R timing features. **Input dimension 254.** Label ∈ {N, S, V, F}.

| split | patients | N | S | V | F | total | baseline |
|---|---|---|---|---|---|---|---|
| train | 17 | 36,418 | 726 | 2,970 | 382 | 40,496 | 89.9% |
| dev | 5 | 9,430 | 218 | 818 | 32 | 10,498 | 89.8% |
| test | 22 | 44,240 | 1,837 | 3,220 | 388 | 49,685 | 89.0% |

Full documentation in `reports/data_card.md`.

### Facts established before any training

These are measured, not assumed. They set expectations for every later phase.

1. **The floor is ~89%.** A model that always predicts N scores 89.9% on train. Accuracy is therefore useless as a metric; **macro-F1** is the single number.
2. **V is easy, S is hard.** V has a distinctive wide QRS — findable from waveform alone. S is morphologically near-identical to N; only its *early arrival* distinguishes it.
3. **The R-R features already separate S.** Median `pre_rr/local_rr`: N 1.002, S 0.784, V 0.720, F 0.951. The hypothesis behind the Phase 4 feature ablation is pre-validated.
4. **The S distribution shifts between train and test** — 1.79% → 3.70%. A genuine, free distribution mismatch for the C3 phase.
5. **Rare classes concentrate in single patients.** Record 208 holds 372 of 382 training F beats. In test, patient 232 holds 75% of all S beats and patient 213 holds 93% of all F beats. Expect poor generalization on F, and treat it as a data limitation, not a bug.
6. **Dev-F is unreliable — 32 beats.** One misclassification moves F recall ~3 points. Trust the test set for F; treat dev-F as directional.
7. **The real unit of data is the patient, not the beat.** 40,496 training rows are 17 hearts. Patient diversity, not row count, is the binding constraint.

---

## Code architecture

**Two kinds of file, and a hard rule between them:**

- **`src/*.py` — modules.** All algorithms. Every module takes its variant **as an argument**, never as a copy-pasted file.
- **`notebooks/*.ipynb` — experiments.** Configure, call, plot, interpret. Narrative and results.

> **The rule: no algorithm logic in a notebook.** If you are writing a `for` loop over layers, or a derivative, in a cell — it belongs in a module. Notebooks call `train(config)` and plot what comes back.

This is not tidiness for its own sake. The Phase 4 ablation table has 9 axes; the module signatures below are derived directly from it, so each axis is one keyword argument and each ablation is a loop.

### Module map — each row is a Phase 4 ablation axis

| Module | Key argument | Variants | Ablation axis |
|---|---|---|---|
| `init.py` | `method=` | `zeros` `random` `xavier` `he` | 1 |
| `model.py` | `l2=` | `0, 0.01, 0.1, 1.0` | 2 |
| `model.py` | `keep_prob=` | `1.0, 0.8, 0.5` | 3 |
| `train.py` | `batch_size=` | `full, 512, 64, 8` | 4 |
| `optimizers.py` | `name=` | `gd` `momentum` `rmsprop` `adam` | 5 |
| `train.py` | `lr_decay=` | `none` `inverse` `exponential` | 6 |
| `model.py` | `batch_norm=` | `True` `False` | 7 |
| `losses.py` | `class_weights=` | `none` `balanced` | 8 |
| `data.py` | `features=` | `waveform` `waveform+rr` | 9 |

Plus: `activations.py` (relu/tanh/sigmoid/softmax + derivatives), `metrics.py` (confusion matrix, per-class P/R/F1, macro-F1), `gradcheck.py`, `plots.py`, `config.py`.

### Signature style

Course-faithful: module-level functions operating on a `parameters` dict (`{"W1":…, "b1":…}`), exactly as in C1W4 — that dict *is* what this project is rehearsing. A `Config` dataclass and a single `train(config)` entry point wrap them for notebook use.

```python
# src/init.py
def initialize_parameters(layer_dims, method="he", seed=None):
    """layer_dims e.g. [254, 64, 32, 4];  method: zeros|random|xavier|he"""

# src/optimizers.py
def init_optimizer_state(parameters, name)
def update_parameters(parameters, grads, state, name, lr, **hp)

# src/losses.py
def compute_loss(AL, Y, name, class_weights=None, parameters=None, l2=0.0)
def loss_backward(AL, Y, name, class_weights=None)
```

Scale factors matching the course: `random` → ×0.01, `xavier` → √(1/n_prev), `he` → √(2/n_prev).

### The payoff

`src/config.py` holds one dataclass with every knob; `train(config)` runs it and appends a row to `results.csv`. An entire ablation becomes:

```python
from dataclasses import replace
for opt in ["gd", "momentum", "rmsprop", "adam"]:
    for seed in [1, 2, 3]:
        train(replace(cfg, optimizer=opt, seed=seed))
```

Twelve runs, four lines, everything logged. That is why the modules are parameterised.

### Notebooks

```
notebooks/
  01_data_exploration.ipynb     Phase 0 review — the data card, made interactive
  02_logistic_regression.ipynb  Phase 2
  03_shallow_nn.ipynb           Phase 2
  04_deep_nn_gradcheck.ipynb    Phase 2
  05_softmax_multiclass.ipynb   Phase 3
  06_ablations.ipynb            Phase 4a
  07_batchnorm.ipynb            Phase 4b
  08_error_analysis.ipynb       Phase 5
  09_pytorch.ipynb              Phase 6
```

Two practical requirements:

1. **Autoreload in every notebook's first cell** — otherwise editing a module does nothing until you restart the kernel:
   ```python
   %load_ext autoreload
   %autoreload 2
   ```
2. **Make `src` importable.** Use a `pyproject.toml` + `pip install -e .` so notebooks just `from ecg.init import initialize_parameters`, rather than `sys.path` hacks in every file.

**Committing notebooks:** discipline is *Restart & Run All before committing*, so what is on GitHub matches what the code does. Outputs stay committed — being able to read results on GitHub is most of the value. If diffs become painful, add `nbstripout` and let `results/results.csv` + `results/figures/` be the record instead.

---

## Conventions

- **One `SEED` constant**, threaded through init, shuffling, subsampling.
- **`results/results.csv`**, appended by every run: `run_id, phase, features, arch, init, optimizer, lr, batch_size, l2, dropout, epochs, seed, train_loss, train_acc, dev_acc, macro_f1, per_class_f1, wall_clock`. Start at run #1 — Phase 4 is unreadable without it.
- **Three seeds per ablation**, report mean ± std. A 0.3% gap on one seed is noise.
- **Numerical stability from the start:** log-sum-exp inside the softmax *loss*; clip before `log` in binary cross-entropy.
- **`rec` is never a model input.** It is bookkeeping. Feeding it in would let the network memorise patients.
- Test set is read **once, at the end.** Every decision is made on dev.

---

## Phase 0 — Build the dataset ✅ COMPLETE

Five gates, all passed:

| gate | result |
|---|---|
| 48 records downloaded | ✅ 144 files, 90 MB |
| AAMI mapping exact | ✅ N 90,631 · S 2,781 · V 7,236 · F 803 · Q 8,043 = 109,494 |
| Extraction clean | ✅ (105039, 250) + (105039, 4), no NaNs, 39 edge beats skipped |
| Splits patient-disjoint | ✅ asserted, 17 / 5 / 22 patients |
| Waveforms plausible | ✅ R-peak aligned within 3–6 ms; V wide, F between N and V, S ≈ N |

Artifacts: `src/build_dataset.py`, `data/build/*.npy`, `data/build/splits.json`, `reports/data_card.md`, `results/figures/`.

---

## Phase 1 — Repository setup ✅ COMPLETE

Do this before writing model code, so every experiment is tracked from the first commit.

1. **`git init`** on the project root.
2. **`.gitignore`** — critically, the data must not be committed. 293 MB of regenerable files, and GitHub rejects files over 100 MB:
   ```
   data/physionet/
   data/build/
   __pycache__/
   *.pyc
   .venv/
   ```
   `data/build/splits.json` is the exception — **force-add it**. It is small and it records the single most important decision in the project.
3. **`pyproject.toml` + `pip install -e .`** — makes `src/` an importable package (`ecg`), so notebooks use `from ecg.init import initialize_parameters` instead of `sys.path` hacks. Dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `wfdb`, `jupyter` (add `torch` at Phase 6).
4. **Scaffold the module skeletons and notebook folder** — `src/ecg/{config,data,init,activations,losses,model,optimizers,metrics,gradcheck,plots,train}.py` with signatures and docstrings in place (implementations come in Phase 2), plus `notebooks/` with the autoreload cell as a template.
5. **`README.md`** — what the project is, the data card summary table, and a "reproduce from scratch" section: clone → `pip install -e .` → `python -m ecg.build_dataset`. The repo must rebuild the dataset without any file over 1 MB being committed.
6. **First commit** — the Phase 0 work: `src/`, `reports/data_card.md`, `results/figures/`, `PLAN.md`.
7. **GitHub repo** — create via `gh repo create`, push. *Decide public vs. private at this point.*
8. **Commit rhythm from here:** work directly on `main` — one commit per meaningful step. No feature branches: with a single contributor, no review and no CI, they add ceremony without isolating anything. Instead **tag at each phase boundary** once its gate passes:
   ```bash
   git tag phase-0-data && git push origin phase-0-data
   ```
   That gives legible landmarks (`git diff phase-0-data..HEAD`) and a release point on GitHub, with no merges. Branch only for a genuine spike you might throw away.
   Notebooks: *Restart & Run All* before committing.

**Deliverable:** a clean repo where someone can clone, `pip install -e .`, and reproduce the whole dataset from code alone.
**Done when:** a fresh clone rebuilds `data/build/` and the numbers match the data card.

---

## Phase 2 — C1 mechanics: binary, N vs. V ✅ COMPLETE

Filter to two classes. The goal is **working backprop**, not a good score.

1. **Logistic regression from scratch (NumPy)** — single neuron, sigmoid, BCE, vectorised forward/backward, gradient descent.
   - *Correctness gate:* must agree with `sklearn.linear_model.LogisticRegression` on the same split to within ~1%.
2. **One-hidden-layer NN** — sweep hidden size {4, 16, 64}, tanh vs. ReLU.
3. **L-layer deep NN** — `[254, 64, 32, 1]`, forward and backward as a loop over layers.
4. **Gradient check — immediately**, the moment the L-layer backward pass exists, on a tiny net and a handful of examples. *Before* training anything or reading any curve.

**Ties to:** C1W2 → C1W4.
**Deliverable:** the phase notebook — loss curves for all three, table of accuracy / precision / recall / F1 against the trivial baseline.
**Done when:** gradient check < 1e-7 relative error and scratch LR matches sklearn.

---

## Phase 3 — Multi-class: softmax over 4 classes ✅ COMPLETE

Reuse the Phase 2 network. Swap exactly three things:
- sigmoid(1) → **softmax(4)**
- BCE → **categorical cross-entropy** (log-sum-exp form)
- labels → **one-hot**

**Re-run the gradient check** — new loss, new backward path.

Run twice: **waveform only (250 inputs)** and **waveform + R-R (254 inputs)**.

**Ties to:** C2W3 (softmax).
**Deliverable:** the phase notebook — 4×4 confusion matrix, per-class P/R/F1, macro-F1, against the ~89% baseline, for both feature sets.
**Done when:** you can state in one sentence what the R-R features did to class S recall.

---

## Phase 4 — C2 ablation lab ✅ COMPLETE (split into 4a + 4b)

No new models. The Phase 3 network is the testbed; one variable at a time, 3 seeds each, every run logged.

| # | Axis | Variants | Ties to |
|---|---|---|---|
| 1 | Initialization | zeros / random×0.01 / Xavier / He | C2W1 |
| 2 | L2 regularization | λ ∈ {0, 0.01, 0.1, 1.0} | C2W1 |
| 3 | Dropout | keep_prob ∈ {1.0, 0.8, 0.5} | C2W1 |
| 4 | Mini-batch size | {full, 512, 64, 8} | C2W2 |
| 5 | Optimizer | GD / Momentum / RMSprop / Adam | C2W2 |
| 6 | LR decay | constant vs. decayed | C2W2 |
| 7 | Batch norm | with vs. without | C2W3 |
| 8 | **Class imbalance** | none / weighted loss / oversampling | — |
| 9 | **Features** | 250 waveform vs. 254 with R-R | — |

**Start with 8 and 9** — they have real hypotheses attached and are the most interesting.

**If dev results bounce across seeds**, switch to **patient-wise cross-validation over the 22 DS1 patients** instead of the fixed 17/5 split. That buys more effective training *and* validation patients at k× compute — minutes on this data.

**Deliverable:** the phase notebook — one section per axis: plot, numbers, 2–3 sentences on *why*. "No effect" is a valid finding if the mechanism is explained.

---

## What Phases 3-4 established (read before Phase 5)

1. **The input representation is the binding constraint.** 4 timing features reach
   dev macro-F1 0.5608; the same network on 254 features reaches 0.4414 and on 250
   waveform samples 0.4004. Adding the waveform to the timing features *hurts*.
2. **No C2 technique moved the result more than ~0.05.** Initialisation, L2, dropout,
   batch size, four optimisers, LR decay, batch norm, class weights — all small.
   Changing the input moved it 0.24.
3. **Normalisation must precede mixing.** Input scaling raised S-F1 from 0.017 to
   0.133; batch norm (which acts after W*A+b) reached only 0.023. Adam failed for the
   same reason: both act downstream of the mixing.
4. **Class F is unlearnable here.** F1 = 0.000 in the best model, across 189 runs.
   382 training beats, 372 from one patient.
5. **Two settings genuinely break the model:** zeros init (exactly the floor) and
   full-batch descent (30 updates total).

## What Phase 5 established

6. **Dev-based model selection failed.** The model chosen on dev (timing only,
   test 0.4193) is the *worst* of three on test; the 254-feature model reaches 0.4794.
   The choice was kept, because switching after seeing test would destroy the estimate.
7. **The cause is patients per class, not beats per class.** Dev's 218 S beats came
   from effectively two patients, whose S beats arrive nearly on time (ratio 0.992).
   Test's dominant S patient sits at 0.738, indistinguishable from V's 0.726.
   S-vs-V separability by timing: AUC 0.668 on dev, **0.514 on test**.
8. **We tuned the wrong model.** Timing-only is high bias (train 0.5634, dev 0.5572 —
   gap 0.006); the 254-feature model is high variance (gap 0.571). Phase 4's 174 runs
   of regularisation and optimisation were applied to the high-bias one, where they
   cannot help. **Fix the dev split before any further tuning.**

## What Phase 6 established

9. **The from-scratch NumPy code is correct.** It agrees with PyTorch to **4e-17** —
   float64 rounding error — on loss, gradients and weights across 10 descent steps.
   Stronger than gradient checking, which cannot catch forward and backward being
   wrong together.
10. **PyTorch is not faster at this scale.** 20.9s NumPy vs 21.2s torch float64
    (1.02x slower); 18.8s at float32. No GPU, and framework overhead dominates for a
    254-16-4 network.
11. **A CNN reads shape, but cannot be judged on class S here.** V-F1 rose 0.428 ->
    0.520 (the convolution works), S-F1 only 0.088 -> 0.112. And S cannot be measured
    on our dev set at all, per finding 7. **Inconclusive, and blocked by the dev split.**

## Phase 5 — C3 strategy memo ✅ COMPLETE

Mostly analysis, no new models.

1. **Single-number metric.** Justify macro-F1 at an ~89% majority. Show the arithmetic: ignoring S and F entirely still scores ~96%.
2. **Bias/variance.** Place every Phase 2–4 model in its regime; state the *next action* for each.
3. **Error analysis.** Pull 30 misclassified beats and **plot** them against the class-mean waveform of both true and predicted class — a 250-vector cannot be eyeballed as numbers. Categorise failure modes and count them.
4. **Distribution mismatch.** Class S is 1.79% of train, 3.70% of test. Quantify how much of the test drop that explains.
5. **Limitations.** Single lead, one database, 44 patients, F concentrated in one person per split, dev-F unreliable at 32 beats.

**Ties to:** C3W1–W2.
**Deliverable:** the phase notebook — prose, not code.

---

## Phase 6 — PyTorch port ✅ COMPLETE

1. **Agreement test first.** Same init, same data order, same LR and batch size — NumPy vs. PyTorch loss over the first 10 steps must match to ~1e-5. A far stronger check on the from-scratch code than comparing final accuracy.
2. Then train properly with `nn.Module` / `nn.CrossEntropyLoss` / `torch.optim.Adam`; compare wall-clock and macro-F1.

---

## Phase 7 — More data (optional, deferred by decision)

Three PhysioNet databases use the same annotation format and were scanned but **deliberately not used** — data quantity is not the current bottleneck, and adding them turns the work back into data engineering.

| database | records | rate | beats | S | V | F |
|---|---|---|---|---|---|---|
| **svdb** | 78 | 128 Hz | 184,582 | **12,198** | 9,943 | 23 |
| **incartdb** | 75 | 257 Hz | 175,874 | 1,960 | **20,013** | 219 |
| **ltdb** | 7 | 128 Hz | 668,735 | 1,500 | 64,095 | **2,908** |

Priority if revisited:

1. **svdb** — 4.4× more S beats than all of MIT-BIH, 78 new recordings, ~4 min download. Makes the Phase 4 R-R experiment far more convincing. The one worth doing.
2. **incartdb** — train on MIT-BIH, test on INCART. Different country, hospital, equipment, and lead set: a genuine cross-database distribution shift and the best C3W2 material available. ~45 min download.
3. **ltdb** — only if F becomes the focus. Adds F beats but just 7 patients, so it does not fix the "F is one person" problem.

**Three costs to plan for:** rates differ (360/128/257 Hz) so resampling returns; svdb's leads are unnamed (`ECG1`/`ECG2`) so lead matching is imperfect; **INCART's 75 records are not 75 people** — splitting by record there would recreate the exact leakage this project removed.

---

## Pacing

| # | Phase | Ties to | Effort | Status |
|---|---|---|---|---|
| 0 | Build the dataset | — | ½ day | ✅ done |
| 1 | Repository setup | — | 1 hour | ✅ done |
| 2 | C1 mechanics, N vs. V | C1W2–W4 | 2–3 days | ✅ done |
| 3 | 4-class softmax | C2W3 | 1 day | ✅ done |
| 4a | C2 ablation lab | C2W1–W2 | 2–3 days | ✅ done |
| 4b | Batch norm | C2W3 | 1 hour | ✅ done |
| 5 | C3 strategy memo | C3W1–W2 | 2 days | ✅ done |
| 6 | PyTorch port + CNN | — | 1 day | ✅ done |
| 7 | More data | — | optional | deferred |

---

## Risks

- **Class S is the real difficulty.** Expect poor recall without R-R features. That is the correct result, not a bug.
- **Class F will generalise badly.** 372 of 382 training F beats come from one patient. Report it; do not chase it.
- **Dev is only 5 patients**, and every Phase 4 decision is read off it. Watch for seed noise; escalate to patient-wise cross-validation if needed.
- **Never feed `rec` to the model.**
- **Never touch the test set** until Phase 5.
- **Numerical stability.** Log-sum-exp for softmax CE; clip before `log` in BCE — the BCE one bites in Phase 2, before softmax exists.
- **Commit before each phase's experiments**, so a bad refactor never costs logged results.
- **Skeletons raise `NotImplementedError` with the phase in the message** — that is the to-do list.
