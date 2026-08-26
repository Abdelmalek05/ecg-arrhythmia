# this file builds the dataset from the raw physionet records
# run it with: python -m ecg.build_dataset
# it downloads the 48 records, cuts one window around every heartbeat,
# adds 4 timing features, and saves splits that never share a patient

from __future__ import annotations

import collections
import concurrent.futures as cf
import itertools
import json
import time
import urllib.request

import numpy as np
import wfdb

from .paths import RAW, BUILD, ensure_dirs

PHYSIONET = "https://physionet.org/files/mitdb/1.0.0/"

FS = 360                    # native sampling rate; no resampling
PRE, POST = 90, 160         # window [r-90, r+160) -> 250 samples, -0.25s .. +0.44s
WIN = PRE + POST
LEAD = "MLII"

# AAMI mapping. Anything not listed is a non-beat annotation and is discarded.
AAMI: dict[str, str] = {}
for _s in "NLRej":
    AAMI[_s] = "N"
for _s in ("A", "a", "J", "S"):
    AAMI[_s] = "S"
for _s in ("V", "E"):
    AAMI[_s] = "V"
AAMI["F"] = "F"
for _s in ("/", "f", "Q"):
    AAMI[_s] = "Q"

ALL_CLASSES = ["N", "S", "V", "F", "Q"]
CLASSES = ["N", "S", "V", "F"]          # Q is dropped: ~15 beats outside paced records

PACED = {"102", "104", "107", "217"}    # pacemaker patients, excluded by convention

# de Chazal inter-patient split: 22 records to train on, 22 different ones to test on.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]

N_DEV_RECORDS = 5
MIN_DEV_PER_CLASS = 30

# Expected AAMI totals across all 48 records. The Step 2 gate.
EXPECTED_TOTALS = {"N": 90631, "S": 2781, "V": 7236, "F": 803, "Q": 8043}


# --------------------------------------------------------------------------- step 1
def download(workers: int = 12) -> None:
    """Fetch every record's header, signal and annotation file. Skips what exists."""
    ensure_dirs()
    recs = wfdb.get_record_list("mitdb")
    jobs = [f"{r}{ext}" for r in recs for ext in (".hea", ".dat", ".atr")]

    def get(fn: str):
        out = RAW / fn
        if out.exists() and out.stat().st_size > 0:
            return 0
        for attempt in range(3):
            try:
                with urllib.request.urlopen(PHYSIONET + fn, timeout=60) as r:
                    data = r.read()
                out.write_bytes(data)
                return len(data)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
        return 0

    t = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        total = sum(ex.map(get, jobs))
    print(f"[1/5] download: {total / 1e6:.1f} MB fetched in {time.time() - t:.0f}s "
          f"({len(jobs)} files, cached ones skipped)")


# --------------------------------------------------------------------------- step 2
def records() -> list[str]:
    return sorted(p.stem for p in RAW.glob("*.hea"))


def annotation_index(rec: str):
    """(r_peak_samples, aami_labels) for one record, non-beat symbols dropped."""
    ann = wfdb.rdann(str(RAW / rec), "atr")
    keep = [(s, sym) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI]
    if not keep:
        return np.empty(0, dtype=int), []
    return (np.array([s for s, _ in keep], dtype=int),
            [AAMI[sym] for _, sym in keep])


def check_annotation_totals() -> collections.Counter:
    """Gate: the AAMI mapping must reproduce the published beat counts exactly."""
    tot = collections.Counter()
    for rec in records():
        _, labels = annotation_index(rec)
        tot.update(labels)
    ok = all(tot[k] == v for k, v in EXPECTED_TOTALS.items())
    print(f"[2/5] annotations: " + "  ".join(f"{k}:{tot[k]}" for k in ALL_CLASSES)
          + f"  total {sum(tot.values())}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"AAMI mapping wrong: got {dict(tot)}, expected {EXPECTED_TOTALS}")
    return tot


# --------------------------------------------------------------------------- step 3
def rr_features(samples: np.ndarray) -> np.ndarray:
    """Four R-R interval features per beat, normalised by the record's mean R-R.

    Class S beats are morphologically near-identical to N; what defines them is
    arriving early. Without these a shape-only model structurally cannot find them.

    Columns: pre_rr, post_rr, local_rr (mean of previous 10), pre_rr / local_rr.
    """
    n = len(samples)
    if n == 0:
        return np.empty((0, 4), dtype=np.float32)
    rr = np.diff(samples).astype(np.float64)
    pre = np.empty(n)
    post = np.empty(n)
    pre[1:] = rr
    pre[0] = rr[0] if n > 1 else 0.0
    post[:-1] = rr
    post[-1] = rr[-1] if n > 1 else 0.0

    local = np.empty(n)
    for i in range(n):
        lo = max(0, i - 10)
        local[i] = pre[lo:i + 1].mean() if i > lo else pre[i]

    mean_rr = pre.mean() if pre.mean() > 0 else 1.0
    ratio = pre / np.maximum(local, 1e-6)
    return np.column_stack([pre / mean_rr, post / mean_rr,
                            local / mean_rr, ratio]).astype(np.float32)


def extract_record(rec: str):
    """(beats[n,250], rr[n,4], labels[n], n_skipped) for one record, or None."""
    r = wfdb.rdrecord(str(RAW / rec))
    if LEAD not in r.sig_name:
        return None                      # 102 and 104 have no MLII (both paced anyway)
    # Select the lead BY NAME: record 114 has MLII on channel 1, not 0.
    sig = r.p_signal[:, r.sig_name.index(LEAD)].astype(np.float64)
    sig = sig / (sig.std() + 1e-8)       # per-record amplitude scaling

    samples, labels = annotation_index(rec)
    rr = rr_features(samples)

    beats, keep_idx, skipped = [], [], 0
    for i, s in enumerate(samples):
        lo, hi = s - PRE, s + POST
        if lo < 0 or hi > len(sig):      # window runs off the end of the signal
            skipped += 1
            continue
        w = sig[lo:hi]
        beats.append(w - np.median(w))   # per-beat baseline removal
        keep_idx.append(i)

    if not beats:
        return None
    return (np.asarray(beats, dtype=np.float32),
            rr[keep_idx],
            np.array([labels[i] for i in keep_idx]),
            skipped)


def extract_all():
    X, R, Y, REC = [], [], [], []
    skipped_total, no_lead = 0, []
    for rec in records():
        out = extract_record(rec)
        if out is None:
            no_lead.append(rec)
            continue
        beats, rr, labels, skipped = out
        X.append(beats)
        R.append(rr)
        Y.append(labels)
        REC.append(np.full(len(labels), rec))
        skipped_total += skipped

    X = np.concatenate(X)
    R = np.concatenate(R)
    Y = np.concatenate(Y)
    REC = np.concatenate(REC)

    assert X.shape[1] == WIN and R.shape[1] == 4
    assert not np.isnan(X).any() and not np.isnan(R).any()
    print(f"[3/5] extraction: {X.shape[0]} beats, {X.shape[1]} samples + {R.shape[1]} rr "
          f"| {skipped_total} edge beats skipped | no MLII: {no_lead} -> PASS")
    return X, R, Y, REC


# --------------------------------------------------------------------------- step 4
def choose_dev_records(per_rec: dict[str, collections.Counter]) -> list[str]:
    """Hold out whole records from DS1 for dev.

    Constraint: every class needs >= MIN_DEV_PER_CLASS beats in dev. F is rare and
    concentrated (record 208 alone holds ~90% of DS1's F beats), so a naive pick
    starves dev of it entirely. Among valid picks, prefer ~20% of DS1 and class
    proportions close to DS1's.
    """
    ds1 = collections.Counter()
    for r in DS1:
        ds1.update(per_rec[r])
    n_ds1 = sum(ds1.values())

    best, best_score = None, None
    for combo in itertools.combinations(DS1, N_DEV_RECORDS):
        c = collections.Counter()
        for r in combo:
            c.update(per_rec[r])
        if min(c[k] for k in CLASSES) < MIN_DEV_PER_CLASS:
            continue
        n_dev = sum(c.values())
        drift = sum(abs(c[k] / n_dev - ds1[k] / n_ds1) for k in CLASSES)
        score = abs(n_dev / n_ds1 - 0.20) * 3 + drift
        if best_score is None or score < best_score:
            best, best_score = combo, score

    if best is None:
        raise RuntimeError("no dev split satisfies the per-class minimum")
    return list(best)


def split_and_save(X, R, Y, REC) -> dict:
    keep = Y != "Q"
    X, R, Y, REC = X[keep], R[keep], Y[keep], REC[keep]

    per_rec = {r: collections.Counter(Y[REC == r]) for r in np.unique(REC)}
    dev = choose_dev_records(per_rec)
    train = [r for r in DS1 if r not in dev]
    test = list(DS2)

    # The single most important assertion in the project.
    s_tr, s_dv, s_te = set(train), set(dev), set(test)
    assert not (s_tr & s_dv) and not (s_tr & s_te) and not (s_dv & s_te), \
        "SPLIT LEAK: a patient appears in more than one split"

    cidx = {c: i for i, c in enumerate(CLASSES)}
    counts = {}
    for name, recs in (("train", train), ("dev", dev), ("test", test)):
        m = np.isin(REC, recs)
        y = np.array([cidx[c] for c in Y[m]], dtype=np.int64)
        np.save(BUILD / f"{name}_X.npy", X[m])
        np.save(BUILD / f"{name}_rr.npy", R[m])
        np.save(BUILD / f"{name}_y.npy", y)
        np.save(BUILD / f"{name}_rec.npy", REC[m])
        counts[name] = collections.Counter(y)

    np.save(BUILD / "classes.npy", np.array(CLASSES))
    splits = {"train": train, "dev": dev, "test": test, "classes": CLASSES}
    (BUILD / "splits.json").write_text(json.dumps(splits, indent=2))

    print(f"[4/5] splits: patient-disjoint PASS "
          f"({len(train)} train / {len(dev)} dev / {len(test)} test records)")
    print(f"      dev records: {dev}")
    for name in ("train", "dev", "test"):
        c = counts[name]
        n = sum(c.values())
        line = "  ".join(f"{k}:{c[cidx[k]]}" for k in CLASSES)
        print(f"      {name:5} {n:>6} beats   {line}   baseline {100 * max(c.values()) / n:.1f}%")
    return splits


# --------------------------------------------------------------------------- step 5
def verify() -> None:
    """Reload everything from disk and re-check the invariants."""
    splits = json.loads((BUILD / "splits.json").read_text())
    total = 0
    for name in ("train", "dev", "test"):
        X = np.load(BUILD / f"{name}_X.npy")
        R = np.load(BUILD / f"{name}_rr.npy")
        y = np.load(BUILD / f"{name}_y.npy")
        rec = np.load(BUILD / f"{name}_rec.npy")
        assert X.shape[0] == R.shape[0] == y.shape[0] == rec.shape[0]
        assert X.shape[1] == WIN and R.shape[1] == 4
        assert not np.isnan(X).any() and not np.isnan(R).any()
        assert set(np.unique(rec)) == set(splits[name])
        total += len(y)
    a, b, c = (set(splits[k]) for k in ("train", "dev", "test"))
    assert not (a & b) and not (a & c) and not (b & c)
    print(f"[5/5] verify: {total} beats, all invariants hold -> PASS")


def main() -> None:
    ensure_dirs()
    download()
    check_annotation_totals()
    X, R, Y, REC = extract_all()
    split_and_save(X, R, Y, REC)
    verify()
    print(f"\ndone -> {BUILD}")


if __name__ == "__main__":
    main()
