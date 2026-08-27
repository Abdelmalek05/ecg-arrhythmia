# this script trains the small CNN and compares it with the flat network
#
# why we are doing this: phase 5 found that the answer for class S is in the SHAPE
# of the beat, and that a flat network could not read it. that was an explanation,
# not a measurement. a CNN slides one filter along the beat, so it can learn a shape
# once and find it anywhere. if the explanation is right, the CNN should do much
# better on class S.
#
# it takes about 20 minutes on one cpu thread.

import json
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

from ecg.data import load_split, fit_standardizer, apply_standardizer
from ecg.metrics import macro_f1, per_class
from ecg.torch_model import SmallCNN, MLP, train_torch

SEEDS = [1, 2, 3]


def main():
    Xtr, ytr, _ = load_split("train", "waveform+rr", "all", columns=True)
    Xdv, ydv, _ = load_split("dev", "waveform+rr", "all", columns=True)
    mu, sd = fit_standardizer(Xtr)
    Xtr_s = apply_standardizer(Xtr, mu, sd)
    Xdv_s = apply_standardizer(Xdv, mu, sd)

    print("%-22s %5s %10s %8s %8s %8s %9s"
          % ("model", "seed", "dev mF1", "S-F1", "V-F1", "F-F1", "time"), flush=True)

    out = {}
    for label, maker in [("MLP (flat, 254)", lambda: MLP(254, (16,), 4)),
                         ("CNN (shape + timing)", lambda: SmallCNN(n_rr=4))]:
        f1s, ss, vs, fs, ts = [], [], [], [], []
        preds = None
        for seed in SEEDS:
            torch.manual_seed(seed)
            r = train_torch(maker(), Xtr_s, ytr, Xdv_s, ydv, epochs=30,
                            batch_size=64, learning_rate=0.001,
                            optimizer="adam", seed=seed)
            pc = per_class(ydv.ravel(), r["pred_dev"], 4)["f1"]
            f = macro_f1(ydv.ravel(), r["pred_dev"], 4)
            f1s.append(f); ss.append(pc[1]); vs.append(pc[2]); fs.append(pc[3])
            ts.append(r["wall_clock"])
            if seed == SEEDS[0]:
                preds = r["pred_dev"]
            print("%-22s %5d %10.4f %8.4f %8.4f %8.4f %8.0fs"
                  % (label, seed, f, pc[1], pc[2], pc[3], r["wall_clock"]), flush=True)
        out[label] = {"mF1": float(np.mean(f1s)), "std": float(np.std(f1s)),
                      "S": float(np.mean(ss)), "V": float(np.mean(vs)),
                      "F": float(np.mean(fs)), "time": float(np.mean(ts))}
        np.save("results/_pred_" + label.split()[0].lower() + ".npy", preds)
        print(flush=True)

    print("%-22s %14s %9s %9s %9s" % ("", "dev mF1", "S-F1", "V-F1", "F-F1"), flush=True)
    for k, d in out.items():
        print("%-22s %9.4f +-%.3f %9.4f %9.4f %9.4f"
              % (k, d["mF1"], d["std"], d["S"], d["V"], d["F"]), flush=True)

    print("\nreference (dev macro-F1, from the numpy phases):", flush=True)
    print("   timing only, 4 features    0.5572   <- selected in phase 5", flush=True)
    print("   254 features, flat         0.4414", flush=True)
    print("   floor                      0.2366", flush=True)

    json.dump(out, open("results/phase6_cnn.json", "w"), indent=1)
    print("\nsaved results/phase6_cnn.json", flush=True)


if __name__ == "__main__":
    main()
