# this script runs the phase 4b experiments: batch norm, on and off
#
# the real question is not "does batch norm help". phase 4a showed that scaling
# the INPUT is what helped. batch norm scales INSIDE the network. so we ask:
# can batch norm do the job instead of scaling the input?
# that is why this is a grid of batch_norm x standardize x features.

import time
from dataclasses import replace

from ecg.config import Config
from ecg.train import train

SEEDS = [1, 2, 3]

BASE = Config(classes="all", output_activation="softmax",
              features="waveform+rr", standardize=False,
              hidden_dims=(16,), hidden_activation="tanh", init="he",
              optimizer="gd", learning_rate=0.1, batch_size=64, epochs=30,
              class_weights="none", phase="4b")

done = 0
start = time.time()

print("=== grid C: batch norm x standardise x features ===", flush=True)
for feats in ["waveform", "waveform+rr", "rr"]:
    for bn in [False, True]:
        for std in [False, True]:
            for seed in SEEDS:
                cfg = replace(BASE, features=feats, batch_norm=bn,
                              standardize=std, seed=seed, note="gridC bn-std-features")
                out = train(cfg, verbose=False)
                done = done + 1
                m = out["metrics"]
                print("[%2d/36] feat=%-12s bn=%-5s std=%-5s seed=%d  dev mF1 %.4f | %s  (%.0fs)"
                      % (done, feats, str(bn), str(std), seed,
                         m["dev_macro_f1"], m["dev_f1_per_class"], out["wall_clock"]),
                      flush=True)

print("\nDONE: %d runs in %.0f minutes" % (done, (time.time() - start) / 60.0), flush=True)
