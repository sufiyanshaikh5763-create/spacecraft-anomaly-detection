import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from evaluate import load_labels, flags_to_sequences, score_events, precision_recall
from errors import get_errors      # use whatever name evaluate_all.py imports from
from threshold import find_threshold, WARMUP
from prune import prune

labels = load_labels()
os.makedirs("errors", exist_ok=True)

# Cache raw errors once (the first run predicts every channel, like evaluate_all.py)
raw = {}
for chan in labels["chan_id"].unique():
    path = f"errors/{chan}.npz"
    if os.path.exists(path):
        z = np.load(path)
        raw[chan] = (z["error"], z["steps"])
    else:
        _, _, error, smoothed, steps = get_errors(chan)
        np.savez(path, error=error, smoothed=smoothed, steps=steps)
        raw[chan] = (error, steps)
        print("cached", chan, flush=True)


def evaluate(span, do_prune):
    rows, no_thr, pruned_all = [], 0, 0
    for _, r in labels.iterrows():
        error, steps = raw[r["chan_id"]]
        e_s = pd.Series(error).ewm(span=span).mean().to_numpy()[WARMUP:]
        st = steps[WARMUP:]

        eps = find_threshold(e_s)
        if eps is None:
            kept = []
            no_thr += 1
        else:
            flags = e_s > eps
            kept = prune(e_s, flags, verbose=False) if do_prune else flags_to_sequences(flags)
            if do_prune and not kept:
                pruned_all += 1

        pred = [(int(st[s]), int(st[e])) for s, e in kept]
        rows.append(score_events(pred, r["anomaly_sequences"], r["class"]))

    df = pd.DataFrame(rows)
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    p, rc = precision_recall(tp, fp, fn)
    return {
        "span": span, "prune": do_prune, "TP": tp, "FP": fp, "FN": fn,
        "precision": round(p, 2), "recall": round(rc, 2),
        "F1": round(2 * tp / max(2 * tp + fp + fn, 1), 2),
        "point_rec": round(df.tp_point.sum() / max(df.tp_point.sum() + df.fn_point.sum(), 1), 2),
        "ctx_rec": round(df.tp_ctx.sum() / max(df.tp_ctx.sum() + df.fn_ctx.sum(), 1), 2),
        "no_threshold": no_thr, "all_pruned": pruned_all,
    }


results = [evaluate(span, do_prune)
           for span in [875, 300, 100, 25]
           for do_prune in [True, False]]

out = pd.DataFrame(results)
print(out.to_string(index=False))
out.to_csv("ablation_results.csv", index=False)