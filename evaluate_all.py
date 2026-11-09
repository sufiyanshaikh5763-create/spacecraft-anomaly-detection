import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
from evaluate import load_labels, score_events, precision_recall
from errors import get_errors
from threshold import find_threshold, WARMUP
from prune import prune

labels = load_labels()
rows = []

for _, r in labels.iterrows():
    chan = r["chan_id"]
    if not os.path.exists(f"models/{chan}.keras"):
        print(f"{chan}: no model, skipping")
        continue

    _, _, _, smoothed, steps = get_errors(chan)
    e_s, st = smoothed[WARMUP:], steps[WARMUP:]

    eps = find_threshold(e_s)
    kept = [] if eps is None else prune(e_s, e_s > eps, verbose=False)
    pred_seqs = [(int(st[s]), int(st[e])) for s, e in kept]

    s = score_events(pred_seqs, r["anomaly_sequences"], r["class"])
    s.update({"chan_id": chan, "spacecraft": r["spacecraft"], "n_pred": len(pred_seqs)})
    rows.append(s)
    print(f"{chan}: TP={s['tp']} FP={s['fp']} FN={s['fn']}")

res = pd.DataFrame(rows)
res.to_csv("lstm_results.csv", index=False)


def summarize(df, name):
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    p, rc = precision_recall(tp, fp, fn)
    pt = df.tp_point.sum() / max(df.tp_point.sum() + df.fn_point.sum(), 1)
    cx = df.tp_ctx.sum() / max(df.tp_ctx.sum() + df.fn_ctx.sum(), 1)
    print(f"{name:14s} TP={tp:3d} FP={fp:4d} FN={fn:3d} | precision={p:.2f} "
          f"recall={rc:.2f} | recall point={pt:.2f} contextual={cx:.2f}")


print("\n=== LSTM ===")
summarize(res, "ALL")
summarize(res[res.spacecraft == "SMAP"], "SMAP")
summarize(res[res.spacecraft == "MSL"], "MSL")

# Baseline on the same channels, for a fair side-by-side
base = pd.read_csv("baseline_results.csv")
base = base[base.chan_id.isin(res.chan_id)]
print("\n=== Baseline (same channels) ===")
summarize(base, "ALL")
summarize(base[base.spacecraft == "SMAP"], "SMAP")
summarize(base[base.spacecraft == "MSL"], "MSL")