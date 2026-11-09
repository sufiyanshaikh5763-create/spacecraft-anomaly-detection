import numpy as np
import pandas as pd
from evaluate import (DATA_DIR, load_labels, flags_to_sequences,
                      score_events, precision_recall)

labels = load_labels()
rows = []

for _, r in labels.iterrows():
    chan = r["chan_id"]
    train = np.load(f"{DATA_DIR}/train/{chan}.npy")[:, 0]
    test = np.load(f"{DATA_DIR}/test/{chan}.npy")[:, 0]

    mu, sigma = train.mean(), train.std()
    flags = np.abs(test - mu) > 3 * sigma
    pred_seqs = flags_to_sequences(flags)

    s = score_events(pred_seqs, r["anomaly_sequences"], r["class"])
    s.update({"chan_id": chan, "spacecraft": r["spacecraft"], "n_pred": len(pred_seqs)})
    rows.append(s)

res = pd.DataFrame(rows)
res.to_csv("baseline_results.csv", index=False)


def summarize(df, name):
    p, rc = precision_recall(df.tp.sum(), df.fp.sum(), df.fn.sum())
    pt = df.tp_point.sum() / max(df.tp_point.sum() + df.fn_point.sum(), 1)
    cx = df.tp_ctx.sum() / max(df.tp_ctx.sum() + df.fn_ctx.sum(), 1)
    print(f"{name:6s} TP={df.tp.sum():3d} FP={df.fp.sum():4d} FN={df.fn.sum():3d} "
          f"| precision={p:.2f} recall={rc:.2f} "
          f"| recall point={pt:.2f} contextual={cx:.2f}")


summarize(res, "ALL")
summarize(res[res.spacecraft == "SMAP"], "SMAP")
summarize(res[res.spacecraft == "MSL"], "MSL")

print("\nTwo channels we plotted earlier:")
print(res[res.chan_id.isin(["S-1", "P-1"])][["chan_id", "tp", "fp", "fn", "n_pred"]])