import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from evaluate import load_labels, flags_to_sequences, score_events
from errors import get_errors          # use the same name evaluate_all.py imports
from threshold import find_threshold, WARMUP
from prune import prune

SPAN = 25            # chosen smoothing span
DO_PRUNE = True

labels = load_labels()


def detect(error, steps, span=SPAN, do_prune=DO_PRUNE):
    e_s = pd.Series(error).ewm(span=span).mean().to_numpy()[WARMUP:]
    st = steps[WARMUP:]
    eps = find_threshold(e_s)
    if eps is None:
        return e_s, st, None, []
    flags = e_s > eps
    kept = prune(e_s, flags, verbose=False) if do_prune else flags_to_sequences(flags)
    return e_s, st, eps, [(int(st[s]), int(st[e])) for s, e in kept]


# Score every channel at the chosen setting, to pick channels to plot
base = pd.read_csv("baseline_results.csv").drop_duplicates("chan_id").set_index("chan_id")
rows = []
for _, r in labels.drop_duplicates("chan_id").iterrows():
    z = np.load(f"errors/{r['chan_id']}.npz")
    _, _, _, pred = detect(z["error"], z["steps"])
    sc = score_events(pred, r["anomaly_sequences"], r["class"])
    rows.append({"chan_id": r["chan_id"], "lstm_tp": sc["tp"],
                 "base_tp": int(base.loc[r["chan_id"], "tp"])})
cmp = pd.DataFrame(rows)
cmp["diff"] = cmp.lstm_tp - cmp.base_tp
lstm_win = cmp.sort_values("diff", ascending=False).iloc[0]
base_win = cmp.sort_values("diff").iloc[0]

chosen = [("E-2", "success")]
chosen.append(("P-1", "failure"))
if lstm_win["diff"] > 0:
    chosen.append((lstm_win["chan_id"], "lstm_beats_baseline"))
if base_win["diff"] < 0:
    chosen.append((base_win["chan_id"], "baseline_beats_lstm"))
print("Plotting:", chosen)


def plot_channel(chan, tag):
    pred, actual, error, _, steps = get_errors(chan)
    e_s, st, eps, pred_seqs = detect(error, steps)
    row = labels[labels.chan_id == chan].iloc[0]
    sc = score_events(pred_seqs, row["anomaly_sequences"], row["class"])

    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(steps, actual, lw=0.8, label="actual")
    ax[0].plot(steps, pred, lw=0.8, alpha=0.8, label="LSTM prediction")
    ax[0].set_title(f"{chan} ({row['spacecraft']}) - TP={sc['tp']} FP={sc['fp']} FN={sc['fn']}")
    ax[0].legend(loc="upper right")

    ax[1].plot(st, e_s, color="crimson", lw=1)
    if eps is not None:
        ax[1].axhline(eps, color="black", ls="--")
    for s, e in pred_seqs:
        ax[1].axvspan(s, e, color="green", alpha=0.35)
    ax[1].set_title(f"smoothed error (EWMA span {SPAN}) and threshold")
    ax[1].set_xlabel("time step")
    ax[1].legend(handles=[
        Patch(color="orange", alpha=0.3, label="labeled anomaly"),
        Patch(color="green", alpha=0.35, label="flagged by LSTM pipeline"),
        plt.Line2D([0], [0], color="black", ls="--", label="threshold")],
        loc="upper right")

    for a in ax:
        for s, e in row["anomaly_sequences"]:
            a.axvspan(s, e, color="orange", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"final_{tag}_{chan}.png", dpi=130)
    plt.close()


for chan, tag in chosen:
    plot_channel(chan, tag)
print("Saved final_*.png")