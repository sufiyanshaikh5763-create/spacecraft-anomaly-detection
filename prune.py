import sys
import numpy as np
from evaluate import load_labels, flags_to_sequences, score_events, precision_recall
from threshold import find_threshold, WARMUP   # your threshold.py from Step 8

P = 0.13        # minimum percent drop between neighbouring peaks
BUFFER = 100    # steps excluded around flagged sequences when finding max normal error


def prune(e_s, flags, buffer=BUFFER, verbose=True):
    seqs = flags_to_sequences(flags)
    if not seqs:
        return []

    peaks = np.array([e_s[s:e + 1].max() for s, e in seqs])

    # highest error among normal points, with and without the buffer
    masked = np.zeros(len(e_s), dtype=bool)
    for s, e in seqs:
        masked[max(0, s - buffer): e + buffer + 1] = True
    max_normal_plain = e_s[~flags].max()
    max_normal = e_s[~masked].max() if (~masked).any() else max_normal_plain

    order = np.argsort(peaks)[::-1]                    # highest peak first
    ranked = np.append(peaks[order], max_normal)
    drops = (ranked[:-1] - ranked[1:]) / ranked[:-1]

    if verbose:
        print(f"max normal error: {max_normal:.4f} with buffer, "
              f"{max_normal_plain:.4f} without")
        for rank, i in enumerate(order):
            print(f"   rank {rank}: peak={peaks[i]:.4f}  drop to next={drops[rank]*100:.1f}%")

    big = np.where(drops >= P)[0]
    if len(big) == 0:
        return []                                      # no clear cliff: all normal
    keep = order[: big.max() + 1]                      # everything above the last big drop
    return [seqs[i] for i in sorted(keep)]


if __name__ == "__main__":
    chan = sys.argv[1] if len(sys.argv) > 1 else "E-2"
    z = np.load(f"errors/{chan}.npz")
    e_s = z["smoothed"][WARMUP:]
    steps = z["steps"][WARMUP:]

    eps = find_threshold(e_s)
    row = load_labels().query("chan_id == @chan").iloc[0]
    print(f"Channel {chan}")

    if eps is None:
        print("No threshold found: nothing flagged, nothing to prune.")
        kept = []
    else:
        flags = e_s > eps
        n_before = len(flags_to_sequences(flags))
        kept = prune(e_s, flags)
        print(f"Sequences before pruning: {n_before}, after: {len(kept)}")

    pred_seqs = [(int(steps[s]), int(steps[e])) for s, e in kept]
    print("Kept:", pred_seqs)
    print("Labeled:", row["anomaly_sequences"])
    sc = score_events(pred_seqs, row["anomaly_sequences"], row["class"])
    p, r = precision_recall(sc["tp"], sc["fp"], sc["fn"])
    print(f"TP={sc['tp']} FP={sc['fp']} FN={sc['fn']}  precision={p:.2f} recall={r:.2f}")