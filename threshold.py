import sys
import numpy as np
from evaluate import load_labels, flags_to_sequences, score_events, precision_recall

WARMUP = 100                        # skip the EWMA warm-up spike at the start
Z_VALUES = np.arange(2.5, 10.01, 0.5)
MAX_SEQS = 5                        # as I recall the Telemanom code; check the repo


def find_threshold(e_s):
    mean, sd = e_s.mean(), e_s.std()
    best_score, best_eps = -np.inf, None

    for z in Z_VALUES:
        eps = mean + z * sd
        above = e_s > eps
        n_pts = int(above.sum())
        if n_pts == 0:
            continue

        # sequences of more than one point
        seqs = [s for s in flags_to_sequences(above) if s[1] > s[0]]

        kept = e_s[~above]
        mean_drop = (mean - kept.mean()) / mean
        sd_drop = (sd - kept.std()) / sd
        score = (mean_drop + sd_drop) / (n_pts + len(seqs) ** 2)

        if score >= best_score and len(seqs) <= MAX_SEQS and n_pts < 0.5 * len(e_s):
            best_score, best_eps = score, eps

    return best_eps


if __name__ == "__main__":
    chan = sys.argv[1] if len(sys.argv) > 1 else "E-2"
    z = np.load(f"errors/{chan}.npz")
    e_s = z["smoothed"][WARMUP:]
    steps = z["steps"][WARMUP:]

    eps = find_threshold(e_s)
    row = load_labels().query("chan_id == @chan").iloc[0]

    print(f"Channel {chan}")
    if eps is None:
        print("No threshold found: nothing flagged.")
        pred_seqs = []
    else:
        flags = e_s > eps
        idx_seqs = flags_to_sequences(flags)
        pred_seqs = [(int(steps[s]), int(steps[e])) for s, e in idx_seqs]
        print(f"mean={e_s.mean():.4f}  std={e_s.std():.4f}  threshold={eps:.4f}")
        print(f"Flagged {int(flags.sum())} points in {len(pred_seqs)} sequences:")
        for (a, b), (s, e) in zip(pred_seqs, idx_seqs):
            print(f"   steps {a}-{b}   peak error = {e_s[s:e+1].max():.4f}")

    print("Labeled anomalies:", row["anomaly_sequences"])
    sc = score_events(pred_seqs, row["anomaly_sequences"], row["class"])
    p, r = precision_recall(sc["tp"], sc["fp"], sc["fn"])
    print(f"TP={sc['tp']} FP={sc['fp']} FN={sc['fn']}  precision={p:.2f} recall={r:.2f}")