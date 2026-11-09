import ast
import pandas as pd

DATA_DIR = "data/data"


def load_labels():
    labels = pd.read_csv(f"{DATA_DIR}/labeled_anomalies.csv")
    labels["anomaly_sequences"] = labels["anomaly_sequences"].apply(ast.literal_eval)
    labels["class"] = labels["class"].apply(
        lambda s: [c.strip() for c in s.strip("[]").split(",")]
    )
    return labels


def flags_to_sequences(flags):
    """Turn a boolean array into a list of (start, end) runs of True (end inclusive)."""
    seqs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            seqs.append((start, i - 1))
            start = None
    if start is not None:
        seqs.append((start, len(flags) - 1))
    return seqs


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def score_events(pred_seqs, true_seqs, true_classes):
    """
    Event-level scoring:
      TP = labeled anomaly overlapped by at least one predicted sequence
      FN = labeled anomaly overlapped by none
      FP = predicted sequence that overlaps no labeled anomaly
    """
    tp = fn = 0
    by_class = {"point": [0, 0], "contextual": [0, 0]}   # [tp, fn]

    for t, cls in zip(true_seqs, true_classes):
        hit = any(overlaps(p, t) for p in pred_seqs)
        if hit:
            tp += 1
            by_class[cls][0] += 1
        else:
            fn += 1
            by_class[cls][1] += 1

    fp = sum(1 for p in pred_seqs if not any(overlaps(p, t) for t in true_seqs))

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "tp_point": by_class["point"][0], "fn_point": by_class["point"][1],
        "tp_ctx": by_class["contextual"][0], "fn_ctx": by_class["contextual"][1],
    }


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall