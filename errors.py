import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from windows import load_channel
from evaluate import load_labels

# EWMA span. As I recall the Telemanom config, it is batch_size * window * 0.05
# = 70 * 250 * 0.05 = 875. Check it against the repo's config.yaml.
SMOOTH_SPAN = 875


def get_errors(chan_id, span=SMOOTH_SPAN):
    d = load_channel(chan_id)
    model = keras.models.load_model(f"models/{chan_id}.keras")

    pred = model.predict(d["X_test"], batch_size=256, verbose=0).ravel()
    actual = d["y_test"]

    error = np.abs(pred - actual)
    smoothed = pd.Series(error).ewm(span=span).mean().to_numpy()
    steps = np.arange(len(pred)) + d["test_offset"]   # prediction i <-> test step i + 250
    return pred, actual, error, smoothed, steps


if __name__ == "__main__":
    chan = sys.argv[1] if len(sys.argv) > 1 else "P-1"
    pred, actual, error, smoothed, steps = get_errors(chan)

    os.makedirs("errors", exist_ok=True)
    np.savez(f"errors/{chan}.npz", pred=pred, actual=actual,
             error=error, smoothed=smoothed, steps=steps)

    row = load_labels().query("chan_id == @chan").iloc[0]
    seqs = row["anomaly_sequences"]

    # Mark which predicted steps fall inside a labeled anomaly
    in_anom = np.zeros(len(steps), dtype=bool)
    for s, e in seqs:
        in_anom |= (steps >= s) & (steps <= e)

    print(f"Channel {chan}")
    print(f"Mean raw error      inside anomalies : {error[in_anom].mean():.4f}")
    print(f"Mean raw error      outside anomalies: {error[~in_anom].mean():.4f}")
    print(f"Mean smoothed error inside anomalies : {smoothed[in_anom].mean():.4f}")
    print(f"Mean smoothed error outside anomalies: {smoothed[~in_anom].mean():.4f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(steps, actual, label="actual", linewidth=0.8)
    axes[0].plot(steps, pred, label="prediction", linewidth=0.8, alpha=0.8)
    axes[0].set_title(f"{chan} - actual vs predicted (test)")
    axes[0].legend(loc="upper right")

    axes[1].plot(steps, smoothed, color="crimson", linewidth=1)
    axes[1].set_title("smoothed error (EWMA)")
    axes[1].set_xlabel("time step")

    for ax in axes:
        for s, e in seqs:
            ax.axvspan(s, e, color="orange", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"errors_{chan}.png", dpi=120)
    plt.show()