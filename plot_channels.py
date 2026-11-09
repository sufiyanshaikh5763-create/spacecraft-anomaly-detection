import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data/data"

# Load labels and convert the string columns into real Python lists
labels = pd.read_csv(f"{DATA_DIR}/labeled_anomalies.csv")
labels["anomaly_sequences"] = labels["anomaly_sequences"].apply(ast.literal_eval)
labels["class"] = labels["class"].apply(lambda s: [c.strip() for c in s.strip("[]").split(",")])
    


def plot_channel(chan_id):
    row = labels[labels["chan_id"] == chan_id].iloc[0]
    sequences = row["anomaly_sequences"]
    classes = row["class"]

    train = np.load(f"{DATA_DIR}/train/{chan_id}.npy")[:, 0]  # column 0 = telemetry
    test = np.load(f"{DATA_DIR}/test/{chan_id}.npy")[:, 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

    # Training data (all normal)
    ax1.plot(train, color="steelblue", linewidth=0.8)
    ax1.set_title(f"{chan_id} ({row['spacecraft']}) - training telemetry (normal)")

    # Test data with shaded anomalies
    ax2.plot(test, color="steelblue", linewidth=0.8)
    for (start, end), cls in zip(sequences, classes):
        color = "red" if cls == "point" else "orange"
        ax2.axvspan(start, end, color=color, alpha=0.3, label=cls)
    ax2.set_title(f"{chan_id} - test telemetry with labeled anomalies")
    ax2.set_xlabel("time step")

    # Avoid duplicate legend entries
    handles, names = ax2.get_legend_handles_labels()
    unique = dict(zip(names, handles))
    ax2.legend(unique.values(), unique.keys(), loc="upper right")

    plt.tight_layout()
    plt.savefig(f"plot_{chan_id}.png", dpi=120)
    plt.show()


plot_channel("S-1")   # point anomaly
plot_channel("P-1")   # contextual anomalies