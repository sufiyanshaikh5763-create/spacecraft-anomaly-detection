import os
from evaluate import load_labels
from train_lstm import train_channel

labels = load_labels()
for chan in labels["chan_id"].unique():
    if os.path.exists(f"models/{chan}.keras"):
        print(f"{chan}: already trained, skipping")
        continue
    print(f"Training {chan}...")
    train_channel(chan, verbose=0)