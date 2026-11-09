import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info messages

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from windows import load_channel

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Settings (the layer sizes and dropout come from the paper)
UNITS = 80
DROPOUT = 0.3
BATCH_SIZE = 70
MAX_EPOCHS = 35
PATIENCE = 10
MIN_DELTA = 0.0003


def build_model(window, n_features):
    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(UNITS, return_sequences=True),
        layers.Dropout(DROPOUT),
        layers.LSTM(UNITS, return_sequences=False),
        layers.Dropout(DROPOUT),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_channel(chan_id, verbose=2):
    d = load_channel(chan_id)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]

    model = build_model(X_train.shape[1], X_train.shape[2])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, min_delta=MIN_DELTA,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        shuffle=False,            # keep time order
        callbacks=[early_stop],
        verbose=verbose,
    )

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{chan_id}.keras")
    return model, history, d


if __name__ == "__main__":
    chan = sys.argv[1] if len(sys.argv) > 1 else "P-1"
    model, history, d = train_channel(chan)

    best_val = min(history.history["val_loss"])
    print(f"\nEpochs run: {len(history.history['loss'])}")
    print(f"Best validation MSE: {best_val:.5f}")

    # Sanity check: a "lazy" predictor that just repeats the last telemetry value
    lazy_val_mse = np.mean((d["X_val"][:, -1, 0] - d["y_val"]) ** 2)
    print(f"Lazy 'repeat last value' validation MSE: {lazy_val_mse:.5f}")

    # Loss curve
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(f"{chan} - training curve")
    plt.legend()
    plt.savefig(f"loss_{chan}.png", dpi=120)
    plt.show()