import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DATA_DIR = "data/data"
WINDOW = 250


def make_windows(data, window=WINDOW):
    """
    data: array of shape (n_steps, n_features), column 0 = telemetry.
    Returns:
      X: (n_steps - window, window, n_features)
      y: (n_steps - window,)  telemetry value right after each window
    """
    # all windows of length `window`, then drop the last one (it has no next step)
    X = sliding_window_view(data, (window, data.shape[1]))[:, 0]
    X = X[:-1].astype(np.float32)
    y = data[window:, 0].astype(np.float32)
    return X, y


def load_channel(chan_id, window=WINDOW, val_fraction=0.2):
    train = np.load(f"{DATA_DIR}/train/{chan_id}.npy")
    test = np.load(f"{DATA_DIR}/test/{chan_id}.npy")

    X_train_all, y_train_all = make_windows(train, window)
    X_test, y_test = make_windows(test, window)

    # time-ordered split: last 20% of training windows become validation (no shuffling)
    split = int(len(X_train_all) * (1 - val_fraction))
    X_train, y_train = X_train_all[:split], y_train_all[:split]
    X_val, y_val = X_train_all[split:], y_train_all[split:]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "test_offset": window,   # prediction i  <->  test step i + window
    }


if __name__ == "__main__":
    train = np.load(f"{DATA_DIR}/train/P-1.npy")
    d = load_channel("P-1")

    for name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        print(f"{name:8s}", d[name].shape)

    # sanity checks: window 0 is rows 0..249, its target is telemetry at row 250
    assert np.allclose(d["X_train"][0], train[0:WINDOW])
    assert np.isclose(d["y_train"][0], train[WINDOW, 0])
    # last window's target is the final row of the file
    assert np.isclose(d["y_test"][-1], np.load(f"{DATA_DIR}/test/P-1.npy")[-1, 0])
    print("All checks passed.")