import numpy as np
import pandas as pd

DATA_DIR = "data/data"   # the inner data folder

labels = pd.read_csv(f"{DATA_DIR}/labeled_anomalies.csv")
print(labels.head())
print("Total channels:", len(labels))
print(labels["spacecraft"].value_counts())

train = np.load(f"{DATA_DIR}/train/P-1.npy")
test = np.load(f"{DATA_DIR}/test/P-1.npy")
print("train shape:", train.shape)
print("test shape:", test.shape)
print(labels['anomaly_sequences'].dtypes)