import os
from evaluate import load_labels

chans = list(load_labels()["chan_id"].unique())
missing = [c for c in chans if not os.path.exists(f"models/{c}.keras")]

print(f"Expected {len(chans)}, found {len(chans) - len(missing)}, missing {len(missing)}")
print(missing)