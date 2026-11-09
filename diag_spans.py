import numpy as np
import pandas as pd
from evaluate import load_labels

chan = "P-1"
z = np.load(f"errors/{chan}.npz")
error, steps = z["error"], z["steps"]
seqs = load_labels().query("chan_id == @chan").iloc[0]["anomaly_sequences"]

in_any = np.zeros(len(steps), dtype=bool)
for s, e in seqs:
    in_any |= (steps >= s) & (steps <= e)

for span in [1, 25, 100, 300, 875]:
    sm = pd.Series(error).ewm(span=span).mean().to_numpy()
    out = sm[~in_any]
    print(f"\nspan={span}: outside  max={out.max():.3f}  99th pct={np.percentile(out, 99):.3f}")
    for s, e in seqs:
        m = (steps >= s) & (steps <= e)
        print(f"   anomaly {s}-{e}: peak={sm[m].max():.3f}")