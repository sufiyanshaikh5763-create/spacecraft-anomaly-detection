import pandas as pd

r = pd.read_csv("lstm_results.csv")
none = r[r.n_pred == 0]
some = r[r.n_pred > 0]

print(f"Channels evaluated: {len(r)}")
print(f"Channels with NO flagged sequence: {len(none)}  (FN there: {none.fn.sum()})")
print(f"Channels with flags:               {len(some)}  (TP {some.tp.sum()}, FP {some.fp.sum()}, FN {some.fn.sum()})")
print("\nChannels with flags:")
print(some[["chan_id", "spacecraft", "tp", "fp", "fn", "n_pred"]].to_string(index=False))