# Spacecraft Anomaly Detection with LSTMs

Anomaly detection on real NASA spacecraft telemetry (SMAP satellite and MSL Mars rover). One LSTM per channel learns to predict normal behaviour, and the moments where reality stops matching the prediction are flagged. The pipeline is compared against a simple 3-sigma baseline on 81 channels and 105 labeled anomalies.

It follows the approach of Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding* (KDD 2018, NASA JPL).

## The idea in plain words

Spacecraft send back thousands of readings, and no human can watch them all. Instead of listing every possible fault, we show a model a lot of **normal** data and ask it to guess the next reading. When the real value is far from the guess, something may be wrong.

There are two kinds of anomalies in the dataset:

- **Point anomalies:** the value leaves its normal range (easy to see).
- **Contextual anomalies:** the value looks fine on its own but is wrong for the situation (hard to see, and invisible to a simple range check).

## How it works

1. **Windows:** the last 250 steps of every column (telemetry plus commands) are the input, and the next telemetry value is the target.
2. **Model:** two LSTM layers (80 units, 0.3 dropout) and a dense output layer, trained on normal data only, one model per channel.
3. **Errors:** absolute difference between prediction and reality on the test file, smoothed with an exponentially weighted moving average (EWMA).
4. **Threshold:** chosen automatically, without labels, by trying `mean + z * std` for z from 2.5 to 10 and keeping the line that removes the most extreme errors while flagging as little as possible.
5. **Pruning:** flagged sequences are ranked by peak error, and weak ones without a clear drop (at least 13 percent) below them are treated as normal.
6. **Scoring:** event-level. A labeled anomaly counts as found if any flagged sequence overlaps it. Flagged sequences that overlap nothing are false alarms.

## Results

Compared on all labeled anomalies in the dataset (105 anomalies; the channel P-2 is listed twice in the labels file and is counted twice).

| Method | Precision | Recall | F1 | Point recall | Contextual recall |
|---|---|---|---|---|---|
| 3-sigma baseline | 0.05 | 0.55 | 0.09 | 0.77 | 0.23 |
| LSTM pipeline, EWMA span 875, pruning on | 0.72 | 0.20 | 0.31 | 0.31 | 0.05 |
| LSTM pipeline, EWMA span 25, pruning on | 0.82 | 0.51 | 0.63 | 0.68 | 0.28 |
| LSTM pipeline, EWMA span 25, pruning off | 0.71 | 0.64 | 0.67 | 0.79 | 0.42 |

What this shows:

- The LSTM pipeline raises far fewer false alarms than the baseline (precision 0.71 to 0.82 versus 0.05).
- With the heavy smoothing setting (span 875), 55 of the 82 label entries produced no alarm at all, which caused the low recall. Lighter smoothing recovered most of it. The full ablation over four spans, with pruning on and off, is in `results/ablation_results.csv`.
- With lighter smoothing the pipeline also catches more contextual anomalies than the baseline, although the margin with pruning on is small.
- Some channels remain hard. For example, P-1 is never flagged.

**Important caveat:** the span-25 settings were chosen after looking at results on the labeled data, so those numbers are optimistic. The span-875 row shows the original setting for comparison. A cleaner evaluation would choose the setting on some channels and report on others.

### Example plots

![E-2: a success](results/final_success_E-2.png)
![P-1: an honest failure](results/final_failure_P-1.png)

<!-- Add the other two final_*.png images from results/ here -->

Each plot shows the real telemetry with the model's prediction, and below it the smoothed error with the threshold line. The labeled anomaly is shaded orange and the flagged region green.

## Project structure

```
.
├── check_data.py        # check the data and labels load and line up
├── plot_channels.py     # plot a point and a contextual channel with anomalies shaded
├── windows.py           # build the 250-step windows and the validation split
├── evaluate.py          # shared helpers: load labels, group flags into events, score
├── baseline.py          # the 3-sigma baseline -> baseline_results.csv
├── train_lstm.py        # train one LSTM channel and save the model
├── errors.py            # predict, compute and smooth the errors
├── diag_spans.py        # compare error peaks at different smoothing spans
├── threshold.py         # automatic threshold without labels
├── prune.py             # prune weak flagged sequences
├── train_all.py         # train every channel, skipping finished ones
├── check_models.py      # list channels with no saved model
├── evaluate_all.py      # score all channels -> lstm_results.csv
├── diag_misses.py       # show where the missed anomalies come from
├── ablation.py          # test smoothing spans and pruning -> ablation_results.csv
├── plot_final.py        # draw the final figures
├── requirements.txt
└── results/             # CSVs and final plots
```

## Setup

Tested with Python 3.13 and TensorFlow 2.21 on Windows (CPU only).

```
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Data

The data is not included in this repository. Download the NASA SMAP and MSL telemetry dataset released with the Telemanom paper (a copy is available on Kaggle; the original is in the [Telemanom repository](https://github.com/khundman/telemanom)). Check the dataset's own license before redistributing it.

The scripts expect this layout:

```
data/
└── data/
    ├── train/                  # one .npy file per channel (normal behaviour)
    ├── test/                   # one .npy file per channel (contains anomalies)
    └── labeled_anomalies.csv
```

If your data sits elsewhere, change `DATA_DIR` at the top of the scripts.

## Running it

Run the scripts in this order:

```
python check_data.py           # sanity check
python plot_channels.py        # look at the data
python baseline.py             # baseline results
python train_lstm.py P-1       # train one channel (try this first)
python train_all.py            # train all channels (takes hours on CPU)
python evaluate_all.py         # LSTM results and comparison
python diag_misses.py          # where do the misses come from?
python ablation.py             # smoothing and pruning ablation
python plot_final.py           # final figures
```

Models are saved to `models/` and cached errors to `errors/`. Both are created by the scripts and are not committed.

**Keras version note:** a model saved with one Keras version may fail to load under an older one. Train and evaluate in the same environment.

## Limitations

- The smoothing setting was chosen using labeled results (see above).
- The threshold search runs once over each whole test file, not over sliding windows as in the original method, so results will not match the paper exactly.
- The model cannot predict the first 250 steps of a file and the first 100 smoothed values are skipped, so anomalies in the first 350 test steps cannot be found by the LSTM pipeline.
- The 3-sigma baseline is deliberately naive: it fragments its flags, which inflates its false alarms, and some of its hits are chance.
- A flagged event that overlaps a labeled anomaly counts as a hit even if it covers only a small part of it.
- Some hyperparameters (EWMA span, sequence cap, error buffer, training settings) follow the original Telemanom configuration as understood by the author and were not exhaustively re-verified against the reference code.

## Reference

Hundman, K., Constantinou, V., Laporte, C., Colwell, I., and Soderstrom, T. (2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Original code: [khundman/telemanom](https://github.com/khundman/telemanom).
