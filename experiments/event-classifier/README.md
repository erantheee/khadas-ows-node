# Edge2 DMIC event classifier (2026-11-09)

Replaces hand-tuned thresholds for the board's sound-event layer. 34-dim
1 s window features -> 4 classes (quiet / speech / keyboard / wind).

Run:

    python3 train_event_model.py

Current data are the board's existing labeled clips. Two of them (wind,
keyboard) were recorded in a noisy office, so their labels are contaminated:
the first model reaches 0.721 temporal-split test accuracy, but wind recall is
only 21% and keyboard 38%. Treat the model as a pipeline proof, not an
acceptance result.

Next step: one clean labeled capture on the board (48 kHz stereo, continuous
segments are already written to 记录/dmic-dataset/continuous/):

    quiet 20 s -> speech 20 s -> blowing at the board 20 s -> quiet 20 s

Then retrain and report the holdout confusion matrix before switching the
live event stream from the rule fallback to the model.
