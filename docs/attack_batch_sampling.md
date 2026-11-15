# Property-percentile attack batch sampling

Enable this in `notebooks/ts_inverse_har.py`:

```python
"attack_batch_sampling": "property_percentiles",
"attack_sampling_target_channel": "HUFL",
"attack_sampling_threshold": 0.5,
"attack_sampling_duration": 1,
"attack_sampling_candidate_batches": None,
```

The default `"random"` retains the existing dataloader behavior. Percentile
sampling currently supports ETT datasets and does not require model poisoning.
The channel can be a CSV column name or zero-based input index. If channel,
threshold, or duration options are omitted, their corresponding poison settings
are used, with defaults of 0.5 for threshold and 1 for duration.

Candidates are full batches from one seeded random partition of the benign
training windows. By default all full batches are scored; a positive
`attack_sampling_candidate_batches` caps the pool. The pool must contain at least
`number_of_batches` batches. Remainder windows that cannot form a full batch are
excluded. `attack_sampling_seed` defaults to the experiment seed and uses its own
generator, keeping the selections reproducible across clean and poisoned models.

The score is the mean per-window fraction of input timesteps in elevated runs on
the selected channel. Values must be strictly above the threshold, and a run must
last at least the specified duration. Thresholds use preprocessed tensor units;
all scores are fractions in [0, 1]. Other channels, labels, and timestamps are not
modified. Distinct sliding windows can still overlap in time.

Candidates are sorted by score. The requested attack batches are chosen at evenly
spaced ranks from minimum to maximum and attacked in ascending score order. A
single attack selects the median rank. Ties use stable candidate order; distinct
batches can have the same score. When `number_of_batches` exceeds
`attack_number_of_batches`, the first entries are still spread across the full
pool; the remaining recording-only entries are selected from unused candidates.

Console output includes settings, scanning progress, min/max/mean/std, score
percentiles with a text chart, and every selected batch's rank percentile, score,
and window indices. Identical candidate scores produce an explicit warning.
Random batches can still have a narrow score range; this selection spans the
available distribution rather than constructing new extreme batches.

The run's `config.json` stores resolved settings, percentile values, and selected
batch/window indices and scores. `attack_sampling_percentiles.png` shows the
candidate percentile curve and selected attack batches alongside the saved
reconstructions. Existing property evaluation reads the ordered batch files.

This is controlled experimental sampling using access to the benign dataset;
results describe the selected property strata rather than random-client performance.
