# Property inference evaluation

Evaluate one saved reconstruction run from the repository root:

```bash
python src/pi_eval/evaluate_reconstruction_run.py reconstructions/ETTh1_<timestamp>
```

Omit the folder to select the latest timestamped run under `reconstructions/`:

```bash
python src/pi_eval/evaluate_reconstruction_run.py
```

Generate a separate report for every immediate subfolder in `reconstructions/`:

```bash
python src/pi_eval/evaluate_reconstruction_run.py all
```

`all` also works with `--reconstructions-dir /path/to/runs`. If a folder fails,
the script reports its error, continues with the remaining folders, and exits
with a nonzero status after processing them all.

Reports are written to `evaluation_results/<run-folder>_property_inference.md`.
Re-evaluating the same run replaces its report. The report includes all batch tables,
a combined table with each property's values averaged across batches, the complete `config.json`,
and the effective evaluation settings.

The script reuses `property_inference.py` and supports ETT datasets. The validation
baseline uses the CSV path and preprocessing from the saved config. If the data has
moved, pass `--root-path /path/to/ETT-small`. Use `--help` for thresholds, durations,
baseline sampling options, or custom reconstruction/output directories.

Combined results keep every condition × threshold × duration as a separate row.
Columns are Duration, Channel, Threshold, Mean reconstruction error, Mean random
error, Mean reconstruction error - Mean random error, List of all reconstruction
errors, and List of all random errors. Means give each batch equal weight and use
unrounded values. Lists follow the numeric batch order printed above the table;
each random error is the mean absolute error across validation draws for that
batch. All errors and differences are in percentage points. Negative differences
mean reconstruction outperforms the random baseline. Zero errors are included.
Partial runs are reported explicitly.
Equal-shaped batches use the same seeded validation draws.

To also log combined results to an existing W&B run, pass the optional Python
argument `wandb_run`:

```python
from pi_eval.evaluate_reconstruction_run import main

main(["reconstructions/ETTh1_<timestamp>"], wandb_run=wandb.run)
```

The attack worker passes its active W&B run automatically when W&B logging is
enabled. This does not create or finish a W&B run. Without this argument,
evaluation only writes Markdown, as before.

Each row logs five keys under `<Channel>/<Threshold>/<Duration>/`, using the
column names above verbatim; for example, `HUFL/0.5/1/Mean reconstruction error`.
Scalars and error lists retain full precision. Lists follow the numeric batch
order, also logged as `property_inference/batch_indices`. Each folder produces
one history entry with `property_inference/run_folder`, so `main(["all"],
wandb_run=wandb.run)` keeps separate entries for each evaluated folder.

The single-file script remains available:

```bash
python src/pi_eval/property_inference.py reconstruction.pt --root-path /path/to/ETT-small --output report.md
```
