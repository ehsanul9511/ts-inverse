# Property inference evaluation

Evaluate one saved reconstruction run from the repository root:

```bash
python src/pi_eval/evaluate_reconstruction_run.py reconstructions/ETTh1_<timestamp>
```

Omit the folder to select the latest timestamped run under `reconstructions/`:

```bash
python src/pi_eval/evaluate_reconstruction_run.py
```

Reports are written to `evaluation_results/<run-folder>_property_inference.md`.
Re-evaluating the same run replaces its report. The report includes all batch tables,
a combined table with each property's values averaged across batches, the complete `config.json`,
and the effective evaluation settings.

The script reuses `property_inference.py` and supports ETT datasets. The validation
baseline uses the CSV path and preprocessing from the saved config. If the data has
moved, pass `--root-path /path/to/ETT-small`. Use `--help` for thresholds, durations,
baseline sampling options, or custom reconstruction/output directories.

Combined results keep every condition × threshold × duration as a separate row.
Each column is averaged across batches with equal batch weight, using unrounded
values. The error-reduction column averages the individual batches' percentages
for that exact property; it does not compute a ratio of averaged errors or average
different properties together. Zero baseline error is N/A and excluded only from
that row's error-reduction average. Each row shows the contributing batch count.
Other columns include all batches. Partial runs are reported explicitly.
Equal-shaped batches use the same seeded validation draws.

The single-file script remains available:

```bash
python src/pi_eval/property_inference.py reconstruction.pt --root-path /path/to/ETT-small --output report.md
```
