#!/usr/bin/env python3
"""Evaluate every batch in one reconstruction run and write a Markdown report.

Examples (from the repository root):
  python src/pi_eval/evaluate_reconstruction_run.py reconstructions/ETTh1_<timestamp>
  python src/pi_eval/evaluate_reconstruction_run.py

With no folder argument, select the latest timestamped folder in reconstructions/.
The validation baseline uses the dataset and preprocessing in that folder's config.json.
Use --root-path to relocate the ETT CSV directory without changing preprocessing.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

# Support both direct script execution and python -m pi_eval.evaluate_reconstruction_run.
SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_eval.property_inference import (
    ETT_CHANNELS, baseline_results, comparison, format_baseline_table,
    sample_validation_batches, table, windows,
)

JOINT_GROUPS = [
    ('MUFL', 'HUFL'), ('HUFL', 'LUFL'), ('LUFL', 'MUFL'), ('MUFL', 'HUFL', 'LUFL'),
    ('MULL', 'HULL'), ('HULL', 'LULL'), ('LULL', 'MULL'), ('MULL', 'HULL', 'LULL'),
]


def folder_timestamp(path):
    match = re.search(r'_(\d{8}_\d{6}(?:_\d{1,6})?(?:[+-]\d{4})?)$', path.name)
    if match is None:
        return None
    for fmt in ('%Y%m%d_%H%M%S_%f%z', '%Y%m%d_%H%M%S%z',
                '%Y%m%d_%H%M%S_%f', '%Y%m%d_%H%M%S'):
        try:
            timestamp = datetime.strptime(match[1], fmt)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=ZoneInfo('America/New_York'))
            return timestamp.timestamp()
        except ValueError:
            continue
    return None


def latest_run(root):
    if not root.is_dir():
        raise ValueError(f'Reconstructions directory does not exist: {root}')
    candidates = [(folder_timestamp(p), p) for p in root.iterdir() if p.is_dir()]
    candidates = [(stamp, p) for stamp, p in candidates if stamp is not None]
    if not candidates:
        raise ValueError(f'No timestamped reconstruction folders found in {root}')
    return max(candidates, key=lambda item: (item[0], item[1].name))[1]


def batch_files(folder):
    # Current saver: batch_0.pt. Also accept the earlier ..._batch_0_seed_*.pt names.
    found = []
    for path in folder.glob('*.pt'):
        match = re.search(r'(?:^|_)batch_(\d+)(?:_|\.pt$)', path.name)
        if match:
            found.append((int(match[1]), path))
    if not found:
        raise ValueError(f'No batch .pt files found in {folder}')
    indices = [index for index, _ in found]
    if len(set(indices)) != len(indices):
        raise ValueError('Multiple files have the same batch index; select one file per batch before evaluation')
    return sorted(found)


def baseline_settings(config, args, channel_count):
    dataset = config.get('dataset')
    if dataset not in {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'}:
        raise ValueError(f'Property inference currently supports ETT datasets; got {dataset!r}')
    root = args.root_path or config.get('root_path')
    if not root:
        raise ValueError('config.json must provide root_path, or pass --root-path for the validation baseline')
    features = args.features or config.get('features', 'M')
    if isinstance(features, list):
        # Older configs overwrite the ETT mode with model feature indices.
        # M and MS both load all seven input channels; S loads only the target.
        if channel_count not in (1, 7):
            raise ValueError('Cannot infer ETT input mode from saved channels; pass --features')
        features = 'S' if channel_count == 1 else 'M'
    if features not in ('M', 'MS', 'S'):
        raise ValueError(f'Invalid ETT features mode: {features!r}; pass --features')
    return SimpleNamespace(
        dataset=dataset, root_path=Path(root), data_path=config.get('data_path'),
        label_len=config.get('label_len', 48), pred_len=config.get('pred_len', 96),
        features=features, target=config.get('target', 'OT'),
        scale=config.get('scale', True), baseline_draws=args.baseline_draws,
        baseline_seed=args.baseline_seed, baseline_stride=args.baseline_stride,
    )


def evaluate_run(folder, args):
    import torch
    with (folder / 'config.json').open(encoding='utf-8') as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError('config.json must contain a dictionary')
    files = batch_files(folder)
    first_data = torch.load(files[0][1], map_location='cpu', weights_only=True)
    first_shape = windows(first_data['original_data'], 'original_data').shape
    settings = baseline_settings(config, args, first_shape[-1])
    names = args.channel_names or ([settings.target] if settings.features == 'S' else ETT_CHANNELS)
    if len(set(names)) != len(names):
        raise ValueError('Channel names must be unique and in saved tensor order')
    conditions = [(name, [i], 1) for i, name in enumerate(names)]
    conditions += [(' + '.join(group), [names.index(c) for c in group], len(group))
                   for group in JOINT_GROUPS if all(c in names for c in group)]
    combined_rows, details = {}, []
    # Equal-shaped batches use identical random validation draws for comparability.
    baseline_cache = {}
    for batch_index, path in files:
        print(f'Evaluating batch {batch_index}: {path.name}', flush=True)
        data = first_data if path == files[0][1] else torch.load(path, map_location='cpu', weights_only=True)
        x = windows(data['original_data'], f'{path.name}: original_data')
        y = windows(data['reconstructed_data'], f'{path.name}: reconstructed_data')
        if x.shape != y.shape:
            raise ValueError(f'{path.name}: original/reconstruction shapes differ: {x.shape} vs {y.shape}')
        if x.shape[-1] != len(names):
            raise ValueError(f'{path.name}: expected {len(names)} channels, got {x.shape[-1]}; check --channel-names')
        if any(duration < 1 or duration > x.shape[1] for duration in args.durations):
            raise ValueError(f'{path.name}: durations must be between 1 and window length {x.shape[1]}')
        if 'seq_len' in config and config['seq_len'] != x.shape[1]:
            raise ValueError(f'{path.name}: window length does not match config seq_len')
        if x.shape not in baseline_cache:
            baseline_cache[x.shape] = sample_validation_batches(settings, x.shape)
        baselines, pool_size = baseline_cache[x.shape]
        parts = [f'## Batch {batch_index}: {path.name}',
                 f'Windows: {len(x)}; timesteps: {x.shape[1]}; channels: {x.shape[2]}. '
                 f'Validation candidate windows: {pool_size}.']
        for duration in args.durations:
            rows = [(name, comparison(x, y, args.thresholds, duration, [i], 1))
                    for i, name in enumerate(names)]
            parts.append(table(f'### Elevated fraction: minimum duration {duration}', rows, args.thresholds))
            parts.append(table('Absolute batch-property error (percentage points)', rows, args.thresholds, True))
            selected = ['HUFL', 'MUFL', 'LUFL']
            if all(c in names for c in selected):
                coincident = comparison(x, y, args.thresholds, duration, [names.index(c) for c in selected], 2)
                parts.append(table('### Coincident elevated fraction: at least 2 of HUFL, MUFL, LUFL',
                                   [('At least 2 channels', coincident)], args.thresholds))
            metrics = baseline_results(x, y, baselines, args.thresholds, duration, conditions)
            parts.append(format_baseline_table(metrics))
            for row in metrics:
                key = (duration, row['condition'], row['threshold'])
                combined_rows.setdefault(key, []).append(row)
        details.append('\n\n'.join(parts))

    summary = [
        '## Combined performance by property', '',
        'Each row averages the same condition, threshold, and duration across batches. '
        'All batches have equal weight.', '',
        '| Duration | Channel / condition | Threshold | Mean original % | Mean recon % | Mean random % | Mean recon error (pp) | Mean random error (pp) | Mean error reduction % | Batches with defined reduction |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for (duration, condition, threshold), rows in combined_rows.items():
        means = [np.mean([row[field] for row in rows]) for field in
                 ('original', 'reconstructed', 'random_mean', 'attack_error', 'baseline_error')]
        gains = [row['error_reduction'] for row in rows if row['error_reduction'] is not None]
        gain = f'{np.mean(gains):.2f}' if gains else 'N/A'
        cells = ' | '.join(f'{mean:.2f}' for mean in means)
        summary.append(f'| {duration} | {condition} | {threshold:g} | {cells} | {gain} | {len(gains)}/{len(rows)} |')
    evaluation_config = dict(vars(settings), thresholds=args.thresholds, durations=args.durations,
                             channel_names=names, conditions=[label for label, _, _ in conditions])
    warnings = []
    if isinstance(config.get('features'), list) and args.features is None:
        warnings.append(
            f'The saved features field contains model indices, so baseline input mode '
            f'`{settings.features}` was inferred from the stored channel count. '
            'ETT M and MS have identical input preprocessing. The original config is preserved below.'
        )
    expected = config.get('attack_number_of_batches')
    if expected is not None and [i for i, _ in files] != list(range(expected)):
        warnings.append(f'**Partial run:** config requests {expected} batches; found indices {[i for i, _ in files]}. '
                        'Combined results cover only the files present.')
    report = [
        f'# Property inference evaluation: {folder.name}',
        f'Input folder: `{folder.resolve()}`',
        *warnings,
        '\n'.join(summary),
        '## Metric definitions',
        'Error reduction = 100 × (1 − reconstruction error / mean random validation error). '
        'Errors are absolute differences of batch-level property percentages, in percentage points. '
        'Negative reduction means reconstruction is worse than the random baseline.',
        'Combined rows preserve each condition, threshold, and duration separately. Each numeric column '
        'is an unweighted average across batches, using unrounded values (not weighted by window count). '
        'Mean error reduction is the average of the individual batches\' error-reduction percentages for '
        'that exact row, not a ratio of the averaged errors. Zero baseline error gives N/A; it is excluded '
        'only from that row\'s error-reduction average. The contributing batch count is shown for every row. '
        'Other columns include all batches. The separate 2-of-3 coincidence table remains descriptive, '
        'matching the standalone script.',
        'Thresholds are strictly greater-than comparisons in saved tensor units. Durations are timesteps. '
        'Fractions are computed per window and averaged; runs never cross window boundaries. '
        'OT measures elevated temperature, not load. No normalization or sample matching is applied.',
        f'Baseline: {args.baseline_draws} draws, each sampled without replacement from validation; '
        f'seed {args.baseline_seed}, stride {args.baseline_stride}. Equal-shaped batches reuse the same draws. '
        'Draws may reuse windows, and windows can overlap. Training-lookback windows are excluded. '
        'This assumes attacks used training windows; saved tensors do not provide timestamps to verify that. '
        'SD describes variation across draws, not a confidence interval.',
        '## Run configuration',
        'Complete saved configuration (including dataset, model, attack, defense, and preprocessing parameters):',
        '```json\n' + json.dumps(config, indent=2, ensure_ascii=False) + '\n```',
        '## Effective evaluation settings',
        '```json\n' + json.dumps(evaluation_config, indent=2, default=str) + '\n```',
        *details,
    ]
    return '\n\n'.join(report) + '\n'


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', nargs='?', type=Path, help='Run folder; defaults to the latest timestamped run')
    parser.add_argument('--reconstructions-dir', type=Path, default=REPO_ROOT / 'reconstructions')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'evaluation_results')
    parser.add_argument('--root-path', type=Path, help='Override the ETT CSV directory saved in config.json')
    parser.add_argument('--features', choices=['M', 'MS', 'S'], help='Override the saved ETT feature mode')
    parser.add_argument('--channel-names', nargs='+', help='Override channel names in saved tensor order')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0, .25, .5, .75, 1, 1.5, 2])
    parser.add_argument('--durations', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--baseline-draws', type=int, default=10)
    parser.add_argument('--baseline-seed', type=int, default=43)
    parser.add_argument('--baseline-stride', type=int, default=1)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.baseline_draws < 10 or args.baseline_stride < 1:
        parser.error('Use at least 10 baseline draws and a positive stride')
    if not all(np.isfinite(args.thresholds)):
        parser.error('Thresholds must be finite')
    if any(duration < 1 for duration in args.durations):
        parser.error('Durations must be positive')
    try:
        folder = args.folder.resolve() if args.folder else latest_run(args.reconstructions_dir)
        report = evaluate_run(folder, args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / f'{folder.name}_property_inference.md'
        output.write_text(report, encoding='utf-8')
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f'Saved evaluation report: {output}')
    return output


if __name__ == '__main__':
    main()
