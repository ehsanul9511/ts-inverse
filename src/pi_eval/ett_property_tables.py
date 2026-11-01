#!/usr/bin/env python3
"""Compare elevated-load and coincident elevated-load fractions in saved tensors.

Requires numpy and torch; baseline additionally uses installed ts_inverse.datahandler.
Pass --root-path /path/to/ETT-small to enable 10 validation-batch draws.
Requires the same CSV and preprocessing used by the attack. Example:
  python ett_property_tables.py reconstruction.pt --thresholds 0 .5 1 1.5 2

Expected dictionary keys: original_data, reconstructed_data.
Accepts [B,T,C], [T,C], [num_batches,B,T,C], or lists of [B,T,C]
 tensors (including unequal batch sizes). Each window is processed separately.
All percentages are means of per-window fractions, p(X)=mean_i q(x_i).
No sample/channel matching, normalization, smoothing, or rescaling is performed.
Batch-level properties do not require corresponding sample order.
Thresholds are in saved tensor units, applied identically to both datasets.
Coincidence means at least k selected channels exceed the threshold at each
step. The particular elevated channels may change within a sustained run.
"""
import argparse
from pathlib import Path
import numpy as np

ETT_CHANNELS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']


def windows(value, name):
    import torch
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f'{name} is empty')
        return np.concatenate([windows(v, name) for v in value], axis=0)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f'{name} must contain tensors')
    a = value.detach().cpu().to(torch.float64).numpy()
    if a.ndim == 2:
        a = a[None]
    elif a.ndim == 4:
        a = a.reshape(-1, *a.shape[-2:])
    if a.ndim != 3 or any(s == 0 for s in a.shape):
        raise ValueError(f'{name}: expected nonempty [B,T,C], got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError(f'{name} contains NaN/Inf')
    return a


def run_fractions(mask, minimum):
    """[B,T] boolean mask -> [B] fraction in maximal runs of length >= minimum."""
    result = np.zeros(mask.shape[0], dtype=float)
    for i, row in enumerate(mask):
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        result[i] = lengths[lengths >= minimum].sum() / len(row)
    return result


def comparison(x, y, thresholds, duration, channels, k):
    result = []
    for threshold in thresholds:
        a = (x[:, :, channels] > threshold).sum(axis=-1) >= k
        b = (y[:, :, channels] > threshold).sum(axis=-1) >= k
        result.append((100 * run_fractions(a, duration).mean(),
                       100 * run_fractions(b, duration).mean()))
    return result


def table(title, rows, thresholds, error=False):
    lines = [title, '', '| Channel / condition | ' + ' | '.join(f'{t:g}' for t in thresholds) + ' |',
             '|---|' + '---:|' * len(thresholds)]
    for label, values in rows:
        cells = ([f'{abs(a-b):.2f}' for a, b in values] if error else
                 [f'{a:.2f} -> {b:.2f}' for a, b in values])
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def sample_validation_batches(args, shape):
    """Use project preprocessing, including the scaler fitted on training data."""
    from ts_inverse.datahandler import get_ett_dataset
    _, datasets, _ = get_ett_dataset(
        dataset=args.dataset, root_path=str(args.root_path),
        data_path=args.data_path, seq_len=shape[1],
        label_len=args.label_len, pred_len=args.pred_len,
        features=args.features, target=args.target, scale=args.scale,
        timeenc=0, freq='h' if args.dataset.startswith('ETTh') else 't',
    )
    # The ETT validation dataset includes seq_len training lookback rows.
    # Skip them so every baseline input lies wholly inside validation.
    candidates = [(d, i) for d in datasets
                  for i in range(shape[1], len(d), args.baseline_stride)]
    if len(candidates) < shape[0]:
        raise ValueError(f'Only {len(candidates)} eligible validation windows for '
                         f'a baseline batch of {shape[0]}. Reduce --baseline-stride.')
    rng = np.random.default_rng(args.baseline_seed)
    batches = []
    for _ in range(args.baseline_draws):
        chosen = rng.choice(len(candidates), size=shape[0], replace=False)
        batch = np.stack([np.asarray(candidates[j][0][candidates[j][1]][0],
                                    dtype=np.float64) for j in chosen])
        if batch.shape != shape or not np.isfinite(batch).all():
            raise ValueError(f'Validation input must be finite with shape {shape}; '
                             f'got {batch.shape}. Check loader feature selection.')
        batches.append(batch)
    return batches, len(candidates)


def baseline_table(x, y, batches, thresholds, duration, conditions):
    lines = ['### Comparison with random validation batches', '',
             '| Channel / condition | Threshold | Original % | Recon % | Random % (mean ± SD) | Recon error (pp) | Random error (mean ± SD, pp) | Error reduction % |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, channels, k in conditions:
        paired = np.asarray(comparison(x, y, thresholds, duration, channels, k))
        original, reconstructed = paired[:, 0], paired[:, 1]
        random_values = np.asarray([
            [v[1] for v in comparison(x, b, thresholds, duration, channels, k)]
            for b in batches])
        # Average absolute errors, NOT absolute error of the average prediction.
        errors = np.abs(random_values - original[None, :])
        for j, threshold in enumerate(thresholds):
            attack_error = abs(original[j] - reconstructed[j])
            baseline_error = errors[:, j].mean()
            gain = (f'{100 * (1 - attack_error / baseline_error):.2f}'
                    if baseline_error > 0 else 'N/A')
            lines.append(f'| {label} | {threshold:g} | {original[j]:.2f} | '
                         f'{reconstructed[j]:.2f} | '
                         f'{random_values[:, j].mean():.2f} ± {random_values[:, j].std(ddof=1):.2f} | '
                         f'{attack_error:.2f} | {baseline_error:.2f} ± {errors[:, j].std(ddof=1):.2f} | {gain} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filepath', type=Path)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0, .25, .5, .75, 1, 1.5, 2])
    parser.add_argument('--durations', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--channel-names', nargs='+', default=ETT_CHANNELS)
    parser.add_argument('--coincident-channels', nargs='+', default=['HUFL', 'MUFL', 'LUFL'])
    parser.add_argument('--min-channels', type=int, default=2)
    parser.add_argument('--output', type=Path, help='Optional Markdown report path')
    parser.add_argument('--root-path', type=Path, help='ETT CSV directory; enables validation baseline')
    parser.add_argument('--dataset', choices=['ETTh1','ETTh2','ETTm1','ETTm2'], default='ETTh1')
    parser.add_argument('--data-path', help='CSV filename, default <dataset>.csv')
    parser.add_argument('--label-len', type=int, default=48)
    parser.add_argument('--pred-len', type=int, default=96)
    parser.add_argument('--features', choices=['M','MS','S'], default='M')
    parser.add_argument('--target', default='OT')
    parser.add_argument('--scale', action=argparse.BooleanOptionalAction, default=True,
                        help='Must match attack preprocessing (default: train-fitted standardization)')
    parser.add_argument('--baseline-draws', type=int, default=10)
    parser.add_argument('--baseline-seed', type=int, default=43)
    parser.add_argument('--baseline-stride', type=int, default=1,
                        help='Candidate-window stride; use window length for nonoverlapping candidates')
    args = parser.parse_args()
    if args.baseline_draws < 10 or args.baseline_stride < 1:
        parser.error('Use at least 10 baseline draws and a positive stride')
    import torch
    data = torch.load(args.filepath, map_location='cpu', weights_only=True)
    x = windows(data['original_data'], 'original_data')
    y = windows(data['reconstructed_data'], 'reconstructed_data')
    if x.shape != y.shape:
        parser.error(f'Original/reconstruction shapes differ: {x.shape} vs {y.shape}')
    names = args.channel_names
    if len(names) != x.shape[-1] or len(set(names)) != len(names):
        parser.error('--channel-names must give one unique name per channel in stored order')
    if not all(np.isfinite(args.thresholds)):
        parser.error('Thresholds must be finite')
    if any(r < 1 or r > x.shape[1] for r in args.durations):
        parser.error('Durations must be between 1 and the window length')
    selected = args.coincident_channels
    if len(set(selected)) != len(selected) or any(c not in names for c in selected):
        parser.error('Coincident channels must be distinct members of --channel-names')
    if not 1 <= args.min_channels <= len(selected):
        parser.error('--min-channels must be between 1 and the number of selected channels')
    indices = [names.index(c) for c in selected]
    parts = [f'# Property comparison: {args.filepath.name}',
             f'Windows: {len(x)}; steps per window: {x.shape[1]}; channels: {x.shape[2]}.',
             'Cells: original % -> reconstructed %. Threshold comparison: strictly greater than.',
             'Thresholds use saved units. Durations use timesteps (hours only for hourly data).',
             'Fractions are computed per window, then averaged. Runs never cross window boundaries.',
             'Error tables show |p(original)-p(reconstructed)| in percentage points, not paired-window MAE.',
             'OT denotes elevated temperature, not load. Zero/zero cells may reflect no qualifying events.']
    baselines = None
    if args.root_path:
        baselines, pool_size = sample_validation_batches(args, x.shape)
        parts.extend([
            f'Baseline: {args.baseline_draws} independent draws, each with {len(x)} windows '
            f'sampled without replacement from {pool_size} validation candidates. Seed: {args.baseline_seed}.',
            f'Candidate stride: {args.baseline_stride}. Windows may overlap when stride < {x.shape[1]}; '
            'draws may reuse windows. Training lookback windows are excluded.',
            f'Loader: {args.dataset}, features={args.features}, scale={args.scale}, '
            f'label_len={args.label_len}, pred_len={args.pred_len}. '
            'These settings and CSV channel order must match the attack. '
            'This baseline assumes attacked windows came from training; '
            'the saved tensors contain no timestamps to exclude validation attacks.',
            'Random error = mean_draw |p(original)-p(random batch)|. '
            'Error reduction = 100*(1-recon_error/random_error); negative is worse, '
            'N/A means zero baseline error. SD is variation across draws, not a confidence interval.'
        ])
    for duration in args.durations:
        rows = [(name, comparison(x, y, args.thresholds, duration, [c], 1)) for c, name in enumerate(names)]
        parts.append(table(f'## Elevated-load operating fraction: minimum duration {duration}', rows, args.thresholds))
        parts.append(table('Absolute batch-property error (percentage points)', rows, args.thresholds, True))
        values = comparison(x, y, args.thresholds, duration, indices, args.min_channels)
        rows = [(f'At least {args.min_channels} of ' + ', '.join(selected), values)]
        parts.append(table(f'## Coincident elevated-load fraction: minimum duration {duration}', rows, args.thresholds))
        parts.append(table('Absolute batch-property error (percentage points)', rows, args.thresholds, True))
        if baselines is not None:
            conditions = [(name, [c], 1) for c, name in enumerate(names)]
            # Explicit joint events for the bottom baseline rows only.
            # Every listed channel must exceed the threshold at each step.
            joint_groups = [
                ('MUFL', 'HUFL'),
                ('HUFL', 'LUFL'),
                ('LUFL', 'MUFL'),
                ('MUFL', 'HUFL', 'LUFL'),
                ('MULL', 'HULL'),
                ('HULL', 'LULL'),
                ('LULL', 'MULL'),
                ('MULL', 'HULL', 'LULL'),
            ]
            for group in joint_groups:
                if any(channel not in names for channel in group):
                    parser.error('Joint baseline rows require channels: ' + ', '.join(group))
                conditions.append((' + '.join(group),
                                   [names.index(channel) for channel in group],
                                   len(group)))
            parts.append(baseline_table(x, y, baselines, args.thresholds, duration, conditions))
    report = '\n\n'.join(parts) + '\n' 
    print(report)
    if args.output:
        if args.output.resolve() == args.filepath.resolve():
            parser.error('--output must not overwrite the input file')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
