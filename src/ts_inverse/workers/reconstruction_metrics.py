"""Optional ETT reconstruction evaluators; no changes to the attack objective.

CLI: python -m ts_inverse.workers.reconstruction_metrics --help
TS2Vec source: https://github.com/zhihanyue/ts2vec (see vendor/LICENSE).

Baseline-relative evaluation (enabled by default with eval_ts2vec):
  eval_ts2vec_baseline: True
  ts2vec_baseline_samples: 32  # capped at the number of eligible windows
  ts2vec_baseline_seed: defaults to ts2vec_seed
  ts2vec_attack_split: 'train' # required; val/test attacks need time-aware exclusion

Baseline windows are sampled without replacement from a disjoint-window grid
in BOTH val and test, independent of ts2vec_reference_split. They exclude
preceding-partition lookback. Each channel uses the corresponding reference
channel; the joint encoder uses the same full multivariate windows. The pool
is fixed across calls/runs with the same data and baseline seed.

New keys under each ts2vec metric (joint, channel_N, channel_average):
  baseline_distance/mean, reconstruction_distance/mean, relative_gain/mean
  relative_gain_valid/mean, relative_gain_valid_count
G = 1 - mean(reconstruction distance) / mean(baseline distance).
Baseline expectation is estimated over the sampled reference windows for each
original. Aggregate means cover THIS evaluation batch (and channels for
channel_average), not all worker attack batches. Gains are not clipped:
1=perfect, 0=baseline-equivalent, negative=worse. Zero/nonfinite denominators
produce no gain value and are flagged; missing values must not be read as zero.
No checkpoint metadata or training behavior is changed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset


@contextmanager
def _isolated_rng(seed):
    """Preserve CPU, initialized CUDA, NumPy and Python random streams."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@contextmanager
def _checkpoint_lock(path):
    # Linux worker processes may simultaneously encounter a missing checkpoint.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic_save(payload, path):
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3 or min(value.shape) == 0:
        raise ValueError('Expected nonempty [batch, time, channels] arrays.')
    if not np.isfinite(value).all():
        raise ValueError('Reconstruction metric inputs contain NaN/Inf.')
    return value


def _feature_batch(x):
    import pycatch22
    features, names = [], None
    for window in x:
        channels = []
        for series in window.T:
            # Constant series have undefined normalized dynamics.
            if np.ptp(series) == 0:
                channels.append(np.full(22, np.nan))
                continue
            result = pycatch22.catch22_all(series.tolist(), catch24=False)
            if names is not None and names != result['names']:
                raise RuntimeError('catch22 feature ordering changed.')
            names = result['names']
            channels.append(result['values'])
        features.append(channels)
    if names is None:
        names = pycatch22.catch22_all(list(range(32)), catch24=False)['names']
    return np.asarray(features, dtype=np.float64), names


def evaluate_ts2vec(encoder, original, reconstructed):
    """Paired whole-window cosine/RMSE and aligned timestamp embedding RMSE."""
    x, y = _array(original), _array(reconstructed)
    if x.shape != y.shape:
        raise ValueError('Original and reconstruction shapes differ.')
    # encode handles eval/no_grad. One pass gives both temporal and max-pooled features.
    zx, zy = np.split(encoder.encode(np.concatenate([x, y])), 2, axis=0)
    if not np.isfinite(zx).all() or not np.isfinite(zy).all():
        raise RuntimeError('TS2Vec produced nonfinite embeddings.')
    a, b = zx.max(axis=1), zy.max(axis=1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = denom > 0
    cosine = np.full(len(x), np.nan)
    cosine[valid] = 1 - np.clip((a[valid] * b[valid]).sum(-1) / denom[valid], -1, 1)
    values = {
        'cosine_distance': cosine,
        'embedding_rmse': np.sqrt(np.mean((a - b) ** 2, axis=-1)),
        'temporal_embedding_rmse': np.sqrt(np.mean((zx - zy) ** 2, axis=(1, 2))),
    }
    metrics = {'ts2vec/cosine_valid_fraction': float(valid.mean())}
    for name, v in values.items():
        finite = np.isfinite(v)
        if finite.any():
            metrics[f'ts2vec/{name}/mean'] = float(v[finite].mean())
        metrics.update({f'ts2vec/{name}/{i}': float(v[i]) for i in np.flatnonzero(finite)})
    return metrics


def evaluate_ts2vec_channels(encoder, original, reconstructed):
    """Apply one shared univariate encoder to corresponding channels separately."""
    x, y = _array(original), _array(reconstructed)
    if x.shape != y.shape:
        raise ValueError('Original and reconstruction shapes differ.')
    metrics = {}
    collected = {}
    for channel in range(x.shape[2]):
        scores = evaluate_ts2vec(encoder, x[:, :, channel:channel + 1],
                                y[:, :, channel:channel + 1])
        for key, value in scores.items():
            suffix = key.removeprefix('ts2vec/')
            metrics[f'ts2vec/channel_{channel}/{suffix}'] = value
            # Average matching per-sample scores over channels, then over samples.
            if suffix.rsplit('/', 1)[-1].isdigit():
                collected.setdefault(suffix, []).append(value)
    by_metric = {}
    for suffix, values in collected.items():
        name, sample = suffix.rsplit('/', 1)
        metrics[f'ts2vec/channel_average/{suffix}'] = float(np.mean(values))
        metrics[f'ts2vec/channel_average/{name}/valid_channels/{sample}'] = len(values)
        by_metric.setdefault(name, []).append(float(np.mean(values)))
    for name, values in by_metric.items():
        metrics[f'ts2vec/channel_average/{name}/mean'] = float(np.mean(values))
    return metrics


def evaluate_catch22(original, reconstructed, reference_std, reference_names):
    """Raw and reference-standardized errors; undefined values are counted, not zeroed."""
    x, y = _array(original), _array(reconstructed)
    if x.shape != y.shape:
        raise ValueError('Original and reconstruction shapes differ.')
    a, names = _feature_batch(x)
    b, other_names = _feature_batch(y)
    if names != other_names or names != list(reference_names):
        raise ValueError('catch22 feature names differ from calibration.')
    scale = np.asarray(reference_std)
    if scale.shape != a.shape[1:]:
        raise ValueError('catch22 reference channel/feature shape mismatch.')
    valid = np.isfinite(a) & np.isfinite(b)
    eligible = np.isfinite(scale) & (scale > 0)
    scaled_valid = valid & eligible[None]
    error = np.abs(a - b)
    standardized = np.full_like(error, np.nan)
    np.divide(error, scale[None], out=standardized, where=scaled_valid)
    metrics = {
        'catch22/valid_fraction': float(valid.mean()),
        'catch22/reference_eligible_fraction': float(eligible.mean()),
        'catch22/standardized_valid_fraction': float(scaled_valid.mean()),
        'catch22/invalid_count': int((~valid).sum()),
    }
    for k, name in enumerate(names):
        mask = valid[:, :, k]
        metrics[f'catch22/{name}/valid_count'] = int(mask.sum())
        if mask.any():
            metrics[f'catch22/{name}/mae'] = float(error[:, :, k][mask].mean())
        smask = scaled_valid[:, :, k]
        if smask.any():
            metrics[f'catch22/{name}/standardized_mae'] = float(standardized[:, :, k][smask].mean())
        for c in range(x.shape[2]):
            mask = valid[:, c, k]
            if mask.any():
                metrics[f'catch22/channel_{c}/{name}/mae'] = float(error[:, c, k][mask].mean())
    if scaled_valid.any():
        metrics['catch22/standardized_mae/mean'] = float(standardized[scaled_valid].mean())
    for i in range(len(x)):
        if scaled_valid[i].any():
            metrics[f'catch22/standardized_mae/{i}'] = float(standardized[i][scaled_valid[i]].mean())
    return metrics


class ETTReconstructionEvaluator:
    def __init__(self, config, val_datasets, test_datasets):
        self.config = config
        self.use_ts2vec = config.get('eval_ts2vec', False)
        self.use_catch22 = config.get('eval_catch22', False)
        self.encoder = None
        self.channel_encoder = None
        self.calibration = None
        # Baseline inputs are always taken from BOTH validation and test.
        # Nonoverlap with attacked inputs assumes the worker attacks the train split.
        self.baseline_datasets = list(val_datasets) + list(test_datasets)
        self.use_baseline = self.use_ts2vec and config.get('eval_ts2vec_baseline', True)
        self.baseline_windows = None
        self._baseline_cache_key = None
        self._baseline_cache = None
        if self.use_baseline and config.get('ts2vec_attack_split', 'train') != 'train':
            raise ValueError('Baseline nonoverlap currently requires attacked TRAIN windows. '
                             'Other attack splits need timestamp-aware exclusion.')
        selected = config.get('ts2vec_reference_split', 'val_test')
        if selected not in {'val', 'test', 'val_test'}:
            raise ValueError('ts2vec_reference_split must be val, test or val_test.')
        self.datasets = ((list(val_datasets) if selected in {'val', 'val_test'} else []) +
                         (list(test_datasets) if selected in {'test', 'val_test'} else []))
        self.reference = None
        self.seed = int(config.get('ts2vec_seed', 43))
        d = self.datasets[0]
        self.channels = int(d[0][0].shape[-1])
        self.length = int(d.seq_len)
        self.arch = dict(input_dims=self.channels,
                         output_dims=int(config.get('ts2vec_output_dims', 320)),
                         hidden_dims=int(config.get('ts2vec_hidden_dims', 64)),
                         depth=int(config.get('ts2vec_depth', 10)))
        # Use actual dataset fields: merged worker config may overwrite features with [0].
        self.identity = dict(dataset=config['dataset'], reference_split=selected,
            channels=self.channels, seq_len=self.length, label_len=int(d.label_len),
            pred_len=int(d.pred_len), features=d.features, scale=bool(d.scale), target=d.target,
            root_path=str(Path(d.root_path).resolve()), data_path=str(d.data_path),
            lengths=[len(ds) for ds in self.datasets], boundary_policy='skip_seq_len_v1',
            stride=int(config.get('ts2vec_reference_stride', 1)))
        if self.identity['stride'] < 1:
            raise ValueError('ts2vec_reference_stride must be positive.')
        source = Path(d.root_path) / d.data_path
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        self.identity['source_sha256'] = digest.hexdigest()
        self.path = Path(config.get('ts2vec_checkpoint',
            f"/scratch/ejk5818/ts-inverse/data/_evaluation/ts2vec_{config['dataset']}"))
        if self.use_ts2vec:
            self.encoder = self._load_or_train(self.path, self.arch, self.identity)
            if config.get('eval_ts2vec_per_channel', True):
                channel_path = Path(config.get('ts2vec_channel_checkpoint',
                                               str(self.path) + '.univariate.pt'))
                if channel_path.resolve() == self.path.resolve():
                    raise ValueError('Joint and univariate checkpoints must have different paths.')
                self.channel_encoder = self._load_or_train(
                    channel_path, {**self.arch, 'input_dims': 1},
                    {**self.identity, 'representation': 'shared_univariate_v1'},
                    univariate=True)
        if self.use_catch22:
            self._load_or_calibrate()
        self.reference = None  # release windows after setup

    def _reference_windows(self):
        if self.reference is None:
            subsets = [Subset(d, range(d.seq_len, len(d), self.identity['stride']))
                       for d in self.datasets]
            loader = DataLoader(ConcatDataset(subsets), batch_size=256, shuffle=False,
                                generator=torch.Generator().manual_seed(self.seed))
            batches = [batch[0].float().numpy() for batch in loader]
            if not batches:
                raise ValueError('No ETT reference windows after boundary exclusion.')
            self.reference = _array(np.concatenate(batches))
        return self.reference

    def _get_baseline_windows(self):
        """Fixed random sample of disjoint input windows wholly inside val/test.

        Training-partition attacked inputs are disjoint from this pool. Dataset
        indices start at seq_len to exclude previous-partition lookback history.
        Baselines are sampled without inspecting original/reconstruction values.
        """
        if self.baseline_windows is None:
            candidates = []
            seen_splits = set()
            for d in self.baseline_datasets:
                if d.set_type not in (1, 2):
                    raise ValueError('Baseline datasets must be ETT validation/test splits.')
                split_key = (str(Path(d.root_path).resolve()), str(d.data_path), d.set_type)
                if split_key in seen_splits:
                    continue
                seen_splits.add(split_key)
                if d.seq_len != self.length or d.features != self.datasets[0].features:
                    raise ValueError('Baseline and encoder dataset shapes/features differ.')
                candidates.extend((d, i) for i in range(d.seq_len, len(d), d.seq_len))
            requested = int(self.config.get('ts2vec_baseline_samples', 32))
            if requested < 1 or not candidates:
                raise ValueError('Need positive ts2vec_baseline_samples and eligible baseline windows.')
            rng = np.random.default_rng(int(self.config.get('ts2vec_baseline_seed', self.seed)))
            indices = rng.choice(len(candidates), size=min(requested, len(candidates)), replace=False)
            self.baseline_windows = _array(np.stack([
                candidates[int(j)][0][candidates[int(j)][1]][0] for j in indices
            ]))
            if self.baseline_windows.shape[1:] != (self.length, self.channels):
                raise ValueError('Baseline shape differs from evaluated inputs.')
        return self.baseline_windows

    def _baseline_scores(self, x):
        # Cache the last original batch only; reconstruction changes do not invalidate it.
        key = (x.shape, hashlib.sha256(x.tobytes()).digest())
        if key != self._baseline_cache_key:
            baselines = self._get_baseline_windows()
            result = {}
            families = [('ts2vec', self.encoder, None)]
            if self.channel_encoder is not None:
                families += [(f'ts2vec/channel_{c}', self.channel_encoder, c)
                             for c in range(self.channels)]
            names = ('cosine_distance', 'embedding_rmse', 'temporal_embedding_rmse')
            for family, encoder, channel in families:
                for i in range(len(x)):
                    original = np.repeat(x[i:i + 1], len(baselines), axis=0)
                    reference = baselines
                    if channel is not None:
                        original = original[:, :, channel:channel + 1]
                        reference = reference[:, :, channel:channel + 1]
                    scores = evaluate_ts2vec(encoder, original, reference)
                    for name in names:
                        values = [scores[f'ts2vec/{name}/{j}'] for j in range(len(baselines))
                                  if f'ts2vec/{name}/{j}' in scores]
                        result[(family, name, i)] = (
                            float(np.mean(values)) if values else np.nan, len(values))
            self._baseline_cache_key, self._baseline_cache = key, result
        return self._baseline_cache

    def _relative_baseline_metrics(self, x, reconstruction_scores):
        """Ratio of mean distances, NOT mean of sample/channel ratios.

        G=1-mean_i d(x_i,reconstruction_i)/mean_i mean_b d(x_i,b).
        Each channel is compared with that SAME channel of reference windows.
        channel_average pools valid sample/channel entries before taking a ratio.
        These means describe this evaluate() batch, not all attack batches.
        """
        baseline = self._baseline_scores(x)
        result = {'ts2vec/baseline/sample_count': len(self.baseline_windows)}
        groups = {}
        for (family, name, i), (denominator, count) in baseline.items():
            root = f'{family}/{name}'
            result[f'{root}/baseline_valid_count/{i}'] = count
            if np.isfinite(denominator):
                result[f'{root}/baseline_distance/{i}'] = denominator
            numerator = reconstruction_scores.get(f'{root}/{i}', np.nan)
            valid = np.isfinite(numerator) and np.isfinite(denominator)
            if valid:
                groups.setdefault(root, []).append((numerator, denominator))
                if family.startswith('ts2vec/channel_'):
                    groups.setdefault(f'ts2vec/channel_average/{name}', []).append((numerator, denominator))
                if denominator > 0:
                    result[f'{root}/relative_gain/{i}'] = 1 - numerator / denominator
            result[f'{root}/relative_gain_valid/{i}'] = int(valid and denominator > 0)
        families = ['ts2vec']
        if self.channel_encoder is not None:
            families += [f'ts2vec/channel_{c}' for c in range(self.channels)] + ['ts2vec/channel_average']
        for family in families:
            for name in ('cosine_distance', 'embedding_rmse', 'temporal_embedding_rmse'):
                root = f'{family}/{name}'
                pairs = groups.get(root, [])
                result[f'{root}/relative_gain_valid_count'] = len(pairs)
                if pairs:
                    numerator, denominator = np.asarray(pairs).mean(axis=0)
                    result[f'{root}/baseline_distance/mean'] = float(denominator)
                    result[f'{root}/reconstruction_distance/mean'] = float(numerator)
                    result[f'{root}/relative_gain_valid/mean'] = int(denominator > 0)
                    if denominator > 0:
                        result[f'{root}/relative_gain/mean'] = float(1 - numerator / denominator)
                else:
                    result[f'{root}/relative_gain_valid/mean'] = 0
        return result

    def _load_or_train(self, path, architecture, identity, univariate=False):
        from .ts2vec_vendor import TS2Vec
        device = self.config.get('ts2vec_device', 'cpu')
        if isinstance(device, int):
            device = f'cuda:{device}'
        if torch.device(device).type == 'cuda':
            torch.cuda.init()
        # Training may be invoked from worker evaluation's torch.no_grad block.
        with _checkpoint_lock(path), _isolated_rng(self.seed), torch.enable_grad():
            encoder = TS2Vec(**architecture, device=device,
                batch_size=int(self.config.get('ts2vec_batch_size', 16)),
                lr=float(self.config.get('ts2vec_learning_rate', 0.001)))
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved['identity'] != identity or saved['architecture'] != architecture:
                    raise ValueError(f'Incompatible evaluator checkpoint {path}. Use a different path.')
                encoder.net.load_state_dict(saved['state_dict'])
                print(f'Loaded TS2Vec evaluator: {path}')
            else:
                epochs = int(self.config.get('ts2vec_train_epochs', 20))
                if epochs < 1:
                    raise ValueError('ts2vec_train_epochs must be positive when checkpoint is missing.')
                windows = self._reference_windows()
                if univariate:
                    windows = windows.transpose(0, 2, 1).reshape(-1, self.length, 1).copy()
                if len(windows) < encoder.batch_size:
                    raise ValueError('Reference set smaller than TS2Vec batch size.')
                print(f'Training TS2Vec on {len(windows)} {identity["reference_split"]} windows.')
                encoder.fit(windows, n_epochs=epochs, verbose=True)
                _atomic_save(dict(identity=identity, architecture=architecture,
                    state_dict={k: v.detach().cpu() for k, v in encoder.net.state_dict().items()},
                    seed=self.seed, epochs=epochs), path)
                print(f'Saved TS2Vec evaluator: {path}')
            encoder.net.eval()
            encoder.net.requires_grad_(False)

        return encoder

    def _load_or_calibrate(self):
        version = importlib.metadata.version('pycatch22')
        path = Path(self.config.get('catch22_reference_path', str(self.path) + '.catch22.pt'))
        with _checkpoint_lock(path):
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved['identity'] != self.identity or saved['version'] != version:
                    raise ValueError(f'Incompatible catch22 calibration {path}. Use a different path.')
            else:
                print('Calibrating catch22 on ETT reference windows.')
                values, names = _feature_batch(self._reference_windows())
                finite = np.isfinite(values)
                count = finite.sum(0)
                mean = np.divide(np.where(finite, values, 0).sum(0), count,
                                 out=np.zeros(count.shape), where=count > 0)
                variance = np.divide(np.where(finite, (values - mean) ** 2, 0).sum(0), count,
                                     out=np.full(count.shape, np.nan), where=count >= 2)
                saved = dict(identity=self.identity, version=version, names=names,
                             std=torch.from_numpy(np.sqrt(variance)), count=torch.from_numpy(count))
                _atomic_save(saved, path)
            self.calibration = saved

    def evaluate(self, original, reconstructed, prefix='inputs'):
        x, y = _array(original), _array(reconstructed)
        if x.shape != y.shape or x.shape[1:] != (self.length, self.channels):
            raise ValueError('Evaluator expects paired windows with its trained length and channels.')
        metrics = {}
        # encode's DataLoader otherwise consumes the global torch RNG even without shuffle.
        with _isolated_rng(self.seed):
            if self.use_ts2vec:
                metrics.update(evaluate_ts2vec(self.encoder, x, y))
                if self.channel_encoder is not None:
                    metrics.update(evaluate_ts2vec_channels(self.channel_encoder, x, y))
                if self.use_baseline:
                    metrics.update(self._relative_baseline_metrics(x, metrics))
            if self.use_catch22:
                metrics.update(evaluate_catch22(x, y, self.calibration['std'].numpy(),
                                               self.calibration['names']))
        return {f'{prefix}/{key}': value for key, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description='Prepare/load ETT TS2Vec evaluation checkpoint.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--dataset', default='ETTh1', choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'])
    parser.add_argument('--data-path')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--channel-checkpoint')
    parser.add_argument('--no-per-channel', action='store_true')
    parser.add_argument('--reference-split', default='val_test', choices=['val', 'test', 'val_test'])
    parser.add_argument('--features', default='M', choices=['M', 'MS', 'S'])
    parser.add_argument('--target', default='OT')
    parser.add_argument('--seq-len', type=int, default=96)
    parser.add_argument('--label-len', type=int, default=48)
    parser.add_argument('--pred-len', type=int, default=96)
    parser.add_argument('--timeenc', type=int, default=0)
    parser.add_argument('--freq', default='h')
    parser.add_argument('--no-scale', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--reference-stride', type=int, default=1)
    parser.add_argument('--catch22', action='store_true')
    args = parser.parse_args()
    from ..datahandler import get_ett_dataset
    _, val, test = get_ett_dataset(dataset=args.dataset, root_path=args.root_path,
        data_path=args.data_path, seq_len=args.seq_len, label_len=args.label_len,
        pred_len=args.pred_len, features=args.features, target=args.target,
        scale=not args.no_scale, timeenc=args.timeenc, freq=args.freq)
    config = dict(dataset=args.dataset, eval_ts2vec=True, eval_catch22=args.catch22,
        ts2vec_checkpoint=args.checkpoint, ts2vec_reference_split=args.reference_split,
        ts2vec_device=args.device, ts2vec_train_epochs=args.epochs,
        ts2vec_batch_size=args.batch_size, ts2vec_seed=args.seed,
        ts2vec_reference_stride=args.reference_stride)
    config['eval_ts2vec_per_channel'] = not args.no_per_channel
    if args.channel_checkpoint:
        config['ts2vec_channel_checkpoint'] = args.channel_checkpoint
    ETTReconstructionEvaluator(config, val, test)


if __name__ == '__main__':
    main()
