import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from ts_inverse.workers.attack_ts_inverse_worker import AttackTSInverseWorker
from pi_eval.property_inference import run_fractions


class AuxiliaryDataset(Dataset):
    def __init__(self, root, features='M', seq_len=6, label_len=2, pred_len=2, channels=3, count=100):
        self.root_path, self.data_path = str(root), 'ETTh1.csv'
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.features, self.target = features, 'OT'
        self.scale, self.timeenc, self.freq = True, 0, 'h'
        self.scaler = SimpleNamespace(mean_=np.zeros(channels), scale_=np.ones(channels))
        self.data = torch.sin(torch.arange(count * channels).float().reshape(count, channels) / 3)
        self.marks = torch.zeros(count, 4)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        end = index + self.seq_len
        return (self.data[index:end], self.data[end-self.label_len:end+self.pred_len],
                self.marks[index:end], self.marks[end-self.label_len:end+self.pred_len])


class TinyForecast(torch.nn.Module):
    name = 'TinyForecast'

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.decoder = torch.nn.Module()
        self.decoder.projection = torch.nn.Linear(4, 3)
        self.calls = []
        self.output_attention = False
        self.invalid_output = False

    def forward(self, x, x_mark, decoder, y_mark):
        self.calls.append((self.training, self.decoder.projection.weight.requires_grad))
        hidden = torch.tanh(self.encoder(x[:, -2:] + decoder[:, :2]))
        out = self.decoder.projection(hidden)
        if self.invalid_output:
            out = out + 1
        return (out, None) if self.output_attention else out


class PoisonedGlobalModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'ETTh1.csv').write_text('date,HUFL,MUFL,OT\n2020-01-01,1,2,3\n')
        self.worker = AttackTSInverseWorker(0)
        self.dataset = AuxiliaryDataset(self.root)
        self.worker.train_datasets = [self.dataset]
        self.worker.test_datasets = [self.dataset]
        self.worker.val_datasets = [self.dataset]
        self.config = dict(
            dataset='ETTh1', device='cpu', seed=43, poison_global_model=True, poison_strategy='constant_bias',
            poison_feasibility_only=True, poison_target_channel='HUFL', poison_measurement_channel='MUFL',
            poison_feature_epochs=1, poison_train_epochs=1, poison_train_batch_size=8,
            poison_train_learning_rate=.01, poison_measurement_C=10., poison_hidden_coordinate=1,
            poison_property_threshold=0., poison_property_duration=2,
            poison_validation_batch_sizes=[1, 3, 8], poison_validation_batches=2,
            poison_checkpoint_path=str(self.root / 'poison.pt'), poison_feasibility_output_dir=str(self.root),
        )

    def prepare(self, model):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.worker.load_or_train_poisoned_global_model(model, self.config)

    def test_architecture_fixed_row_holdout_and_actual_gradient_identity(self):
        model = TinyForecast()
        original_names = [name for name, _ in model.named_parameters()]
        ids = [id(p) for p in model.parameters()]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        data = self.dataset.data.clone()
        rng = torch.random.get_rng_state().clone()
        self.assertIs(self.prepare(model), model)
        self.assertEqual([name for name, _ in model.named_parameters()], original_names)
        self.assertEqual([id(p) for p in model.parameters()], ids)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.testing.assert_close(self.dataset.data, data)
        self.assertTrue(all(not training for training, _ in model.calls))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.decoder.projection._forward_pre_hooks)
        self.assertEqual(torch.count_nonzero(model.decoder.projection.weight[1]).item(), 0)
        self.assertEqual(model.decoder.projection.bias[1].item(), 10)
        torch.testing.assert_close(model.decoder.projection.weight[[0, 2]], before['decoder.projection.weight'][[0, 2]])
        self.assertFalse(torch.equal(model.encoder.weight, before['encoder.weight']))
        report = self.worker.poison_feasibility_results
        for part in report['design']['partitions']:
            self.assertLessEqual(part['preparation_stop'] - 1 + part['window_span'], part['holdout_start'])
        self.assertEqual([r['batch_size'] for r in report['results']], [1, 3, 8])
        for result in report['results']:
            self.assertLess(result['max_identity_error'], 1e-5)
            for row in result['rows']:
                self.assertAlmostEqual(row['estimate'], row['feature_average'] - row['target_correction'], places=5)
        self.assertEqual(self.config['poison_gradient_parameter_name'], 'decoder.projection.weight')
        self.assertEqual(self.config['poison_gradient_coordinate'], [1, 1])
        self.assertEqual(self.config['num_learn_epochs'], 0)
        self.assertFalse(self.config['load_lti_model'])
        self.assertTrue((self.root / 'poison_feasibility.md').exists())
        markdown = (self.root / 'poison_feasibility.md').read_text()
        self.assertLess(markdown.index('## Held-out batches'), markdown.index('## Aggregate diagnostics'))
        for result in report['results']:
            self.assertIn(f"### Batch size {result['batch_size']}: {len(result['rows'])} held-out batches", markdown)
            for row in result['rows']:
                self.assertIn(f"| {row['batch'] + 1} | {row['true_property']:.6f} | {row['estimate']:.6f} "
                              f"| {row['estimate'] - row['true_property']:+.6f} |", markdown)

    def test_short_runs_excluded_and_scalar_identity_multiple_sizes(self):
        # An isolated high sample must not count; the maximal run of 3 does.
        values = torch.tensor([[1., 0., 1., 1., 1., 0.], [1., 0., 1., 0., 1., 0.], [1., 1., 1., 1., 1., 1.]])
        phi = torch.tensor(run_fractions(values.numpy() > 0, 2))
        torch.testing.assert_close(phi, torch.tensor([.5, 0., 1.], dtype=phi.dtype))
        for size in [1, 3, 8]:
            with self.subTest(batch_size=size):
                projection = torch.nn.Linear(4, 3).double()
                with torch.no_grad():
                    projection.weight[1].zero_()
                    projection.bias[1].fill_(10)
                h = torch.randn(size, 2, 4, dtype=torch.float64)
                y = torch.randn(size, 2, 3, dtype=torch.float64)
                torch.nn.functional.mse_loss(projection(h), y).backward()
                decoded = 3 * projection.weight.grad[1, 1] / 20
                analytic = ((1-y[..., 1]/10)*h[..., 1]).mean()
                torch.testing.assert_close(decoded, analytic, rtol=1e-12, atol=1e-12)

    def test_reload_preserves_predictions_gradients_and_skips_preparation(self):
        original = self.prepare(TinyForecast())
        loaded = self.prepare(TinyForecast())
        self.assertTrue(all(trainable_projection for _, trainable_projection in loaded.calls))
        self.assertEqual(self.config['poison_model_source'], 'loaded')
        batch = default_collate([self.dataset[10], self.dataset[11]])
        x, y, xm, ym = batch
        decoder = torch.cat([y[:, :2], torch.zeros_like(y[:, -2:])], dim=1)
        outputs, gradients = [], []
        for model in [original, loaded]:
            out = model(x, xm, decoder, ym)
            loss = torch.nn.functional.mse_loss(out, y[:, -2:])
            gradients.append(torch.autograd.grad(loss, tuple(model.parameters())))
            outputs.append(out)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.config['poison_measurement_C'] = 20.
        with self.assertRaisesRegex(ValueError, 'Incompatible weights-only checkpoint'):
            self.prepare(TinyForecast())

    def test_postprojection_transform_rejected_and_flags_restored(self):
        model = TinyForecast()
        model.invalid_output = True
        with self.assertRaisesRegex(ValueError, 'not the raw final forecast'):
            self.prepare(model)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.decoder.projection._forward_pre_hooks)

    def test_ms_property_channel_can_differ_from_measurement_channel(self):
        self.dataset.features = 'MS'
        self.config['poison_measurement_channel'] = 'OT'
        model = TinyForecast()
        model.output_attention = True
        self.prepare(model)
        self.assertEqual(self.config['poison_output_channels_in_loss'], 1)
        self.assertLess(max(r['max_identity_error'] for r in self.worker.poison_feasibility_results['results']), 1e-5)

    def test_feasibility_only_worker_skips_all_attack_work(self):
        self.worker._init_attack_worker_process = Mock(return_value=(TinyForecast(), None, self.config))
        self.worker.init_logger_object = Mock()
        self.worker._update_config = Mock()
        self.worker._end_logger_object = Mock()
        self.worker.init_attack = Mock()
        self.worker.start_attack = Mock()
        self.worker.load_or_train_poisoned_global_model = Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            self.worker.worker_process({}, {}, {}, {}, None)
        self.worker.load_or_train_poisoned_global_model.assert_called_once()
        self.worker.init_attack.assert_not_called()
        self.worker.start_attack.assert_not_called()
        self.worker._end_logger_object.assert_called_once()

    def test_physical_threshold_matches_equivalent_standardized_threshold(self):
        self.dataset.scaler.mean_[:] = 100
        self.dataset.scaler.scale_[:] = 2
        self.config.update(poison_property_threshold=.5, poison_property_units='standardized')
        self.prepare(TinyForecast())
        standardized = self.worker.poison_feasibility_results
        self.config.update(poison_property_threshold=101., poison_property_units='physical',
                           poison_checkpoint_path=str(self.root / 'physical.pt'))
        self.prepare(TinyForecast())
        physical = self.worker.poison_feasibility_results
        self.assertEqual(standardized['constant_property_baseline'], physical['constant_property_baseline'])
        for a, b in zip(standardized['results'], physical['results']):
            self.assertEqual([r['true_property'] for r in a['rows']], [r['true_property'] for r in b['rows']])

    def test_recorded_state_and_gradient_use_prepared_model_without_update(self):
        self.config.update(poison_feasibility_only=False, update_model=False, number_of_batches=1)
        self.worker.g = torch.Generator().manual_seed(43)
        loader = DataLoader(self.dataset, batch_size=3)
        self.worker.val_dataloader = loader
        model = TinyForecast()
        with contextlib.redirect_stdout(io.StringIO()):
            xs, ys, states, gradients, _ = self.worker.train_model_and_record(model, loader, self.config)
        self.assertEqual(len(gradients), 1)
        self.assertFalse(model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, states[0][name], rtol=0, atol=0)
        decoder = torch.cat([xs[0][:, -2:], torch.zeros_like(ys[0])], dim=1)
        out = model(xs[0], self.worker.all_batch_input_marks[0], decoder, self.worker.all_batch_target_marks[0])
        expected = torch.autograd.grad(torch.nn.functional.mse_loss(out, ys[0]), tuple(model.parameters()))
        for actual, reference in zip(gradients[0], expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_fixed_projection_identity_and_full_projection_preserved(self):
        for mode, alpha, beta in [('M', 1., 0.), ('MS', -2., .3)]:
            with self.subTest(mode=mode):
                self.dataset.features = mode
                self.config.update(poison_strategy='fixed_projection', poison_measurement_channel='OT',
                                   poison_gradient_scale=alpha, poison_gradient_offset=beta,
                                   poison_measurement_C=None, poison_feature_epochs=99,
                                   poison_checkpoint_path=str(self.root / f'{mode}.pt'))
                model = TinyForecast()
                model.encoder.bias.requires_grad_(False)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                flags = {name: p.requires_grad for name, p in model.named_parameters()}
                self.prepare(model)
                self.assertFalse(torch.equal(before['encoder.weight'], model.encoder.weight))
                for name in ['decoder.projection.weight', 'decoder.projection.bias', 'encoder.bias']:
                    self.assertTrue(torch.equal(before[name], dict(model.named_parameters())[name]))
                self.assertEqual(flags, {name: p.requires_grad for name, p in model.named_parameters()})
                self.assertFalse(model.decoder.projection._forward_pre_hooks)
                self.assertEqual(self.config['poison_feature_epochs'], 0)
                report = self.worker.poison_feasibility_results
                self.assertNotIn('C', report['design'])
                self.assertEqual([row['stage'] for row in report['preparation_history']], ['measurement'])
                self.assertEqual(report['design']['strategy'], 'fixed_projection')
                for result in report['results']:
                    for row in result['rows']:
                        self.assertLess(row['raw_identity_error'], 1e-5)
                        self.assertLess(row['identity_error'], 1e-5)
                        self.assertNotIn('target_correction', row)
                        self.assertAlmostEqual(row['estimate'], (row['gradient_scalar'] - beta) / alpha)
                markdown = (self.root / f'{mode}_feasibility.md').read_text()
                self.assertIn('(gradient - beta) / alpha', markdown)
                self.assertNotIn('C=', markdown)
                self.assertNotIn('tune C', markdown)

    def test_fixed_projection_objective_backpropagates_through_both_factors(self):
        self.config['poison_strategy'] = 'fixed_projection'
        model = TinyForecast()
        captured, checked = {}, []

        def capture(module, args, output):
            if not module.weight.requires_grad:
                args[0].retain_grad()
                output.retain_grad()
                captured.update(h=args[0], output=output)

        handle = model.decoder.projection.register_forward_hook(capture)
        original_step = torch.optim.Adam.step

        def inspect_step(optimizer, *args, **kwargs):
            h, output = captured['h'], captured['output']
            self.assertIsNotNone(output.grad)  # The residual must not be detached.
            self.assertGreater(output.grad.norm().item(), 0)
            residual_route = output.grad @ model.decoder.projection.weight
            direct_route = h.grad - residual_route
            # The h factor also has its own derivative, beyond its role in forecasts.
            self.assertGreater(direct_route[..., 1].norm().item(), 1e-8)
            torch.testing.assert_close(direct_route[..., [0, 2, 3]], torch.zeros_like(direct_route[..., [0, 2, 3]]))
            self.assertGreater(model.encoder.weight.grad.norm().item(), 0)
            checked.append(True)
            return original_step(optimizer, *args, **kwargs)

        try:
            with patch.object(torch.optim.Adam, 'step', inspect_step):
                self.prepare(model)
        finally:
            handle.remove()
        self.assertTrue(checked)
        self.assertFalse(model.decoder.projection._forward_pre_hooks)

    def test_fixed_projection_weight_budget_uses_fixed_base_references(self):
        for rho in [0., .005]:
            with self.subTest(rho=rho):
                base = TinyForecast()
                with torch.no_grad():
                    base.encoder.bias.zero_()  # Exercise the reference-norm floor.
                base_path = self.root / f'base-{rho}.pt'
                torch.save(base.state_dict(), base_path)
                self.config.update(poison_strategy='fixed_projection', poison_base_checkpoint_path=str(base_path),
                                   poison_max_relative_weight_change=rho, poison_train_epochs=2,
                                   poison_checkpoint_path=str(self.root / f'budget-{rho}.pt'))
                model = TinyForecast()
                references = {name: p.detach().clone() for name, p in base.named_parameters()}
                checked = []
                original_step = torch.optim.Adam.step

                def check_bounds():
                    for name, p in model.named_parameters():
                        reference = references[name]
                        if name.startswith('decoder.') or rho == 0:
                            self.assertTrue(torch.equal(p, reference))
                        else:
                            radius = rho * max(reference.norm().item(), 1e-12)
                            self.assertLessEqual((p - reference).norm().item(), radius + max(1e-20, radius * 1e-4))

                def inspect_step(optimizer, *args, **kwargs):
                    # Before every step, the preceding step must already be projected.
                    check_bounds()
                    checked.append(True)
                    return original_step(optimizer, *args, **kwargs)

                with patch.object(torch.optim.Adam, 'step', inspect_step):
                    self.prepare(model)
                check_bounds()
                self.assertGreater(len(checked), 2)
                checkpoint = torch.load(self.config['poison_checkpoint_path'], weights_only=True)
                self.assertTrue(torch.equal(checkpoint['projection_reference']['weight'], base.decoder.projection.weight))
                self.assertTrue(torch.equal(checkpoint['projection_reference']['bias'], base.decoder.projection.bias))

    def test_fixed_projection_reload_and_checkpoint_isolation(self):
        self.config['poison_strategy'] = 'fixed_projection'
        original = self.prepare(TinyForecast())
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Reload must not train')):
            loaded = self.prepare(TinyForecast())
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[name]))
        self.assertEqual(self.config['poison_model_source'], 'loaded')
        checkpoint = torch.load(self.config['poison_checkpoint_path'], weights_only=True)
        checkpoint['model_state_dict']['decoder.projection.bias'][0] += 1
        torch.save(checkpoint, self.config['poison_checkpoint_path'])
        with self.assertRaisesRegex(ValueError, 'original complete projection'):
            self.prepare(TinyForecast())
        self.config['poison_strategy'] = 'constant_bias'
        self.config.pop('poison_gradient_coordinates', None)
        with self.assertRaisesRegex(ValueError, 'Incompatible weights-only checkpoint'):
            self.prepare(TinyForecast())

    def test_legacy_constant_bias_metadata_and_checkpoint_reload(self):
        self.prepare(TinyForecast())  # Explicit constant_bias retains legacy metadata.
        self.assertEqual(self.config['poison_strategy'], 'constant_bias')
        checkpoint = torch.load(self.config['poison_checkpoint_path'], weights_only=True)
        metadata = checkpoint['poisoning']
        self.assertEqual(set(metadata), {
            'method', 'dataset', 'model_class', 'architecture', 'parameter_names', 'weight_name',
            'coordinate', 'source_csv', 'source_size', 'source_mtime_ns', 'channels', 'property_channel',
            'features', 'target', 'seq_len', 'label_len', 'pred_len', 'scale', 'timeenc', 'freq',
            'scaler_mean', 'scaler_scale', 'threshold', 'threshold_units', 'duration', 'C',
            'auxiliary_split', 'partitions', 'base_checkpoint', 'feature_epochs', 'measurement_epochs',
            'batch_size', 'learning_rate', 'seed', 'initialization_seed',
        })
        self.assertEqual(metadata['method'], 'weights_only_projection_measurement_v1')
        self.assertNotIn('strategy', metadata)
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Legacy reload must not train')):
            self.prepare(TinyForecast())  # Explicit constant_bias now present in config.
        self.config['poison_strategy'] = 'fixed_projection'
        with self.assertRaisesRegex(ValueError, 'Incompatible weights-only checkpoint'):
            self.prepare(TinyForecast())

    def test_fixed_projection_validation_and_exception_cleanup(self):
        for changes in [dict(poison_strategy='unknown'), dict(poison_gradient_scale=0.),
                        dict(poison_gradient_scale=float('nan')), dict(poison_gradient_offset=float('inf')),
                        dict(poison_max_relative_weight_change=-1), dict(poison_max_relative_weight_change=float('inf'))]:
            with self.subTest(changes=changes):
                config = dict(self.config, poison_strategy='fixed_projection')
                config.update(changes)
                with self.assertRaises(ValueError):
                    self.worker.load_or_train_poisoned_global_model(TinyForecast(), config)
        self.config['poison_strategy'] = 'fixed_projection'
        model = TinyForecast()
        model.invalid_output = True
        model.encoder.bias.requires_grad_(False)
        flags = [p.requires_grad for p in model.parameters()]
        with self.assertRaisesRegex(ValueError, 'not the raw final forecast'):
            self.prepare(model)
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        self.assertFalse(model.decoder.projection._forward_pre_hooks)

    def test_single_coordinate_list_matches_legacy_fixed_projection(self):
        self.config.pop('poison_strategy')  # Current default is fixed_projection.
        model = TinyForecast()
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        self.prepare(model)
        legacy = self.worker.poison_feasibility_results
        self.assertEqual(self.config['poison_strategy'], 'fixed_projection')
        self.assertNotIn('coordinates', legacy['design'])
        self.config.update(poison_gradient_coordinates=[{'channel': 'MUFL', 'hidden_coordinate': 1}],
                           poison_checkpoint_path=str(self.root / 'list.pt'))
        other = TinyForecast()
        other.load_state_dict(initial)
        self.prepare(other)
        listed = self.worker.poison_feasibility_results
        self.assertEqual(legacy['design'], listed['design'])
        self.assertEqual(legacy['results'], listed['results'])
        self.assertEqual(legacy['preparation_history'], listed['preparation_history'])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[name]))
        self.assertEqual(self.config['poison_measurement_count'], 1)

    def test_multi_coordinates_identities_reports_and_reload(self):
        cases = [('M', [1, 1, 1]), ('M', [0, 1, 2]), ('MS', [2, 2, 2])]
        for case, (mode, rows) in enumerate(cases):
            with self.subTest(mode=mode, rows=rows):
                self.dataset.features = mode
                coordinates = [[row, j] for row, j in zip(rows, [0, 2, 3])]
                self.config.update(poison_strategy='fixed_projection', poison_gradient_scale=-2., poison_gradient_offset=.3,
                                   poison_gradient_coordinates=[{'channel': row, 'hidden_coordinate': j} for row, j in coordinates],
                                   poison_checkpoint_path=str(self.root / f'multi{case}.pt'),
                                   poison_max_relative_weight_change=.01)
                model = TinyForecast()
                model.encoder.bias.requires_grad_(False)
                before = {name: value.clone() for name, value in model.state_dict().items()}
                self.prepare(model)
                report = self.worker.poison_feasibility_results
                self.assertEqual(report['design']['coordinates'], coordinates)
                self.assertNotIn('coordinate', report['design'])
                self.assertNotIn('poison_gradient_coordinate', self.config)
                self.assertNotIn('poison_hidden_coordinate', self.config)
                self.assertNotIn('poison_measurement_channel', self.config)
                self.assertEqual(self.config['poison_measurement_count'], 3)
                self.assertTrue(torch.equal(before['decoder.projection.weight'], model.decoder.projection.weight))
                self.assertTrue(torch.equal(before['decoder.projection.bias'], model.decoder.projection.bias))
                self.assertFalse(model.encoder.bias.requires_grad)
                self.assertFalse(model.decoder.projection._forward_pre_hooks)
                self.assertFalse(torch.equal(before['encoder.weight'], model.encoder.weight))
                # A single forward per preparation/verification minibatch, not per coordinate.
                partitions = report['design']['partitions']
                prep_count = sum(p['preparation_stop'] - p['preparation_start'] for p in partitions)
                training_calls = (prep_count + self.config['poison_train_batch_size'] - 1) // self.config['poison_train_batch_size']
                self.assertEqual(len(model.calls), training_calls + sum(r['batches'] for r in report['results']))
                for history in report['preparation_history']:
                    self.assertAlmostEqual(history['mse'], .5 * history['individual_mse'] + .5 * history['ensemble_mse'], places=6)
                for result in report['results']:
                    for row in result['rows']:
                        self.assertNotIn('gradient_scalar', row)
                        self.assertEqual(row['coordinates'], coordinates)
                        np.testing.assert_allclose(row['individual_estimates'], (np.array(row['gradient_scalars']) - .3) / -2.)
                        self.assertAlmostEqual(row['estimate'], np.mean(row['individual_estimates']))
                        self.assertLess(max(row['raw_identity_errors']), 1e-5)
                        self.assertLess(max(row['coordinate_identity_errors']), 1e-5)
                        self.assertLess(row['identity_error'], 1e-5)
                    errors = np.array([r['individual_estimates'] for r in result['rows']]) - np.array([r['true_property'] for r in result['rows']])[:, None]
                    for q, summary in enumerate(result['coordinate_summary'][:-1]):
                        self.assertAlmostEqual(summary['mae'], np.abs(errors[:, q]).mean())
                        self.assertAlmostEqual(summary['rmse'], np.sqrt(np.square(errors[:, q]).mean()))
                        self.assertAlmostEqual(summary['bias'], errors[:, q].mean())
                    self.assertEqual(result['coordinate_summary'][-1]['mae'], result['mae'])
                    matrix = result['signed_error_correlation']['matrix']
                    if len(errors) > 1:
                        np.testing.assert_allclose(matrix, np.corrcoef(errors.T))
                    else:
                        self.assertEqual(matrix, [[None] * 3 for _ in range(3)])
                markdown = (self.root / f'multi{case}_feasibility.md').read_text()
                self.assertIn('arithmetic mean', markdown)
                self.assertIn('Gradient scalars (coordinate order)', markdown)
                self.assertIn('Signed bias', markdown)
                self.assertIn('signed estimation-error correlations', markdown)
                loaded = TinyForecast()
                loaded.encoder.bias.requires_grad_(False)
                with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Reload must not train')):
                    self.prepare(loaded)
                self.assertEqual(report['results'], self.worker.poison_feasibility_results['results'])
                self.assertEqual(report['design'], self.worker.poison_feasibility_results['design'])
                self.assertTrue(all(flag for _, flag in loaded.calls))
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, loaded.state_dict()[name]))
                self.config['poison_ensemble_loss_weight'] = .25
                with self.assertRaisesRegex(ValueError, 'Incompatible weights-only checkpoint'):
                    self.prepare(loaded)
                self.config['poison_ensemble_loss_weight'] = .5
                self.config['poison_gradient_coordinates'].reverse()
                with self.assertRaisesRegex(ValueError, 'Incompatible weights-only checkpoint'):
                    self.prepare(loaded)
                self.config['poison_gradient_coordinates'] = self.config['poison_gradient_coordinates'][:1]
                with self.assertRaisesRegex(ValueError, 'Incompatible weights-only checkpoint'):
                    self.prepare(loaded)

    def test_multi_coordinate_validation(self):
        self.config['poison_strategy'] = 'fixed_projection'
        invalid_lists = [[], None, [{'channel': 'HUFL'}], [{'channel': 'missing', 'hidden_coordinate': 0}],
                         [{'channel': 0, 'hidden_coordinate': 4}],
                         [{'channel': 'HUFL', 'hidden_coordinate': 1}, {'channel': 0, 'hidden_coordinate': 1}]]
        for coordinates in invalid_lists:
            with self.subTest(coordinates=coordinates):
                self.config['poison_gradient_coordinates'] = coordinates
                with self.assertRaises(ValueError):
                    self.prepare(TinyForecast())
        self.config['poison_gradient_coordinates'] = [{'channel': 0, 'hidden_coordinate': 0}, {'channel': 0, 'hidden_coordinate': 1}]
        self.dataset.features = 'MS'
        with self.assertRaisesRegex(ValueError, 'MS measurement'):
            self.prepare(TinyForecast())
        self.dataset.features = 'M'
        self.config['poison_strategy'] = 'constant_bias'
        with self.assertRaisesRegex(ValueError, 'only for fixed_projection'):
            self.prepare(TinyForecast())
        self.config['poison_strategy'] = 'fixed_projection'
        for value in [-.1, 1., float('nan'), float('inf')]:
            self.config['poison_ensemble_loss_weight'] = value
            with self.assertRaisesRegex(ValueError, 'poison_ensemble_loss_weight'):
                self.prepare(TinyForecast())

    def test_multi_s_mode_and_coordinate_identity_cannot_cancel(self):
        ds = AuxiliaryDataset(self.root, features='S', channels=1)
        self.worker.train_datasets = self.worker.test_datasets = [ds]
        self.config.update(poison_strategy='fixed_projection', poison_target_channel='OT',
                           poison_gradient_coordinates=[{'channel': 'OT', 'hidden_coordinate': 0},
                                                        {'channel': 0, 'hidden_coordinate': 3}])
        model = TinyForecast()
        model.encoder = torch.nn.Linear(1, 4)
        model.decoder.projection = torch.nn.Linear(4, 1)
        self.prepare(model)
        self.assertEqual(self.config['poison_output_channels_in_loss'], 1)
        for result in self.worker.poison_feasibility_results['results']:
            for row in result['rows']:
                self.assertLess(max(row['raw_identity_errors']), 1e-5)
        # Corrupt two actual coordinates in opposite directions: their average
        # cancels, so an ensemble-only check would miss this.
        def corrupt(gradient):
            gradient = gradient.clone()
            gradient[0, 0] += 1
            gradient[0, 3] -= 1
            return gradient
        handle = model.decoder.projection.weight.register_hook(corrupt)
        try:
            with self.assertRaisesRegex(RuntimeError, r'Raw gradient identity failed at \[0, 0\]'):
                self.prepare(model)
        finally:
            handle.remove()
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.decoder.projection._forward_pre_hooks)

    def test_multi_constant_errors_have_undefined_correlation(self):
        class ConstantWindows(AuxiliaryDataset):
            def __getitem__(self, index):
                return super().__getitem__(0)
        ds = ConstantWindows(self.root)
        self.worker.train_datasets = self.worker.test_datasets = [ds]
        self.config.update(poison_strategy='fixed_projection', poison_ensemble_loss_weight=0.,
                           poison_gradient_coordinates=[{'channel': 0, 'hidden_coordinate': 0}, {'channel': 1, 'hidden_coordinate': 2}],
                           poison_validation_batch_sizes=[1], poison_validation_batches=3)
        self.prepare(TinyForecast())
        report = self.worker.poison_feasibility_results
        self.assertEqual(report['results'][0]['signed_error_correlation']['matrix'], [[None, None], [None, None]])
        for history in report['preparation_history']:
            self.assertEqual(history['mse'], history['individual_mse'])

    def test_real_transformer_cpu_smoke(self):
        from ts_inverse.models.Transformer import Model
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            (self.root / 'ETTh1.csv').write_text('date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n2020,1,2,3,4,5,6,7\n')
            ds = AuxiliaryDataset(self.root, channels=7, seq_len=96, label_len=48, pred_len=96, count=600)
            self.worker.train_datasets = self.worker.test_datasets = [ds]
            self.config.update(poison_holdout_fraction=.5, poison_train_batch_size=128, poison_validation_batch_sizes=[1, 2], poison_validation_batches=1)
            model = Model()
            keys = list(model.state_dict())
            self.prepare(model)
            self.assertEqual(keys, list(model.state_dict()))
            self.assertFalse(model.decoder.projection._forward_pre_hooks)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
