import contextlib
import io
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader
import test_poisoned_global_model as fixtures
from test_poisoned_global_model import TinyForecast


class ClientPropertyInferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PoisonedGlobalModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.worker = self.fixture.worker
        self.config = dict(self.fixture.config, attack_mode='poisoning_assisted_property_inference',
                           poison_feasibility_only=False, update_model=False, defense_name='none',
                           number_of_batches=2, attack_number_of_batches=2, verbose=False,
                           property_inference_output_dir=str(self.fixture.root))
        self.worker.g = torch.Generator().manual_seed(43)
        self.loader = DataLoader(self.fixture.dataset, batch_size=3)
        self.worker.val_dataloader = self.loader
        self.worker._update_config = Mock()

    def run_direct(self, strategy, multi=False, ms=False):
        self.config['poison_strategy'] = strategy
        if ms:
            self.fixture.dataset.features = 'MS'
            self.config['poison_measurement_channel'] = 'OT'
        if multi:
            self.config['poison_gradient_coordinates'] = [
                {'channel': 'OT', 'hidden_coordinate': 0},
                {'channel': 'OT' if ms else 'HUFL', 'hidden_coordinate': 2}]
        model = TinyForecast()
        self.worker.generate_dummy_data = Mock(side_effect=AssertionError('No dummy data'))
        self.worker.gradient_inversion_attack = Mock(side_effect=AssertionError('No inversion'))
        self.worker.log_aggregate_reconstruction_metrics = Mock(side_effect=AssertionError('No reconstruction metrics'))
        self.worker._create_reconstruction_run = Mock(side_effect=AssertionError('No reconstruction files'))
        with contextlib.redirect_stdout(io.StringIO()):
            self.worker.init_attack(model, self.loader, self.config)
            states = [{name: value.clone() for name, value in state.items()} for state in self.worker.all_model_state_dicts]
            gradients = [[value.clone() for value in batch] for batch in self.worker.all_model_gradients]
            self.worker.start_attack(model, self.config)
        report = self.worker.client_property_inference_results
        design = report['design']
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        coords = design.get('coordinates', [design.get('coordinate')])
        for row in report['results'][0]['rows']:
            g = gradients[row['batch']][names.index(design['weight_name'])]
            scalars = np.array([g[k, j].item() for k, j in coords])
            expected = (scalars - design['beta']) / design['alpha'] if strategy == 'fixed_projection' else 3 * scalars / (2 * design['C'])
            self.assertAlmostEqual(row['estimate'], expected.mean())
            self.assertLess(row['identity_error'], 1e-5)
            for detail in row['measurements']:
                self.assertLess(detail['raw_identity_error'], 1e-5)
            self.assertEqual(len(row['per_window_true_property']), 3)
        self.assertFalse(model.decoder.projection._forward_pre_hooks)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, states[-1][name]))
        for original, recorded in zip(gradients, self.worker.all_model_gradients):
            for a, b in zip(original, recorded):
                self.assertTrue(torch.equal(a, b))
        files = list(self.fixture.root.glob('*_client_property_*.json'))
        self.assertEqual(len(files), 1)
        self.assertEqual(json.loads(files[0].read_text())['evaluation'], 'recorded_client_gradients')
        markdown = files[0].with_suffix('.md').read_text()
        self.assertIn('Recorded client batches', markdown)
        self.assertIn('Measurement diagnostics', markdown)
        self.assertNotIn('held-out batches', markdown)
        return model

    def test_constant_bias_end_to_end(self):
        self.run_direct('constant_bias')

    def test_fixed_projection_end_to_end(self):
        self.run_direct('fixed_projection')

    def test_multi_projection_end_to_end(self):
        self.run_direct('fixed_projection', multi=True)

    def test_multi_ms_projection_end_to_end(self):
        self.run_direct('fixed_projection', multi=True, ms=True)

    def test_default_dispatch_and_guards(self):
        worker = self.worker
        worker.gradient_inversion_attack = Mock(return_value='inversion')
        args = (TinyForecast(), {}, 0, [], None, None, None, None)
        self.assertEqual(worker.attack_batch(*args), 'inversion')
        for overrides in [dict(poison_global_model=False), dict(poison_feasibility_only=True),
                          dict(update_model=True), dict(defense_name='dp'), dict(attack_mode='unknown')]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                worker._uses_direct_property_inference(dict(self.config, **overrides))
        with self.assertRaisesRegex(ValueError, 'Missing matching'):
            worker.poisoning_assisted_property_inference(TinyForecast(), self.config, 0, [], None, None)

    def test_corrupt_recorded_gradient_rejected_and_hook_removed(self):
        model = self.run_direct('fixed_projection', multi=True)
        gradients = [g.clone() for g in self.worker.all_model_gradients[0]]
        index = self.config['poison_gradient_parameter_index']
        gradients[index][2, 0] += 1
        with self.assertRaisesRegex(ValueError, 'Recorded client gradient identity failed'):
            self.worker.attack_batch(model, self.config, 0, gradients, None, None,
                                     self.worker.all_batch_inputs[0], self.worker.all_batch_targets[0])
        self.assertFalse(model.decoder.projection._forward_pre_hooks)

    def test_both_mode_compares_same_batches_and_keeps_reconstruction_evaluation(self):
        model = self.run_direct('fixed_projection', multi=True)
        original_direct = [r['estimate'] for r in self.worker.client_property_inference_results['results'][0]['rows']]
        self.config.update(attack_mode='both', num_attack_steps=2,
                           reconstruction_output_dir=str(self.fixture.root / 'reconstructions'))
        del self.worker._create_reconstruction_run  # Restore the real directory/config writer.
        self.worker.log_aggregate_reconstruction_metrics = Mock()
        calls = []

        def reconstruct(m, config, batch, gradients, dx, dy, inputs, targets):
            self.assertIs(gradients, self.worker.all_model_gradients[batch])
            self.assertIs(inputs, self.worker.all_batch_inputs[batch])
            calls.append(batch)
            result = torch.zeros_like(inputs)  # Known property: zero (strict threshold > 0).
            torch.save({'original_data': inputs, 'reconstructed_data': result},
                       self.worker._reconstruction_run_dir / f'batch_{batch}.pt')
            return result

        self.worker.gradient_inversion_attack = Mock(side_effect=reconstruct)
        with patch('pi_eval.evaluate_reconstruction_run.main') as evaluate, contextlib.redirect_stdout(io.StringIO()):
            self.worker.start_attack(model, self.config)
        evaluate.assert_called_once()
        self.worker.log_aggregate_reconstruction_metrics.assert_called_once()
        self.assertEqual(calls, [0, 1])
        report = self.worker.client_property_inference_results
        for expected, row in zip(original_direct, report['results'][0]['rows']):
            self.assertEqual(row['estimate'], expected)
            self.assertEqual(row['reconstruction_estimate'], 0.)
            self.assertEqual(row['reconstruction_absolute_error'], row['true_property'])
        self.assertEqual({r['method'] for r in report['method_comparison']},
                         {'direct_property_inference', 'gradient_inversion'})
        for method in report['method_comparison']:
            field = 'signed_error' if method['method'] == 'direct_property_inference' else 'reconstruction_signed_error'
            errors = np.array([row[field] for row in report['results'][0]['rows']])
            self.assertAlmostEqual(method['mae'], np.abs(errors).mean())
            self.assertAlmostEqual(method['rmse'], np.sqrt(np.square(errors).mean()))
        paths = list(self.fixture.root.glob('*_client_property_*.json'))
        combined = next(path for path in paths if 'method_comparison' in json.loads(path.read_text()))
        self.assertIn('Direct inference versus gradient inversion', combined.with_suffix('.md').read_text())

    def test_both_requires_completed_inversion(self):
        with self.assertRaisesRegex(ValueError, 'num_attack_steps > 0'):
            self.worker._uses_direct_property_inference(dict(self.config, attack_mode='both', num_attack_steps=0))
        self.config.update(attack_mode='both', num_attack_steps=1)
        self.worker.poisoning_assisted_property_inference = Mock(return_value={'true_property': .2})
        self.worker.gradient_inversion_attack = Mock(return_value=None)
        with self.assertRaisesRegex(RuntimeError, 'No completed reconstruction'):
            self.worker.attack_batch(TinyForecast(), self.config, 0, [], None, None, None, None)


if __name__ == '__main__':
    unittest.main()
