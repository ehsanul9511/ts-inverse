import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from ts_inverse.workers.attack_ts_inverse_worker import AttackTSInverseWorker


class Windows(Dataset):
    features, target = 'S', 'OT'
    seq_len, label_len, pred_len = 4, 1, 1

    def __init__(self):
        self.x = (torch.arange(4)[None, :] < torch.arange(5)[:, None]).float().unsqueeze(-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], torch.full((2, 1), float(i)), torch.full((4, 4), float(i)), torch.full((2, 4), float(i))


class Forecast(torch.nn.Module):
    name = 'tiny'

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x, x_mark, decoder, y_mark):
        return self.linear(x.mean(dim=1, keepdim=True))


class PropertyBatchSamplingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Windows()
        self.worker = AttackTSInverseWorker(0)
        self.worker.train_datasets = [self.dataset]
        self.worker.g = torch.Generator().manual_seed(7)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, generator=self.worker.g)
        self.worker.val_dataloader = self.loader
        self.config = dict(dataset='ETTh1', device='cpu', seed=43, number_of_batches=3,
                           attack_number_of_batches=3, attack_batch_sampling='property_percentiles',
                           attack_sampling_target_channel='OT', attack_sampling_threshold=.5,
                           attack_sampling_duration=1, model_train_epochs=0, update_model=False)

    def select(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            loader = self.worker._property_percentile_attack_loader(self.loader, self.config)
        return list(loader), output.getvalue()

    def test_selects_sorted_extremes_and_preserves_labels_marks_and_rng(self):
        before = self.dataset.x.clone()
        rng, sampler_rng = torch.random.get_rng_state().clone(), self.worker.g.get_state().clone()
        batches, output = self.select()
        self.assertEqual([batch[0].mean().item() for batch in batches], [0, .5, 1])
        for batch, index in zip(batches, [0, 2, 4]):
            self.assertEqual(batch[1][0, 0, 0].item(), index)
            self.assertEqual(batch[2][0, 0, 0].item(), index)
            self.assertEqual(batch[3][0, 0, 0].item(), index)
        self.assertTrue(torch.equal(self.dataset.x, before))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(torch.equal(self.worker.g.get_state(), sampler_rng))
        self.assertIn('P000', output)
        self.assertIn('P100', output)
        self.assertIn('Batch 2 (attack)', output)
        first_selection = self.config['attack_sampling_selected_batches']
        self.select()
        self.assertEqual(first_selection, self.config['attack_sampling_selected_batches'])

    def test_extra_recorded_batches_do_not_restrict_attack_percentiles(self):
        self.config['number_of_batches'] = 5
        batches, _ = self.select()
        self.assertEqual([batch[0].mean().item() for batch in batches[:3]], [0, .5, 1])
        indices = [row['window_indices'][0] for row in self.config['attack_sampling_selected_batches']]
        self.assertEqual(len(set(indices)), 5)

    def test_sustained_runs_and_identical_scores(self):
        self.dataset.x[:] = torch.tensor([1., 0., 1., 0.]).reshape(1, 4, 1)
        self.config['attack_sampling_duration'] = 2
        _, output = self.select()
        self.assertTrue(all(row['property_score'] == 0 for row in self.config['attack_sampling_selected_batches']))
        self.assertIn('all candidate scores are identical', output)
        self.config['attack_sampling_duration'] = 1
        self.select()
        self.assertTrue(all(row['property_score'] == .5 for row in self.config['attack_sampling_selected_batches']))

    def test_too_small_candidate_pool_rejected(self):
        self.config['attack_sampling_candidate_batches'] = 2
        with self.assertRaisesRegex(ValueError, 'needs 3'):
            self.select()

    def test_recording_uses_selected_batches_but_default_does_not_call_sampler(self):
        with contextlib.redirect_stdout(io.StringIO()):
            inputs, targets, _, gradients, _ = self.worker.train_model_and_record(Forecast(), self.loader, self.config)
        self.assertEqual([x.mean().item() for x in inputs], [0, .5, 1])
        self.assertEqual([y.item() for y in targets], [0, 2, 4])
        self.assertEqual(len(gradients), 3)
        self.config['attack_batch_sampling'] = 'random'
        with patch.object(self.worker, '_property_percentile_attack_loader') as select:
            self.worker.train_model_and_record(Forecast(), self.loader, self.config)
        select.assert_not_called()

    def test_sampling_metadata_and_chart_saved_with_run(self):
        self.select()
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            self.config['reconstruction_output_dir'] = root
            folder = self.worker._create_reconstruction_run(self.config)
            self.assertTrue((folder / 'attack_sampling_percentiles.png').is_file())
            config = json.loads((folder / 'config.json').read_text())
            self.assertEqual([row['property_score'] for row in config['attack_sampling_selected_batches']], [0, .5, 1])


if __name__ == '__main__':
    unittest.main()
