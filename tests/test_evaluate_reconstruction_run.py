import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from pi_eval import evaluate_reconstruction_run as run


class ReconstructionReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runs = self.root / 'reconstructions'
        self.runs.mkdir()
        self.output = self.root / 'evaluation_results'

    def make_run(self, name):
        folder = self.runs / name
        folder.mkdir()
        config = dict(dataset='ETTh1', root_path=str(self.root), features='S',
                      target='OT', seq_len=2, pred_len=1, label_len=1, scale=True)
        (folder / 'config.json').write_text(json.dumps(config))
        # Unequal batch sizes must still receive equal weight. Create files out
        # of order to verify that both lists are ordered by numeric batch index.
        samples = {
            10: (torch.zeros(1, 2, 1), torch.zeros(1, 2, 1)),
            2: (torch.ones(2, 2, 1), torch.ones(2, 2, 1)),
            0: (torch.zeros(1, 2, 1), torch.tensor([[[0.0], [1.0]]])),
        }
        for index, (original, reconstructed) in samples.items():
            torch.save(dict(original_data=original, reconstructed_data=reconstructed),
                       folder / f'batch_{index}.pt')
        return folder

    def options(self):
        return ['--reconstructions-dir', str(self.runs), '--output-dir', str(self.output),
                '--thresholds', '.5', '2', '--durations', '1', '2']

    @staticmethod
    def baseline(settings, shape):
        return [np.ones(shape)] * 10, 100

    def test_combined_columns_lists_and_difference(self):
        folder = self.make_run('ETTh1_20261106_120000_000000-0500')
        with patch.object(run, 'sample_validation_batches', side_effect=self.baseline), \
             contextlib.redirect_stdout(io.StringIO()):
            output = run.main([str(folder), *self.options()])
        report = output.read_text()
        combined = report.split('## Combined performance by property')[1].split('## Metric definitions')[0]
        header = next(line for line in combined.splitlines() if line.startswith('| Duration'))
        self.assertEqual([cell.strip() for cell in header.strip('|').split('|')], [
            'Duration', 'Channel', 'Threshold', 'Mean reconstruction error', 'Mean random error',
            'Mean reconstruction error - Mean random error', 'List of all reconstruction errors',
            'List of all random errors',
        ])
        self.assertIn('Error lists follow batch order: [0, 2, 10]', combined)
        self.assertIn('| 1 | OT | 0.5 | 16.67 | 66.67 | -50.00 | [50.00, 0.00, 0.00] | [100.00, 0.00, 100.00] |', combined)
        self.assertIn('| 2 | OT | 0.5 | 0.00 | 66.67 | -66.67 | [0.00, 0.00, 0.00] | [100.00, 0.00, 100.00] |', combined)
        self.assertIn('| 1 | OT | 2 | 0.00 | 0.00 | 0.00 | [0.00, 0.00, 0.00] | [0.00, 0.00, 0.00] |', combined)
        self.assertNotIn('Mean error reduction', combined)
        self.assertIn('## Batch 10: batch_10.pt', report)
        self.assertIn('## Run configuration', report)

    def test_all_writes_one_report_per_folder_including_untimestamped_runs(self):
        folders = [self.make_run(name) for name in ['ETTh1_20261106_120000-0500', 'custom_run']]
        (self.runs / 'notes.md').write_text('not a run')
        with patch.object(run, 'sample_validation_batches', side_effect=self.baseline), \
             contextlib.redirect_stdout(io.StringIO()):
            outputs = run.main(['all', *self.options()])
        self.assertEqual({p.name for p in outputs},
                         {f'{folder.name}_property_inference.md' for folder in folders})
        self.assertTrue(all(p.is_file() for p in outputs))
        for folder in folders:
            self.assertIn(folder.name, (self.output / f'{folder.name}_property_inference.md').read_text())

    def test_all_continues_past_failed_folder_and_returns_failure_status(self):
        (self.runs / 'aaa_broken').mkdir()
        folder = self.make_run('zzz_valid')
        with patch.object(run, 'sample_validation_batches', side_effect=self.baseline), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                run.main(['all', *self.options()])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn('aaa_broken', errors.getvalue())
        self.assertIn('Generated 1 reports', errors.getvalue())
        self.assertTrue((self.output / f'{folder.name}_property_inference.md').exists())

    def test_omitted_folder_still_selects_latest_timestamp(self):
        self.make_run('ETTh1_20261105_120000-0500')
        latest = self.make_run('ETTh1_20261106_120000-0500')
        with patch.object(run, 'sample_validation_batches', side_effect=self.baseline), \
             contextlib.redirect_stdout(io.StringIO()):
            output = run.main(self.options())
        self.assertEqual(output.name, f'{latest.name}_property_inference.md')
        self.assertEqual(len(list(self.output.glob('*.md'))), 1)

    def test_all_rejects_empty_root(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                run.main(['all', *self.options()])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('No reconstruction folders', errors.getvalue())

    def test_wandb_logs_exact_names_unrounded_means_and_ordered_lists(self):
        folder = self.make_run('ETTh1_20261106_120000-0500')
        wandb_run = Mock(spec=['log'])
        with patch.object(run, 'sample_validation_batches', side_effect=self.baseline), \
             contextlib.redirect_stdout(io.StringIO()):
            output = run.main([str(folder), *self.options()], wandb_run=wandb_run)
        self.assertTrue(output.is_file())
        wandb_run.log.assert_called_once()
        values = wandb_run.log.call_args.args[0]
        prefix = 'OT/0.5/1/'
        self.assertEqual(values[prefix + 'Mean reconstruction error'], 50 / 3)
        self.assertEqual(values[prefix + 'Mean random error'], 200 / 3)
        self.assertAlmostEqual(values[prefix + 'Mean reconstruction error - Mean random error'], -50)
        self.assertEqual(values[prefix + 'List of all reconstruction errors'], [50.0, 0.0, 0.0])
        self.assertEqual(values[prefix + 'List of all random errors'], [100.0, 0.0, 100.0])
        self.assertEqual(values['OT/2/1/Mean random error'], 0.0)
        self.assertEqual(values['OT/0.5/2/Mean reconstruction error'], 0.0)
        self.assertEqual(values['property_inference/batch_indices'], [0, 2, 10])
        self.assertEqual(values['property_inference/run_folder'], str(folder.resolve()))
        self.assertEqual(len(values), 4 * 5 + 2)  # Four property rows plus metadata.
        self.assertEqual(wandb_run.log.call_args.kwargs, {})  # No attack-step reset.

    def test_wandb_all_logs_separate_entries_for_each_folder(self):
        folders = [self.make_run(name) for name in ['first', 'second']]
        wandb_run = Mock(spec=['log'])
        with patch.object(run, 'sample_validation_batches', side_effect=self.baseline), \
             contextlib.redirect_stdout(io.StringIO()):
            run.main(['all', *self.options()], wandb_run=wandb_run)
        self.assertEqual(wandb_run.log.call_count, 2)
        self.assertEqual(
            {call.args[0]['property_inference/run_folder'] for call in wandb_run.log.call_args_list},
            {str(folder.resolve()) for folder in folders},
        )


if __name__ == '__main__':
    unittest.main()
