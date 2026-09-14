"""Independent-head metrics and immutable baseline fallback in final reports."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from mydata_bench.basic_method.common import create_json, file_hash, fingerprint
from mydata_bench.basic_method.run import fallback_row
from mydata_bench.basic_method.reporting.__main__ import endpoint_accuracy, score_experiment, success_readout, write_reports


class BothHeadReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.samples = [{'example_id': eid, 'holdout': eid != 'm'} for eid in ('s', 'f', 'm')]
        self.labels = {eid: {'reward': 5 if eid == 's' else 1,
                            'split': 'suc' if eid == 's' else 'fail', 'subset': 'task',
                            'source_suc_id': 's', 'video_sha256': 'same-video'} for eid in ('s', 'f', 'm')}
        self.cfg = {'model': 'meter', 'protocol': 'official', 'output_dir': str(self.root),
                    'scopes': ['all_frames'], 'top_k': [8], 'controls': ['wrong_region', 'low_rank']}
        for field, value in [('inputs', self.samples), ('labels', self.labels), ('ranking_inputs', [])]:
            path = self.root / f'{field}.json'
            create_json(path, value)
            self.cfg[field] = str(path)
            self.cfg[field + '_sha256'] = file_hash(path)
        identity = {'config': self.cfg, 'evaluation_ids': ['s', 'f', 'm']}
        create_json(self.root / 'run_identity.json', identity)
        create_json(self.root / 'run_config.json', self.cfg)
        run_id = fingerprint(identity)
        self.base = [{'example_id': sample['example_id'], 'sample_id': fingerprint(sample),
                      'run_id': run_id, 'condition': 'baseline', 'status': 'ok',
                      'progress': 0. if i == 0 else 1., 'success_probability': [1., 0., None][i],
                      'sas_applied': False, 'baseline_fallback': False, 'fallback_reason': None}
                     for i, sample in enumerate(self.samples)]
        self.predictions = self.root / 'predictions'
        self.predictions.mkdir()
        baseline_path = self.predictions / 'baseline.jsonl'
        self.write_rows(baseline_path, self.base)
        for kind in ['target', 'wrong_region', 'low_rank']:
            condition = f'all_frames:{kind}:8'
            rows = [{**self.base[0], 'condition': condition, 'sas_applied': True, 'success_probability': .8},
                    fallback_row(self.samples[1], condition, self.base[1], baseline_path,
                                 {'reason': 'missing_exact_frames'}, run_id),
                    {**self.base[2], 'condition': condition, 'sas_applied': True, 'success_probability': 0.}]
            self.write_rows(self.predictions / f'all_frames_{kind}_8.jsonl', rows)

    @staticmethod
    def write_rows(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_success_is_independent_of_progress_and_keeps_fixed_denominators(self):
        result = score_experiment(self.cfg)
        self.assertEqual(result['conditions']['baseline']['full']['mae'], 4.)
        success = result['secondary_success_head']['baseline']
        self.assertEqual((success['full']['n'], success['full']['expected']), (2, 3))
        self.assertEqual(success['full']['mae'], 0.)
        self.assertEqual(success['full']['accuracy']['0.125/0.875']['all']['rate_all_expected'], 2 / 3)
        self.assertEqual(success['holdout']['accuracy']['0.125/0.875']['all']['rate_all_expected'], 1.)
        self.assertEqual(success['full']['by_task']['task']['n'], 2)
        self.assertEqual(success['full']['pairwise']['continuous_mean_delta'], 1.)
        steered = result['secondary_success_head']['all_frames:target:8']['full']
        self.assertAlmostEqual(steered['mae'], 1 / 3)
        self.assertEqual(steered['baseline_fallback'], 1)
        self.assertEqual(steered['accuracy']['0.125/0.875']['all']['rate_all_expected'], 2 / 3)
        self.assertEqual(steered['accuracy']['0.2/0.8']['all']['rate_all_expected'], 1.)
        self.assertEqual(steered['accuracy']['0.3/0.7']['all']['rate_all_expected'], 1.)
        self.assertEqual(result['conditions']['baseline']['full']['accuracy']['0.3/0.7']['all']['correct'], 0)
        self.assertEqual(success['holdout']['by_task']['task']['accuracy']['0.3/0.7']['all']['correct'], 2)
        self.assertEqual(result['secondary_success_head']['all_frames:target:8']['applied_only']['accuracy']['0.3/0.7']['all']['correct'], 2)
        matched = result['secondary_success_head_matched_controls']['all_frames:8']
        self.assertEqual(matched['example_ids'], ['s'])
        self.assertEqual(matched['conditions']['all_frames:target:8']['accuracy']['0.3/0.7']['all']['correct'], 1)
        self.assertIn('0.3/0.7', result['matched_controls']['all_frames:8']['conditions']['baseline']['accuracy'])

    def test_endpoint_boundaries_and_missing_rows_keep_expected_denominator(self):
        labels = {eid: {'split': split, 'reward': 5 if split == 'suc' else 1}
                  for eid, split in [('s1', 'suc'), ('s2', 'suc'), ('bad', 'suc'),
                                     ('f1', 'fail'), ('f2', 'fail'), ('missing', 'fail')]}
        rows = {eid: {'status': 'ok', 'progress': value} for eid, value in [
            ('s1', .7), ('s2', math.nextafter(.7, 0)), ('f1', .3), ('f2', math.nextafter(.3, 1))]}
        rows['bad'] = {'status': 'parse_error', 'progress': 1.}
        result = endpoint_accuracy(rows, labels, list(labels))
        self.assertEqual((result['all']['correct'], result['all']['valid'], result['all']['expected']), (2, 4, 6))
        self.assertEqual(result['all']['rate_all_expected'], 1 / 3)
        self.assertEqual(result['balanced_accuracy'], .5)
        for split in ('suc', 'fail'):
            self.assertEqual(result[split]['rate_all_expected'], 1 / 3)
        empty = endpoint_accuracy({}, labels, list(labels))
        self.assertEqual(empty['all']['rate_all_expected'], 0.)
        self.assertIsNone(empty['all']['rate_valid'])

    def test_success_fallback_must_equal_success_baseline(self):
        path = self.predictions / 'all_frames_target_8.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]['success_probability'] = .4
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, 'Success-head fallback differs'):
            score_experiment(self.cfg)

    def test_missing_and_invalid_success_do_not_borrow_valid_progress(self):
        for value in [None, True, float('nan'), float('inf'), -.1, 1.1]:
            with self.subTest(value=value):
                raw = {'e': {'status': 'ok', 'progress': 1., 'success_probability': value}}
                row = success_readout(raw)['e']
                self.assertIsNone(row['progress'])
                self.assertEqual(row['status'], 'invalid_success_head')
                self.assertEqual(raw['e']['progress'], 1.)
        row = success_readout({'e': {'status': 'control_unavailable', 'success_probability': 1.}})['e']
        self.assertEqual(row['status'], 'control_unavailable')

    def test_reports_include_both_heads_and_separate_winners(self):
        result = score_experiment(self.cfg)
        destination = self.root / 'report'
        write_reports(destination, [self.cfg], {'meter_official': result}, {}, {'configs': ['meter_official']})
        with (destination / 'metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertEqual({r['head'] for r in rows}, {'progress', 'success'})
        baseline_success = next(r for r in rows if r['head'] == 'success' and r['condition'] == 'baseline')
        self.assertEqual(float(baseline_success['accuracy_03_07']), 2 / 3)
        self.assertEqual(float(baseline_success['suc_accuracy_03_07']), 1.)
        self.assertEqual(float(baseline_success['fail_accuracy_03_07']), .5)
        text = (destination / 'meter_summary.md').read_text()
        self.assertIn('## progress head', text)
        self.assertIn('## success head', text)
        self.assertIn('| meter_official | success | baseline | 2/3 | 0.0000 |', text)
        index = json.loads((destination / 'index.json').read_text())
        self.assertTrue(index['complete'])
        self.assertFalse(index['all_predictions_valid'])


if __name__ == '__main__':
    unittest.main()
