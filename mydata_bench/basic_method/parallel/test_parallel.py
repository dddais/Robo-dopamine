import json
from pathlib import Path
import tempfile
import unittest

from mydata_bench.basic_method.common import append
from mydata_bench.basic_method.grounding import sample_identity
from mydata_bench.basic_method.run import predict_condition
from .core import available_conditions, condition_lock, finish_run, lock_file, strict_rows, verify_manifest


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.execution = self.root / 'execution'
        self.cfg = {'output_dir': str(self.root), 'model': 'sole', 'protocol': 'image_text',
                    'scopes': ['last_frame'], 'controls': [], 'top_k': [8, 32]}
        self.samples = [{'example_id': 'suc/example/1'}, {'example_id': 'fail/example/2'}]

    def rows(self, condition='baseline'):
        return [{'example_id': s['example_id'], 'sample_id': sample_identity(s),
                 'condition': condition, 'run_id': 'original', 'status': 'ok', 'progress': 0.5}
                for s in self.samples]

    def save(self, condition, rows):
        path = self.root / 'predictions' / (condition.replace(':', '_') + '.jsonl')
        append(path, rows)
        return path

    def test_claim_is_exclusive_and_released_without_removing_inode(self):
        path = condition_lock(self.execution, 'last_frame:target:8')
        with lock_file(path) as owner:
            self.assertIsNotNone(owner)
            inode = path.stat().st_ino
            with lock_file(path) as contender:
                self.assertIsNone(contender)
        with lock_file(path) as successor:
            self.assertIsNotNone(successor)
            self.assertEqual(inode, path.stat().st_ino)

    def test_worker_inheritance_keeps_legacy_runner_locked_out(self):
        import subprocess
        import sys
        path = self.root / '.run.lock'
        child = None
        try:
            with lock_file(path) as owner:
                child = subprocess.Popen(
                    [sys.executable, '-c', 'import sys; print("ready", flush=True); sys.stdin.read()'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                    pass_fds=(owner.fileno(),))
                self.assertEqual(child.stdout.readline().strip(), 'ready')
            with lock_file(path) as legacy_runner:
                self.assertIsNone(legacy_runner)
            child.stdin.close()
            child.wait(timeout=5)
            with lock_file(path) as resumed:
                self.assertIsNotNone(resumed)
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdout.close()

    def test_pending_skips_completed_and_claimed_conditions(self):
        self.save('last_frame:target:8', self.rows('last_frame:target:8'))
        self.assertEqual(available_conditions(self.execution, self.cfg, self.samples, 'original'),
                         (['last_frame:target:32'], []))
        with lock_file(condition_lock(self.execution, 'last_frame:target:32')):
            self.assertEqual(available_conditions(self.execution, self.cfg, self.samples, 'original'),
                             ([], ['last_frame:target:32']))

    def test_partial_condition_is_resumable(self):
        self.save('last_frame:target:8', self.rows('last_frame:target:8')[:1])
        pending, _ = available_conditions(self.execution, self.cfg, self.samples, 'original')
        self.assertIn('last_frame:target:8', pending)

    def test_duplicate_rows_are_rejected(self):
        path = self.save('baseline', self.rows() + self.rows()[:1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            strict_rows(path, self.samples, 'original', 'baseline')

    def test_wrong_identity_and_partial_json_are_rejected(self):
        rows = self.rows()
        rows[0]['run_id'] = 'different'
        path = self.save('baseline', rows)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            strict_rows(path, self.samples, 'original', 'baseline')
        path = self.root / 'broken.jsonl'
        path.write_text('{"example_id":')
        with self.assertRaises(json.JSONDecodeError):
            strict_rows(path, self.samples, 'original', 'baseline')

    def test_completion_requires_every_condition(self):
        self.save('baseline', self.rows())
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            finish_run(self.cfg, self.samples, 'original')
        self.assertFalse((self.root / 'completion.json').exists())
        for condition in ['last_frame:target:8', 'last_frame:target:32']:
            self.save(condition, self.rows(condition))
        finish_run(self.cfg, self.samples, 'original')
        finish_run(self.cfg, self.samples, 'original')
        self.assertEqual(json.loads((self.root / 'completion.json').read_text())['examples_per_condition'], 2)

    def test_frozen_predictor_reuses_baseline_and_completed_rows(self):
        from unittest.mock import patch
        condition = 'last_frame:target:8'
        self.save('baseline', self.rows())
        existing = self.rows(condition)[:1]
        path = self.save(condition, existing)
        before = path.read_bytes()
        runtime = type('FakeRuntime', (), {'cfg': {**self.cfg, 'bias': 6}})()
        with patch('mydata_bench.basic_method.run.check_sample', return_value={'eligible': False, 'reason': 'missing_exact_frames'}):
            result = predict_condition(runtime, self.samples, condition, {}, self.root, 'original')
        self.assertTrue(path.read_bytes().startswith(before))
        self.assertEqual(result[self.samples[0]['example_id']], existing[0])
        fallback = result[self.samples[1]['example_id']]
        self.assertTrue(fallback['baseline_fallback'])
        self.assertEqual(fallback['baseline_source']['run_id'], 'original')

    def test_manifest_detects_changed_artifact(self):
        from mydata_bench.basic_method.common import file_hash
        self.execution.mkdir()
        artifact = self.root / 'ranking.json'
        artifact.write_text('{}')
        manifest = {'frozen_artifacts': {str(artifact): file_hash(artifact)}, 'orchestration_sources': {}}
        (self.execution / 'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(verify_manifest(self.execution), manifest)
        artifact.write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_manifest(self.execution)


if __name__ == '__main__':
    unittest.main()
