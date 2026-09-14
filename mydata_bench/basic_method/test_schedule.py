"""Resource queue behavior without launching models or touching experiment results."""
import contextlib
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import schedule


def state(index=0, *, free=80000, util=0, pids=()):
    return schedule.GPUState(index, f'GPU-test-{index}', free, 81920, util, list(pids))


def config(name='grm_official'):
    model, protocol = name.split('_', 1)
    return f'/configs/{name}.yaml', {'model': model, 'protocol': protocol,
            'scopes': ['last_frame', 'all_frames'], 'top_k': [8, 32, 64],
            'controls': ['wrong_region', 'low_rank']}


def sharing_args():
    return SimpleNamespace(jobs_per_gpu=2, job_memory_mib=24576, memory_headroom_mib=8192,
                           min_free_memory_mib=61440, max_utilization=10)


def active_job(pid=101, exclusive=False):
    return {'process': SimpleNamespace(pid=pid), 'exclusive': exclusive, 'reserved_mib': 24576}


class GPUContracts(unittest.TestCase):
    def test_processes_match_uuid_instead_of_query_order(self):
        outputs = ['2, GPU-b, 70000, 81920, 0\n0, GPU-a, 81000, 81920, 1\n',
                   'GPU-b, 123\nGPU-b, 456\n']
        with patch.object(schedule.subprocess, 'run', side_effect=[SimpleNamespace(stdout=s) for s in outputs]) as query:
            snapshot = schedule.gpu_snapshot()
        self.assertEqual(snapshot[2].compute_pids, [123, 456])
        self.assertEqual(snapshot[0].compute_pids, [])
        self.assertTrue(all(call.kwargs['timeout'] == 10 and call.kwargs['check'] for call in query.call_args_list))

    def test_unknown_status_fails_closed(self):
        for output in ('0, GPU-a, [N/A], 81920, 0\n', '', '0, GPU-a, 90000, 81920, 0\n'):
            with self.subTest(output=output), patch.object(schedule.subprocess, 'run', return_value=SimpleNamespace(stdout=output)):
                with self.assertRaises(schedule.GPUProbeError):
                    schedule.gpu_snapshot()
        with patch.object(schedule.subprocess, 'run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 10)):
            with self.assertRaises(schedule.GPUProbeError):
                schedule.gpu_snapshot()

    def test_idle_utilization_does_not_override_other_compute_process(self):
        self.assertTrue(schedule.unavailable_reasons(state(pids=[123]), 61440, 10))
        self.assertTrue(schedule.unavailable_reasons(state(free=60000), 61440, 10))
        self.assertTrue(schedule.unavailable_reasons(state(util=80), 61440, 10))
        self.assertEqual(schedule.unavailable_reasons(state(), 61440, 10), [])

    def test_long_sole_configs_first_preserving_remaining_order_and_membership(self):
        names = ['grm_official', 'sole_image_text', 'meter_official', 'sole_official', 'sole_text_image']
        ordered = schedule.prioritize([config(n) for n in names])
        self.assertEqual([Path(p).stem for p, _ in ordered],
                         ['sole_official', 'sole_image_text', 'sole_text_image', 'grm_official', 'meter_official'])

    def test_sharing_accepts_own_busy_worker_but_rejects_external_process(self):
        args, cfg, jobs = sharing_args(), config()[1], [active_job()]
        self.assertEqual(schedule.admission_reasons(state(pids=[101], free=60000, util=95), jobs, cfg, args), [])
        with patch.object(schedule.os, 'getpgid', return_value=999):
            reasons = schedule.admission_reasons(state(pids=[101, 888]), jobs, cfg, args)
        self.assertIn('external compute processes', reasons[0])

    def test_sharing_recognizes_own_child_group_and_fails_closed_on_unknown_pid(self):
        args, cfg, jobs = sharing_args(), config()[1], [active_job()]
        with patch.object(schedule.os, 'getpgid', return_value=101):
            self.assertEqual(schedule.admission_reasons(state(pids=[102]), jobs, cfg, args), [])
        with patch.object(schedule.os, 'getpgid', side_effect=ProcessLookupError):
            self.assertTrue(schedule.admission_reasons(state(pids=[102]), jobs, cfg, args))

    def test_loading_worker_reservation_counts_before_cuda_context_exists(self):
        args, cfg, jobs = sharing_args(), config()[1], [active_job()]
        # The next worker's 24 GiB alone fits; its full sibling reservation and
        # 8 GiB headroom must fit too, despite no process in nvidia-smi yet.
        self.assertTrue(schedule.admission_reasons(state(free=56000), jobs, cfg, args))
        self.assertEqual(schedule.admission_reasons(state(free=57344), jobs, cfg, args), [])

    def test_slot_limit_and_unprofiled_configs_remain_exclusive(self):
        args, cfg, jobs = sharing_args(), config()[1], [active_job()]
        self.assertTrue(schedule.admission_reasons(state(), jobs + [active_job(102)], cfg, args))
        for name in ('sole_official', 'sole_image_text', 'qwen_official'):
            with self.subTest(config=name):
                self.assertTrue(schedule.admission_reasons(state(), jobs, config(name)[1], args))
        self.assertTrue(schedule.admission_reasons(state(), [active_job(exclusive=True)], cfg, args))
        self.assertEqual(schedule.admission_reasons(state(), [], config('sole_official')[1], args), [])

    def test_empty_gpu_still_checks_utilization_and_compute_processes_in_sharing_mode(self):
        args, cfg = sharing_args(), config()[1]
        self.assertTrue(schedule.admission_reasons(state(util=95), [], cfg, args))
        self.assertTrue(schedule.admission_reasons(state(pids=[101]), [], cfg, args))


class QueueContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.argv = ['--root', str(self.root), '--gpus', '0', '--poll-seconds', '1']
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.select = self.stack.enter_context(patch.object(schedule, 'select_configs', return_value=[config()]))
        self.probe = self.stack.enter_context(patch.object(schedule, 'gpu_snapshot', return_value={0: state()}))
        self.sleep = self.stack.enter_context(patch.object(schedule.time, 'sleep'))
        self.proc = Mock(pid=98765, returncode=0)
        self.proc.poll.return_value = 0
        self.launch = self.stack.enter_context(patch.object(schedule.subprocess, 'Popen', return_value=self.proc))
        self.kill = self.stack.enter_context(patch.object(schedule.os, 'killpg'))
        self.output = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def events(self):
        return [json.loads(line) for p in (self.root / 'scheduler').glob('*.jsonl') for line in p.read_text().splitlines()]

    def test_busy_gpu_and_probe_failure_wait_then_require_consecutive_idle_checks(self):
        self.probe.side_effect = [
            {0: state(pids=[123])}, {0: state()}, schedule.GPUProbeError('temporarily unavailable'),
            {0: state()}, {0: state()}]

        def launch(*args, **kwargs):
            self.assertEqual(self.probe.call_count, 5)
            self.assertEqual(self.sleep.call_count, 4)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-test-0')
            return self.proc

        self.launch.side_effect = launch
        schedule.main(self.argv)
        self.assertEqual(self.launch.call_count, 1)
        self.kill.assert_not_called()
        self.assertEqual(self.events()[-1]['failed'], [])

    def test_busy_first_gpu_does_not_block_idle_second_gpu(self):
        self.probe.return_value = {0: state(pids=[123]), 1: state(1)}
        schedule.main(['--root', str(self.root), '--gpus', '0', '1', '--poll-seconds', '1'])
        self.assertEqual(self.launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-test-1')
        self.assertEqual([e['gpu'] for e in self.events() if e['event'] == 'started'], [1])

    def test_running_slot_stays_reserved_even_when_nvidia_smi_reports_idle(self):
        self.select.return_value = [config(), config('meter_official')]
        first, second = Mock(pid=101), Mock(pid=102)
        first.poll.side_effect = [None, None, 0]
        second.poll.return_value = 0
        self.launch.side_effect = [first, second]
        schedule.main(self.argv)
        events = [e for e in self.events() if e['event'] in ('started', 'finished')]
        self.assertEqual([e['event'] for e in events], ['started', 'finished', 'started', 'finished'])
        self.assertEqual(self.probe.call_count, 4)

    def test_worker_failure_is_not_retried_or_reported_as_success(self):
        self.select.return_value = [config(), config('meter_official')]
        first = Mock(pid=101)
        first.poll.return_value = 1
        self.launch.side_effect = [first, self.proc]
        with self.assertRaises(SystemExit) as exc:
            schedule.main(self.argv)
        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(self.launch.call_count, 2)
        self.assertEqual(self.events()[-1]['failed'], ['/configs/grm_official.yaml'])

    def test_spawn_error_does_not_block_remaining_queue(self):
        self.select.return_value = [config(), config('meter_official')]
        self.launch.side_effect = [OSError('failed to spawn'), self.proc]
        with self.assertRaises(SystemExit) as exc:
            schedule.main(self.argv)
        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(self.events()[-1]['failed'], ['/configs/grm_official.yaml'])

    def test_dry_run_preserves_filter_and_writes_nothing(self):
        schedule.main(self.argv + ['--models', 'grm', '--exclude-configs', 'qwen_official', '--dry-run'])
        self.select.assert_called_once_with(self.root, ['grm'], ['qwen_official'])
        self.assertEqual(list(self.root.iterdir()), [])
        self.launch.assert_not_called()
        output = json.loads(self.output.getvalue())
        self.assertEqual(output['baseline_conditions'], 1)
        self.assertEqual(output['steering_conditions'], 18)
        self.assertEqual(output['gpus']['0']['unavailable_reasons'], [])

    def test_invalid_gpu_or_impossible_memory_requirement_stops_before_launch(self):
        for argv in (['--gpus', '7'], ['--min-free-memory-mib', '100000']):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                schedule.main(self.argv + argv)
        self.launch.assert_not_called()

    def test_interrupt_stops_only_own_worker_and_exits_nonzero(self):
        self.proc.poll.return_value = None

        def interrupt(seconds):
            if self.launch.called:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        self.sleep.side_effect = interrupt
        old_handler = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(SystemExit) as exc:
            schedule.main(self.argv)
        self.assertEqual(exc.exception.code, 130)
        self.kill.assert_called_once_with(self.proc.pid, signal.SIGTERM)
        self.proc.wait.assert_called_once_with(timeout=20)
        self.assertIs(signal.getsignal(signal.SIGTERM), old_handler)
        self.assertTrue(self.events()[-1]['interrupted'])

    def test_two_workers_overlap_and_external_process_delays_second_launch(self):
        self.select.return_value = [config(), config('meter_official')]
        first, second = Mock(pid=101), Mock(pid=102)
        first.poll.side_effect = lambda: 0 if self.launch.call_count >= 2 else None
        second.poll.return_value = 0
        self.probe.side_effect = [
            {0: state()}, {0: state()},
            {0: state(pids=[101, 999], util=95)}, {0: state(pids=[101, 999], util=95)},
            {0: state(pids=[101], util=95)}, {0: state(pids=[101], util=95)}]

        def launch(*a, **kwargs):
            if self.launch.call_count == 1:
                return first
            self.assertEqual(self.probe.call_count, 6)
            return second

        self.launch.side_effect = launch
        with patch.object(schedule.os, 'getpgid', return_value=999):
            schedule.main(self.argv + ['--jobs-per-gpu', '2'])
        events = [e for e in self.events() if e['event'] in ('started', 'finished')]
        self.assertEqual([e['event'] for e in events], ['started', 'started', 'finished', 'finished'])
        self.assertEqual({e['gpu'] for e in events}, {0})
        self.assertEqual({e['pid'] for e in events}, {101, 102})
        self.kill.assert_not_called()

    def test_sharing_can_skip_exclusive_pending_config_until_card_is_empty(self):
        self.select.return_value = [config(), config('qwen_official'), config('meter_official')]
        first, second, third = Mock(pid=101), Mock(pid=102), Mock(pid=103)
        first.poll.side_effect = lambda: 0 if self.launch.call_count >= 2 else None
        second.poll.return_value = third.poll.return_value = 0
        self.launch.side_effect = [first, second, third]
        schedule.main(self.argv + ['--jobs-per-gpu', '2'])
        starts = [e for e in self.events() if e['event'] == 'started']
        self.assertEqual([Path(e['config']).stem for e in starts], ['grm_official', 'meter_official', 'qwen_official'])
        events = [e['event'] for e in self.events() if e['event'] in ('started', 'finished')]
        self.assertEqual(events, ['started', 'started', 'finished', 'finished', 'started', 'finished'])

    def test_interrupt_reaps_both_shared_workers(self):
        self.select.return_value = [config(), config('meter_official')]
        processes = [Mock(pid=101, returncode=-15), Mock(pid=102, returncode=-15)]
        for proc in processes:
            proc.poll.return_value = None
        self.launch.side_effect = processes

        def interrupt(seconds):
            if self.launch.call_count == 2:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        self.sleep.side_effect = interrupt
        with self.assertRaises(SystemExit) as exc:
            schedule.main(self.argv + ['--jobs-per-gpu', '2'])
        self.assertEqual(exc.exception.code, 130)
        self.assertEqual({call.args for call in self.kill.call_args_list}, {(101, signal.SIGTERM), (102, signal.SIGTERM)})
        for proc in processes:
            proc.wait.assert_called_once_with(timeout=20)


if __name__ == '__main__':
    unittest.main()
