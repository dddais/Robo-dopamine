"""SOLE regression checks: exact decimal feedback, rollout isolation and processor parity.

All checks run on CPU. The processor parity checks use local tokenizer/processor
files only and skip when those files are not installed; no model is loaded.
"""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize

from .protocol import SOURCE, SYSTEM_PROMPT, OFFICIAL_QUESTION, create_composite_frame, parse_progress
from .versioning import SOLE_PROTOCOL_VERSION, validate_output_directory, validate_records
from mydata_bench.addbase_eval.protocols import prompt_payload, METER_PROMPT, PROG
from mydata_bench.addbase_eval.runtime import Runtime
from mydata_bench.addbase_eval.run import _predict_condition, rank
from mydata_bench.addbase_eval import score


PROCESSOR = Path(os.environ.get('SOLE_PROCESSOR_PATH', '/home/dais/workspace/model/SOLE-R1-8B'))


def config():
    return {'model': 'sole', 'protocol': 'official', 'sole_protocol_version': SOLE_PROTOCOL_VERSION,
            'batch_size': 2, 'scopes': ['last_frame', 'all_frames'], 'skip_early_layers': 8,
            'min_pixels': 1024, 'max_pixels': 12845056}


class DecimalFeedbackTests(unittest.TestCase):
    def test_answers_preserve_decimal_feedback_without_float_round_trip(self):
        for answer, expected in [('7', '7'), ('57', '57'), ('55.125', '55.125'),
                                 ('+070.00', '+070.00'), ('-0.0', '-0.0'), ('-25', '-25'),
                                 ('101', '100'), ('-120', '-100')]:
            with self.subTest(answer=answer):
                row = parse_progress(f'<think>reasoning</think><answer>{answer}%</answer>')
                self.assertEqual(row['status'], 'ok')
                self.assertEqual(row['percentage_text'], expected)
                self.assertEqual(row['raw_percentage_text'], answer)
                self.assertEqual(row['progress'], float(expected) / 100)

    def test_invalid_answers_stay_invalid(self):
        for raw in ['<answer>57', '<answer>nan</answer>', '<answer>1e3</answer>',
                    '<answer>7</answer><answer>8</answer>', '<answer>' + '9'*400 + '</answer>']:
            with self.subTest(raw=raw[:50]):
                self.assertEqual(parse_progress(raw)['status'], 'parse_error')
                self.assertIsNone(parse_progress(raw)['progress'])


class FakeRuntime:
    cfg = config()

    def __init__(self):
        self.calls = []

    def predict(self, samples, condition, rankings, step, previous):
        self.calls.append((condition, step, {s['example_id']:p for s,p in zip(samples, previous)}))
        answers = ['7', '57', '55.125', '-25', '101', '57', '70'] if condition == 'baseline' else ['14']*7
        rows = []
        for s, p in zip(samples, previous):
            raw = '<answer>broken' if s['example_id'] == 'bad' and step == 3 else f'<answer>{answers[step-1]}%</answer>'
            rows.append({'example_id':s['example_id'], 'condition':condition, 'step':step,
                         'previous_percentage':float(p), 'previous_percentage_text':p, **parse_progress(raw)})
        return rows

    def collect(self, samples, step, previous):
        self.ranking_previous = previous
        return [{'example_id':s['example_id'], 'status':'ok', 'sole_protocol_version':SOLE_PROTOCOL_VERSION,
                 'query_kind':'last_prompt', 'raw_mass':{scope:np.zeros((36,32)).tolist() for scope in self.cfg['scopes']}}
                for s in samples]


class RolloutTests(unittest.TestCase):
    def test_own_history_clipping_failure_isolation_and_terminal_ranking(self):
        runtime = FakeRuntime()
        samples = [{'example_id':s} for s in ['a', 'b', 'bad']]
        rankings = {'last_frame': {'sole_protocol_version':SOLE_PROTOCOL_VERSION}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            base = _predict_condition(runtime, samples, 'baseline', None, output)
            target = _predict_condition(runtime, samples, 'last_frame:target:8', rankings, output)
            self.assertEqual(base['a']['progress_curve'], [0, .07, .57, .55125, -.25, 1., .57, .7])
            self.assertEqual(target['a']['progress_curve'], [0]+[.14]*7)
            self.assertEqual(base['bad']['status'], 'incomplete_rollout')
            self.assertIsNone(base['bad']['progress'])
            calls = {}
            for c,step,p in runtime.calls:
                calls.setdefault((c,step), {}).update(p)
            self.assertEqual(calls['baseline', 3]['a'], '57')
            self.assertEqual(calls['baseline', 6]['a'], '100')
            self.assertEqual(calls['last_frame:target:8', 2]['a'], '14')
            self.assertNotIn('bad', calls['baseline', 4])
            ranking = rank(runtime, samples, base, output)
            self.assertEqual(runtime.ranking_previous, ['57', '57'])
            validate_records(runtime.cfg, ranking.values(), 'test ranking')

    def test_old_predictions_and_rankings_cannot_be_reused(self):
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'predictions').mkdir()
            (folder/'predictions/baseline.jsonl').write_text(json.dumps({'example_id':'a','status':'ok','progress':.5})+'\n')
            with self.assertRaisesRegex(ValueError, 'Incompatible SOLE cache'):
                _predict_condition(runtime, [{'example_id':'a'}], 'baseline', None, folder)
            with self.assertRaisesRegex(ValueError, 'Incompatible SOLE cache'):
                _predict_condition(runtime, [{'example_id':'a'}], 'last_frame:target:8', {'last_frame':{}}, folder)
            self.assertEqual(runtime.calls, [])


class RuntimeOutputTests(unittest.TestCase):
    def test_generation_keeps_text_feedback_and_explicit_greedy_settings(self):
        runtime = Runtime.__new__(Runtime)
        runtime.cfg = dict(config(), max_new_tokens=512)
        runtime.prepare = Mock(return_value=({'input_ids':torch.tensor([[1,2]])},[{}],[1],['prompt']))
        runtime.model = SimpleNamespace(generate=Mock(return_value=torch.tensor([[1,2,3,4,5]])))
        runtime.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0),
                                           batch_decode=Mock(return_value=['<answer>57.00%</answer>']))
        runtime.controller = SimpleNamespace(clear=Mock())
        row = runtime.predict([{'example_id':'a'}], 'baseline', step=7, previous=['7'])[0]
        self.assertEqual(row['percentage_text'], '57.00')
        self.assertEqual(row['previous_percentage_text'], '7')
        self.assertEqual(row['previous_percentage'], 7.)
        self.assertEqual(row['sole_protocol_version'], SOLE_PROTOCOL_VERSION)
        kwargs = runtime.model.generate.call_args.kwargs
        self.assertFalse(kwargs['do_sample'])
        self.assertEqual(kwargs['max_new_tokens'], 512)
        for key in ['temperature','top_p','top_k']:
            self.assertIsNone(kwargs[key])
        runtime.controller.clear.assert_called_once()


class VersionTests(unittest.TestCase):
    def test_only_canonical_official_config_exists_and_is_corrected(self):
        import yaml
        from .versioning import validate_protocol_config
        folder = Path(__file__).resolve().parents[1]/'configs/v2_crossmodel_addbase'
        self.assertEqual(sorted(p.name for p in folder.glob('sole_official*.yaml')), ['sole_official.yaml'])
        cfg = yaml.safe_load((folder/'sole_official.yaml').read_text())
        validate_protocol_config(cfg)
        self.assertEqual(Path(cfg['output_dir']).name, 'sole_official')

    def test_legacy_config_rejected_before_loading_processor_or_model(self):
        cfg = config(); del cfg['sole_protocol_version']
        with patch('mydata_bench.addbase_eval.runtime.AutoProcessor.from_pretrained') as load:
            with self.assertRaisesRegex(ValueError, 'sole_official.yaml'):
                Runtime(cfg)
            load.assert_not_called()

    def test_output_directory_guard_preserves_historical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            validate_output_directory(config(), folder)
            historical = json.dumps({'model':'sole','protocol':'official'})
            (folder/'run_config.json').write_text(historical)
            with self.assertRaisesRegex(ValueError, 'historical results'):
                validate_output_directory(config(), folder)
            self.assertEqual((folder/'run_config.json').read_text(), historical)
            (folder/'run_config.json').unlink()
            (folder/'ranking_last_frame.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unversioned artifacts'):
                validate_output_directory(config(), folder)


class ScoringTests(unittest.TestCase):
    def test_reporting_rejects_obsolete_sole_analysis(self):
        from .versioning import load_analysis
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.json'
            path.write_text(json.dumps({'config':{'model':'sole','protocol':'official'}}))
            with self.assertRaisesRegex(ValueError, 'sole_official.yaml'):
                load_analysis(path)
            path.write_text(json.dumps({'config':config(),'conditions':{}}))
            self.assertEqual(load_analysis(path)['conditions'], {})

    def test_obsolete_unversioned_run_is_no_longer_scoreable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'run_config.json').write_text(json.dumps({'model':'sole','protocol':'official'}))
            with self.assertRaisesRegex(ValueError, 'sole_official.yaml'):
                score.score_experiment(folder, [], {}, folder)

    def test_explicit_corrected_matrix_is_scored_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'sole_official'; folder.mkdir()
            cfg = dict(config(), output_dir=str(folder), top_k=[8,32,64])
            cfg_path = root/'sole_official.yaml'
            # JSON is a YAML subset and keeps the temporary fixture minimal.
            cfg_path.write_text(json.dumps(cfg))
            (folder/'run_config.json').write_text(json.dumps(cfg))
            (root/'inputs.json').write_text('[]')
            (root/'labels_for_scoring_only.json').write_text('{}')
            argv = ['score','--configs',str(cfg_path),'--output-name','analysis_v2']
            with patch.object(score,'OUT',root), patch('sys.argv',argv), \
                 patch.object(score,'score_experiment',return_value={'conditions':{}}) as run:
                score.main()
            self.assertEqual(run.call_args.args[0], folder)
            holm = json.loads((root/'analysis_v2/holm.json').read_text())
            self.assertEqual(holm['cohort']['family_size'], 6)
            self.assertTrue(all(v==1. for v in holm['cohort']['adjusted_p'].values()))
            index = json.loads((root/'analysis_v2/index.json').read_text())
            self.assertEqual(index['config_matrix'], [str(cfg_path)])

    def test_corrected_analysis_rejects_unversioned_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/'predictions').mkdir()
            (folder/'run_config.json').write_text(json.dumps(config()))
            (folder/'predictions/baseline.jsonl').write_text(json.dumps({'example_id':'a','condition':'baseline','status':'ok','progress':.5})+'\n')
            with self.assertRaisesRegex(ValueError, 'Incompatible SOLE cache'):
                score.score_experiment(folder, [], {}, folder)


@unittest.skipUnless((PROCESSOR/'preprocessor_config.json').exists(), 'Local SOLE processor files unavailable')
class OfficialProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
        cls.processor.tokenizer.padding_side = 'left'
        y,x = np.indices((480,640))
        cls.frames = tuple(np.stack([(x+i*17)%256,(y+i*31)%256,(x+y+i*13)%256],axis=-1).astype(np.uint8) for i in range(8))
        # Use the official conversation function independently of prompt_payload.
        nodes = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='make_conversation_image']
        ns = {'json':json, 'system_prompt_template':SYSTEM_PROMPT, 'question_template':'{question}', 'problem_key':'problem'}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
        cls.reference_conversation = staticmethod(ns['make_conversation_image'])

    def test_batched_inputs_match_official_processor_path_exactly(self):
        runtime = Runtime.__new__(Runtime)
        runtime.cfg = config(); runtime.processor = self.processor
        runtime.model = SimpleNamespace(device='cpu', dtype=torch.float32)
        samples = [{'example_id':str(i), 'image_paths':['unused']*8, 'task':task,
                    'sampling':{'selected_source_indices':list(range(8))}, 'cohort':True, 'tracking_path':'unused'}
                   for i,task in enumerate(['pick up the cup', 'move the red cup to the plate on the left'])]
        previous = ['57', '7']
        with patch('mydata_bench.addbase_eval.protocols.read_frames', return_value=self.frames), \
             patch('mydata_bench.addbase_eval.protocols.read_track', return_value={i:[100,100,180,180] for i in range(8)}):
            actual,maps,queries,texts = runtime.prepare(samples,7,previous)
        composite = create_composite_frame(None,self.frames[0],None,self.frames[6],None,self.frames[7],view_type='external')
        h,w = smart_resize(*composite.shape[:2],factor=28,min_pixels=3136,max_pixels=12845056)
        image = Image.fromarray(composite).resize((w,h))
        expected_texts = [self.processor.apply_chat_template(json.loads(self.reference_conversation(
            {'problem':OFFICIAL_QUESTION.format(task_description=s['task'],prev_progress=p)})),
            tokenize=False,add_generation_prompt=True) for s,p in zip(samples,previous)]
        expected = self.processor(text=expected_texts,images=[image,image],padding=True,
                                  return_tensors='pt',add_special_tokens=False)
        self.assertEqual(texts, expected_texts)
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            self.assertTrue(torch.equal(actual[key],expected[key]), key)
        for m,q in zip(maps,queries):
            self.assertEqual(q, actual['input_ids'].shape[1]-1)
            self.assertTrue(set(m['target']['last_frame']) <= set(m['target']['all_frames']))
            self.assertTrue(set(m['target']['all_frames']).isdisjoint(m['negative']['all_frames']))
        self.assertTrue((actual['attention_mask'][0] == 0).any())

    def test_meter_and_ordinary_sole_keep_their_protocols(self):
        sample = {'image_paths':['unused']*8, 'sampling':{'selected_source_indices':list(range(8))}, 'task':'pick up the cup'}
        with patch('mydata_bench.addbase_eval.protocols.read_frames', return_value=self.frames):
            meter = prompt_payload(sample, dict(config(),model='meter'))
            sole = prompt_payload(sample, dict(config(),protocol='text_image',max_pixels=50176))
        self.assertEqual(meter['processor_kwargs'], {})
        self.assertEqual([image.size for image in meter['images']], [(640,480)]*8)
        content = meter['messages'][0]['content']
        self.assertEqual(content[0]['text'], METER_PROMPT.format(task=sample['task']))
        self.assertEqual([item['text'] for item in content[1:] if item['type']=='text'], [PROG]*8)
        self.assertEqual(sole['processor_kwargs'], {})
        self.assertEqual(len(sole['images']), 8)
        self.assertIn('final timestep', sole['messages'][-1]['content'][0]['text'])


if __name__ == '__main__':
    unittest.main()
