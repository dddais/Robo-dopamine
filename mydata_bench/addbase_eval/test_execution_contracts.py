"""Independent numerical and upstream-input regression tests (CPU only)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import AttentionController
from .protocols import SPECIAL_TOKENS, PROG
from .runtime import Runtime
from .score import summary, paired_change, score_experiment
from .run import rank
from .verify_execution import official_meter_inputs, assert_inputs_equal
from mydata_bench.meter_eval.model import RobometerModel
from mydata_bench.top_eval.protocol import parse_progress


class ExceptionalOutputs(unittest.TestCase):
    def test_mixed_valid_and_invalid_answer_tags_are_not_accepted(self):
        for raw in ['<answer>bad</answer><answer>50%</answer>',
                    '<answer>50%</answer><answer>unfinished',
                    '<answer><answer>50%</answer>', '<answer>50%</answer></answer>']:
            with self.subTest(raw=raw):
                self.assertEqual(parse_progress(raw)['status'], 'parse_error')

    def test_nonfinite_missing_and_nonnumeric_outputs_never_become_correct_failures(self):
        labels = {str(i):{'split':'fail','reward':1,'source_suc_id':None,'video_sha256':str(i)} for i in range(7)}
        values = [0.,float('nan'),float('inf'),float('-inf'),None,'0',False]
        rows = {str(i):{'status':'ok','progress':p} for i,p in enumerate(values)}
        result = summary(rows,labels,list(labels))
        self.assertEqual(result['n'],1)
        self.assertEqual(result['invalid'],6)
        self.assertEqual(result['accuracy']['0.125/0.875']['all']['rate_all_expected'],1/7)
        base = {k:{'status':'ok','progress':1.} for k in labels}
        change = paired_change(base,rows,labels,list(labels))
        self.assertEqual(change['n'],1)
        self.assertFalse(change['complete'])

    def test_typed_meter_logits_that_sum_to_one_still_use_softmax(self):
        logits = torch.tensor([[0.,0.,0.,0.,0.,0.,0.,0.,-2.,3.]])
        model = SimpleNamespace(progress_head=lambda h:logits,success_head=lambda h:torch.tensor([[2.]]))
        progress,success,positions = RobometerModel.read_progress(model,torch.zeros(1,3,4),torch.tensor([[0,7,0]]),7)
        expected = (logits.softmax(-1)*torch.linspace(0,1,10)).sum().item()
        self.assertEqual(progress,[[expected]])
        self.assertEqual(positions,[[1]])
        self.assertEqual(success,[[torch.sigmoid(torch.tensor(2.)).item()]])
        self.assertLess(progress[0][0],1.)

    def test_previous_length_is_checked_before_preprocessing(self):
        runtime = Runtime.__new__(Runtime)
        with self.assertRaisesRegex(ValueError,'batch length'):
            runtime.prepare([{},{}],previous=['0'])

    def test_meter_rejects_nonfinite_logits_before_writing_success(self):
        for field in ['progress_head','success_head']:
            model = SimpleNamespace(progress_head=lambda h:torch.zeros(1,10),success_head=lambda h:torch.zeros(1,1))
            setattr(model,field,lambda h:torch.tensor([[float('nan')]]))
            with self.assertRaisesRegex(ValueError,'Non-finite'):
                RobometerModel.read_progress(model,torch.zeros(1,1,4),torch.tensor([[7]]),7)

    def test_ranking_rejects_same_count_but_wrong_cached_ids(self):
        runtime = SimpleNamespace(cfg={'model':'meter','protocol':'official'})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'ranking_observations.jsonl').write_text(json.dumps({'example_id':'wrong','status':'ok'})+'\n')
            with self.assertRaisesRegex(ValueError,'outside the requested ranking set'):
                rank(runtime,[{'example_id':'expected'}],{},folder)

    def test_scoring_rejects_another_input_manifest_and_mixed_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/'predictions').mkdir()
            manifest = folder/'inputs.json'; manifest.write_text('[{"example_id":"different"}]')
            config = {'model':'meter','protocol':'official','inputs':str(manifest)}
            (folder/'run_config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Scoring inputs differ'):
                score_experiment(folder,[],{},folder)
            manifest.write_text('[]')
            (folder/'predictions/baseline.jsonl').write_text('\n'.join(json.dumps(
                {'example_id':str(i),'condition':c,'status':'ok','progress':0.}) for i,c in enumerate(['baseline','last_frame:target:8'])))
            with self.assertRaisesRegex(ValueError,'Mixed prediction conditions'):
                score_experiment(folder,[],{},folder)


class BatchedAttention(unittest.TestCase):
    def setUp(self):
        self.original = ALL_ATTENTION_FUNCTIONS['sdpa']
        self.module = SimpleNamespace(num_key_value_groups=2,is_causal=True)
        self.controller = AttentionController([SimpleNamespace(self_attn=self.module)])
        torch.manual_seed(19)
        self.q = torch.randn(2,4,7,8)
        self.k = torch.randn(2,2,7,8)
        self.v = torch.randn(2,2,7,8)
        self.maps = [{'visual':[1+shift,2+shift,3+shift],
                      'target':{'last_frame':[2+shift]},'negative':{'last_frame':[1+shift,3+shift]}}
                     for shift in [2,0]]
        self.allowed = torch.ones(2,1,7,7,dtype=torch.bool).tril()
        self.allowed[0,:,:,:2] = False  # Left padding in sample 0 only.

    def tearDown(self):
        ALL_ATTENTION_FUNCTIONS.register('sdpa',self.original)

    def expected(self,q,k,v,allowed,bias):
        weights = q @ k.repeat_interleave(2,1).transpose(-1,-2) / np.sqrt(8)
        weights = weights.masked_fill(~allowed,-torch.inf)
        for i,m in enumerate(self.maps):
            weights[i,1,:,[p for p in m['target']['last_frame'] if p<k.shape[2]]] += bias
            weights[i,1,:,[p for p in m['negative']['last_frame'] if p<k.shape[2]]] -= bias
        weights = torch.nan_to_num(weights.softmax(-1))
        return (weights @ v.repeat_interleave(2,1)).transpose(1,2)

    def test_batched_boolean_and_additive_masks_and_zero_bias(self):
        for bias in [0.,6.]:
            for boolean in [True,False]:
                mask = self.allowed if boolean else torch.zeros_like(self.allowed,dtype=torch.float32).masked_fill(~self.allowed,-torch.inf)
                self.controller.steer(self.maps,[{'layer':0,'head':1}],bias,'last_frame')
                actual,_ = self.controller.forward(self.module,self.q,self.k,self.v,mask)
                torch.testing.assert_close(actual,self.expected(self.q,self.k,self.v,self.allowed,bias),rtol=2e-6,atol=2e-6)

    def test_cached_chunk_and_decode_equal_full_causal_prefix(self):
        self.controller.steer(self.maps,[{'layer':0,'head':1}],6.,'last_frame')
        # Prefill once, then grow the KV cache with a chunk and one token.
        for start,end in [(0,5),(5,6),(6,7)]:
            actual,_ = self.controller.forward(self.module,self.q[:,:,start:end],self.k[:,:,:end],
                                                self.v[:,:,:end],self.allowed[:,:,start:end,:end])
            expected = self.expected(self.q[:,:,:end],self.k[:,:,:end],self.v[:,:,:end],self.allowed[:,:,:end,:end],6.)
            torch.testing.assert_close(actual,expected[:,start:end],rtol=2e-6,atol=2e-6)

    def test_batched_ranking_is_observational_and_respects_padding(self):
        maps = [{**m,'target':{'last_frame':m['target']['last_frame'],'all_frames':m['target']['last_frame']}} for m in self.maps]
        state = self.controller.rank(maps,[6,5],num_layers=1,num_heads=4)
        observed,_ = self.controller.forward(self.module,self.q,self.k,self.v,self.allowed)
        expected,_ = self.original(self.module,self.q,self.k,self.v,self.allowed)
        torch.testing.assert_close(observed,expected,rtol=0,atol=0)
        weights = (self.q @ self.k.repeat_interleave(2,1).transpose(-1,-2)/np.sqrt(8)).masked_fill(~self.allowed,-torch.inf).softmax(-1)
        for i,j in enumerate([6,5]):
            np.testing.assert_allclose(state['raw']['last_frame'][i,0],weights[i,:,j][:,maps[i]['target']['last_frame']].sum(-1),rtol=1e-6,atol=1e-7)


METER_PROCESSOR = Path('/home/dais/workspace/model/Qwen3-VL-4B-Instruct')


@unittest.skipUnless((METER_PROCESSOR/'preprocessor_config.json').exists(),'Local Robometer processor unavailable')
class MeterOfficialInputs(unittest.TestCase):
    def test_actual_upstream_collator_matches_runtime_with_left_padding(self):
        processor = AutoProcessor.from_pretrained(str(METER_PROCESSOR),local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        for token in SPECIAL_TOKENS:
            if token not in processor.tokenizer.get_vocab():
                processor.tokenizer.add_special_tokens({'additional_special_tokens':[token]})
        runtime = Runtime.__new__(Runtime)
        runtime.cfg = {'model':'meter','protocol':'official','min_pixels':1024,'max_pixels':16777216}
        runtime.processor = processor
        runtime.prog_id = processor.tokenizer.convert_tokens_to_ids(PROG)
        runtime.model = SimpleNamespace(device='cpu',dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            samples = []
            for i,(w,h) in enumerate([(640,480),(641,481)]):
                y,x = np.indices((h,w))
                array = np.stack([x%256,y%256,(x+y)%256],axis=-1).astype(np.uint8)
                path = Path(tmp)/f'{i}.png'; Image.fromarray(array).save(path)
                samples.append({'image_paths':[str(path)]*8,'task':'pick up the cup' if i==0 else 'move the red cup onto the large plate at the left side',
                                'sampling':{'selected_source_indices':list(range(8))},'cohort':False})
            actual,maps,queries,_ = runtime.prepare(samples)
            assert_inputs_equal(actual,official_meter_inputs(processor,samples))
            for i,q in enumerate(queries):
                self.assertEqual(sum(actual['input_ids'][i]==runtime.prog_id),8)
                self.assertEqual(actual['input_ids'][i,q],runtime.prog_id)
                self.assertEqual(len(maps[i]['visual']),2400)
            self.assertTrue((actual['attention_mask'][0]==0).any())


if __name__=='__main__':
    unittest.main()
