"""Numerical tests for the intervention and protocol validity boundaries."""
import unittest
import fcntl
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .attention import AttentionController
from .protocols import align_input
from mydata_bench.top_eval.protocol import parse_progress
from .score import summary, pairwise, paired_change
from .run import predict_condition
from .prepare import OUT


class AttentionContract(unittest.TestCase):
    def setUp(self):
        self.original=ALL_ATTENTION_FUNCTIONS['sdpa']
        self.module=SimpleNamespace(num_key_value_groups=2,is_causal=True)
        self.controller=AttentionController([SimpleNamespace(self_attn=self.module)])
        torch.manual_seed(42)
        self.q=torch.randn(1,4,5,8)
        self.k=torch.randn(1,2,5,8)
        self.v=torch.randn(1,2,5,8)
        self.maps=[{'visual':[1,2,3], 'target':{'last_frame':[2],'all_frames':[2]},
                    'negative':{'last_frame':[1,3],'all_frames':[1,3]},
                    'wrong':{'last_frame':[3],'all_frames':[3]}}]

    def tearDown(self):
        ALL_ATTENTION_FUNCTIONS.register('sdpa',self.original)

    def expected(self,q,k,v,mask):
        k=k.repeat_interleave(2,1);v=v.repeat_interleave(2,1)
        w=((q@k.transpose(-1,-2))/(8**.5)+mask).softmax(-1)
        return (w@v).transpose(1,2)

    def test_all_rows_bias_preserves_implicit_causal_mask(self):
        self.controller.steer(self.maps,[{'layer':0,'head':1}],6,'last_frame')
        actual,_=self.controller.forward(self.module,self.q,self.k,self.v,None)
        mask=torch.zeros(1,4,5,5)
        mask.masked_fill_(torch.ones(5,5,dtype=torch.bool).triu(1),-torch.inf)
        mask[:,1,:,2]+=6;mask[:,1,:,[1,3]]-=6
        torch.testing.assert_close(actual,self.expected(self.q,self.k,self.v,mask),atol=2e-6,rtol=2e-6)
        base,_=self.original(self.module,self.q,self.k,self.v,None)
        torch.testing.assert_close(actual[:,:,[0,2,3]],base[:,:,[0,2,3]],atol=2e-6,rtol=2e-6)
        # Query zero cannot see any biased key. Query one can see negative key 1.
        torch.testing.assert_close(actual[:,:1],base[:,:1],atol=2e-6,rtol=2e-6)

    def test_decode_extends_with_zero_text_key_bias(self):
        state=self.controller.steer(self.maps,[{'layer':0,'head':1}],6,'last_frame')
        q=self.q[:,:,-1:]; k=torch.cat([self.k,self.k[:,:,:1]],2);v=torch.cat([self.v,self.v[:,:,:1]],2)
        actual,_=self.controller.forward(self.module,q,k,v,None)
        mask=torch.zeros(1,4,1,6);mask[:,1,:,2]+=6;mask[:,1,:,[1,3]]-=6
        torch.testing.assert_close(actual,self.expected(q,k,v,mask),atol=2e-6,rtol=2e-6)
        self.assertEqual(state['diagnostics']['0']['decode_calls'],1)

    def test_rank_observes_correct_query_and_future_keys_zero(self):
        state=self.controller.rank(self.maps,[1],num_layers=1,num_heads=4)
        self.controller.forward(self.module,self.q,self.k,self.v,None)
        self.assertEqual(state['raw']['last_frame'].sum(),0)
        self.assertEqual(state['seen'],{0})

    def test_rank_mass_has_analytic_uniform_value(self):
        state=self.controller.rank(self.maps,[4],num_layers=1,num_heads=4)
        self.controller.forward(self.module,torch.zeros_like(self.q),self.k,self.v,None)
        np.testing.assert_allclose(state['raw']['last_frame'][0,0],.2,rtol=1e-6)
        np.testing.assert_allclose(state['visual'][0,0],.6,rtol=1e-6)

    def test_padding_mask_preserved(self):
        state=self.controller.steer(self.maps,[{'layer':0,'head':1}],6,'last_frame')
        mask=torch.zeros(1,1,5,5);mask.masked_fill_(torch.ones(5,5,dtype=torch.bool).triu(1),-torch.inf)
        mask[:,:,:,0]=-torch.inf;mask[:,:,0,0]=0
        actual,_=self.controller.forward(self.module,self.q,self.k,self.v,mask)
        expected_mask=mask.expand(1,4,5,5).clone();expected_mask[:,1,:,2]+=6;expected_mask[:,1,:,[1,3]]-=6
        torch.testing.assert_close(actual,self.expected(self.q,self.k,self.v,expected_mask),atol=2e-6,rtol=2e-6)


class OutputContract(unittest.TestCase):
    def test_parse_never_coerces_invalid_output_to_failure(self):
        for raw in ['The answer is 0%', '<answer>5','<answer>5%</answer><answer>10%</answer>','<answer>nan</answer>']:
            self.assertIsNone(parse_progress(raw)['progress'])
        self.assertEqual(parse_progress('<think>x</think><answer>75%</answer>')['progress'],.75)
        self.assertEqual(parse_progress('<answer>-25%</answer>')['progress'],-.25)

    def test_video_tubelet_grid_validation(self):
        sample={'cohort':False}
        payload={'videos':[True], 'span_specs':[{'sources':[0,5],'size':[64,64],'mosaic':False},
                                               {'sources':[10,15],'size':[64,64],'mosaic':False}]}
        ids=[3]+[151656]*4+[7]+[151656]*4+[8]
        result=align_input(ids,[[2,4,4]],payload,sample,{'model':'sole'})
        self.assertEqual(result['records'][1]['sources'],[10,15])
        self.assertEqual(len(result['visual']),8)
        with self.assertRaises(ValueError):align_input(ids,[[2,6,4]],payload,sample,{'model':'sole'})

    def test_unavailable_control_does_not_invalidate_baseline_or_target(self):
        payload={'videos':[], 'span_specs':[{'sources':[0],'size':[64,64],'mosaic':False}]}
        sample={'cohort':True,'tracking_path':'unused'}
        with patch('mydata_bench.addbase_eval.protocols.read_track',return_value={0:[0,0,64,64]}):
            result=align_input([7]+[151655]*4+[8],[[1,4,4]],payload,sample,{'model':'meter'})
        self.assertEqual(result['target']['last_frame'],[1,2,3,4])
        self.assertEqual(result['wrong']['last_frame'],[])
        self.assertIn('last_frame',result['control_unavailable'])

    def test_official_mosaic_target_is_in_current_tile(self):
        payload={'videos':[], 'span_specs':[{'sources':[0,1,2],'size':[1162,384],
                    'source_size':[64,64],'mosaic':True}]}
        sample={'cohort':True,'tracking_path':'unused'}
        with patch('mydata_bench.addbase_eval.protocols.read_track',return_value={i:[32,32,48,48] for i in range(3)}):
            result=align_input([7]+[151655]*444+[8],[[1,24,74]],payload,sample,{'model':'sole'})
        self.assertEqual(result['alignment']['last_frame'][0]['boxes'],[[970.,192.,1066.,288.]])
        self.assertTrue(all((p-1)%37>=30 for p in result['target']['last_frame']))
        self.assertTrue(set(result['target']['last_frame']).isdisjoint(result['wrong']['last_frame']))
        self.assertEqual(len(result['target']['last_frame']),len(result['wrong']['last_frame']))


class ScoringContract(unittest.TestCase):
    def test_labels_pairs_denominators_and_missing_bounds(self):
        labels={'s':{'reward':5,'split':'suc','subset':'task','source_suc_id':'s','video_sha256':'v'},
                'f':{'reward':1,'split':'fail','subset':'task','source_suc_id':'s','video_sha256':'v'},
                'x':{'reward':1,'split':'fail','subset':'task','source_suc_id':'s','video_sha256':'v'}}
        rows={'s':{'progress':1.,'status':'ok'},'f':{'progress':.5,'status':'ok'},
              'x':{'progress':None,'status':'parse_error'}}
        result=summary(rows,labels,list(labels))
        self.assertEqual(result['mae'],1.)
        self.assertEqual(result['mae_all_expected_bounds'],[2/3,2.])
        self.assertEqual(result['accuracy']['0.125/0.875']['all']['correct'],1)
        self.assertEqual(result['accuracy']['0.125/0.875']['all']['rate_all_expected'],1/3)
        self.assertEqual(result['pairwise']['n'],1)
        self.assertEqual(result['pairwise']['ordinal_difference_counts']['2'],1)
        self.assertEqual(result['pairwise']['continuous_mean_delta'],.5)
        labels['f']['video_sha256']='different'
        with self.assertRaises(ValueError):pairwise({k:v for k,v in rows.items() if k!='x'},labels)

    def test_reward_labels_progress_bins_and_endpoint_thresholds_are_distinct(self):
        labels={'s':{'reward':5,'split':'suc','subset':'task','source_suc_id':None,'video_sha256':'v'},
                'f':{'reward':1,'split':'fail','subset':'task','source_suc_id':'s','video_sha256':'v'}}
        rows={'s':{'progress':1.,'status':'ok'},'f':{'progress':.15,'status':'ok'}}
        result=summary(rows,labels,list(labels))
        self.assertEqual(result['mae'],.5)
        self.assertEqual(result['ordinal_prediction_distributions']['fail']['counts']['2'],1)
        self.assertEqual(result['prediction_distributions']['fail']['counts']['1'],1)
        self.assertEqual(result['accuracy']['0.125/0.875']['fail']['correct'],0)
        self.assertEqual(result['accuracy']['0.2/0.8']['fail']['correct'],1)

    def test_paired_error_change_sign_and_video_cluster_unit(self):
        labels={}
        for g in range(3):
            for j in range(2):labels[f'{g}_{j}']={'reward':1,'split':'fail','video_sha256':str(g)}
        base={k:{'progress':1.,'status':'ok'} for k in labels}
        rows={k:{'progress':0.,'status':'ok'} for k in labels}
        result=paired_change(base,rows,labels,list(labels))
        self.assertEqual(result['video_clusters'],3)
        self.assertEqual(result['mae_delta'],-4.)
        self.assertEqual(result['mae_delta_ci95'],[-4.,-4.])


class WorkSharingContract(unittest.TestCase):
    def test_an_owned_condition_is_not_executed_or_written_by_a_second_worker(self):
        output=OUT/'research/contract_lock_fixture'
        path=output/'locks/last_frame_target_8.lock'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with patch('mydata_bench.addbase_eval.run._predict_condition') as run:
                self.assertIsNone(predict_condition(None,[],'last_frame:target:8',None,output))
                run.assert_not_called()
        with patch('mydata_bench.addbase_eval.run._predict_condition',return_value={'completed':True}) as run:
            self.assertEqual(predict_condition(None,[],'last_frame:target:8',None,output),{'completed':True})
            run.assert_called_once()


if __name__=='__main__':unittest.main()
