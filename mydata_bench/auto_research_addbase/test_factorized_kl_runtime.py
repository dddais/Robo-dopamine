import unittest
import numpy as np

from .factorized_kl import factorized_kl
from .factorized_kl_runtime import FactorizedKLRuntime
from .verify_meter_port_discrete_replay import runtime


class FactorizedKLRuntimeTests(unittest.TestCase):
    def fake(self,model):
        r,s=runtime(FactorizedKLRuntime,model,'visual_and_task',2,False)
        r.cfg.update(contrast_kl_budget=.8,task_binding_fraction=.5,task_binding_distribution='uniform')
        return r,s

    def test_actual_branches_and_sample_solver(self):
        for model in ['qwen','roboreward']:
            r,s=self.fake(model)
            rows=r.predict([dict(example_id=str(i)) for i in range(5)],'all_frames:target:1',
                {'all_frames':{'ranking':[dict(layer=0,head=0)]}})
            self.assertEqual(s['calls'],[('binding_transport',4),('mass_transport',-4),('binding_task_suppression',4)])
            for row in rows:
                expected=factorized_kl(*[row[f] for f in ['native_class_logits_positive',
                    'native_class_logits_negative','native_class_logits_negative_task']],.8)
                np.testing.assert_array_equal(row['native_class_probabilities'],expected['probabilities'])
                self.assertEqual(row['contrast_weight'],float(expected['alpha']))
                self.assertEqual(row['actual_forward_branches'],3)
                self.assertLessEqual(row['kl_divergence_from_positive'],.8+1e-12)
                self.assertEqual(row['progress'],(row['reward']-1)/4)

    def test_baseline_and_identifier_independence(self):
        r,s=self.fake('qwen');samples=[dict(example_id=str(i)) for i in range(5)]
        rows=r.predict(samples,'baseline');self.assertEqual(s['calls'],[None])
        for row in rows:
            self.assertEqual(row['actual_forward_branches'],1);self.assertEqual(row['contrast_weight'],0)
            self.assertEqual(row['kl_solver_iterations'],0)
            self.assertEqual(row['native_class_probabilities'],row['fixed_cap_native_class_probabilities'])
        rank={'all_frames':{'ranking':[dict(layer=0,head=0)]}}
        a=r.predict(samples,'all_frames:target:1',rank)
        b=r.predict([dict(example_id='renamed'+str(i)) for i in range(5)],'all_frames:target:1',rank)
        self.assertEqual([x['native_class_probabilities'] for x in a],[x['native_class_probabilities'] for x in b])

    def test_unregistered_configuration_rejected_before_model(self):
        r,s=self.fake('qwen');r.cfg['contrast_kl_budget']=1.6
        with self.assertRaises(ValueError):r.predict([dict(example_id='a')],'baseline')
        self.assertEqual(s['calls'],[])


if __name__=='__main__':unittest.main()
