import math
import unittest
import numpy as np
from .select_profiles import nll
from .profile_worker import layer_heads


class FunctionalProfileTests(unittest.TestCase):
    def test_full_native_class_loss_and_shift_invariance(self):
        logits=[-2.,1.,3.,-1.,.5]
        row={'status':'ok','native_class_logits_positive':logits}
        for reward in range(1,6):
            expected=-math.log(np.exp(logits)[reward-1]/np.exp(logits).sum())
            self.assertAlmostEqual(nll(row,reward,'qwen'),expected)
            shifted=dict(row,native_class_logits_positive=[x+1000 for x in logits])
            self.assertAlmostEqual(nll(shifted,reward,'qwen'),expected)
        self.assertAlmostEqual(nll({'status':'ok','native_class_logits_positive':[0]*5},3,'qwen'),math.log(5))

    def test_success_logistic_loss_is_stable(self):
        for z in [-1000.,0.,1000.]:
            for reward in [1,5]:
                result=nll({'status':'ok','success_logit_positive':z},reward,'meter')
                self.assertTrue(math.isfinite(result))
        self.assertAlmostEqual(nll({'status':'ok','success_logit_positive':0.},5,'meter'),math.log(2))

    def test_each_layer_group_uses_eight_distinct_heads(self):
        rows=[{'layer':layer,'head':head,'visual_score':32-head,'task_score':head}
              for layer in [8,9] for head in range(32)]
        selected=layer_heads(rows,9)
        self.assertEqual([r['head'] for r in selected],[0,31,1,30,2,29,3,28])
        self.assertEqual({r['layer'] for r in selected},{9})


if __name__=='__main__':unittest.main()
