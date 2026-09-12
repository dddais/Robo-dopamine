import unittest

import numpy as np

from .factorized_kl import factorized_kl
from .kl_contrast import kl_limited_contrast, logsoftmax


class FactorizedKLTests(unittest.TestCase):
    def test_identical_counterfactuals_and_exchange(self):
        rng=np.random.default_rng(27)
        p,v,t=[rng.normal(size=(16,5)).astype(np.float32) for _ in range(3)]
        for budget in [.05,.2,.8]:
            a=factorized_kl(p,v,v,budget);b=kl_limited_contrast(p,v,budget)
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            a=factorized_kl(p,v,t,budget);b=factorized_kl(p,t,v,budget)
            for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_actual_mean_budget_and_category_permutation(self):
        rng=np.random.default_rng(2702)
        p,v,t=[rng.normal(size=(16,5)).astype(np.float32)*4 for _ in range(3)]
        for budget in [.05,.2,.8]:
            r=factorized_kl(p,v,t,budget)
            z=p.astype(float)+r['alpha'][:,None]*(p.astype(float)-(.5*(v+t)).astype(float))
            logp=logsoftmax(z);prob=np.exp(logp)
            np.testing.assert_allclose(prob,r['probabilities'],rtol=1e-12,atol=1e-12)
            kl=np.sum(prob*(logp-logsoftmax(p.astype(float))),axis=-1)
            self.assertTrue(np.all(kl<=budget+1e-12));self.assertTrue(np.all(r['alpha']<=2))
            order=[4,1,3,0,2];permuted=factorized_kl(p[:,order],v[:,order],t[:,order],budget)
            np.testing.assert_allclose(permuted['probabilities'],prob[:,order],rtol=1e-12,atol=1e-12)

    def test_all_classes_batch_independence_and_invalid_input(self):
        p=np.eye(5,dtype=np.float32)*4;zero=np.zeros_like(p)
        r=factorized_kl(p,zero,zero,.8)
        np.testing.assert_array_equal(r['probabilities'].argmax(-1),np.arange(5))
        singles=np.stack([factorized_kl(p[i],zero[i],zero[i],.8)['probabilities'] for i in range(5)])
        np.testing.assert_array_equal(singles,r['probabilities'])
        bad=p.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):factorized_kl(p,zero,bad,.8)


if __name__=='__main__':unittest.main()
