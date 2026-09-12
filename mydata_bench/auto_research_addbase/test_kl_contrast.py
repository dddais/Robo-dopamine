"""Mathematical safety checks for the new sample-local KL constraint."""
import unittest
import numpy as np
from scipy.special import logsumexp
from .kl_contrast import kl_limited_contrast


class KLContrastTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260911)
        self.pos = rng.normal(size=(200, 5))*3
        self.neg = rng.normal(size=(200, 5))*3

    def divergence(self, z, base):
        logp = z-logsumexp(z, axis=-1, keepdims=True)
        logq = base-logsumexp(base, axis=-1, keepdims=True)
        return (np.exp(logp)*(logp-logq)).sum(-1)

    def test_constraint_and_largest_feasible_gain(self):
        for budget in [.05, .2, .8]:
            out = kl_limited_contrast(self.pos, self.neg, budget)
            a = out['alpha']
            independent = self.divergence(self.pos+a[:, None]*(self.pos-self.neg), self.pos)
            np.testing.assert_allclose(independent, out['kl'], atol=1e-13)
            self.assertTrue(np.all(independent <= budget+1e-12))
            self.assertTrue(np.all((a >= 0) & (a <= 2)))
            limited = a < 2-1e-8
            raised = self.divergence(self.pos+(a[:, None]+1e-7)*(self.pos-self.neg), self.pos)
            self.assertTrue(np.all(raised[limited] > budget))

    def test_exact_boundary_distributions(self):
        zero = kl_limited_contrast(self.pos, self.neg, 0)
        expected = np.exp(self.pos-logsumexp(self.pos, axis=-1, keepdims=True))
        np.testing.assert_allclose(zero['probabilities'], expected, atol=1e-14)
        self.assertTrue(np.all(zero['alpha'] == 0))
        free = kl_limited_contrast(self.pos, self.neg, 1e6)
        self.assertTrue(np.all(free['alpha'] == 2))
        z = 3*self.pos-2*self.neg
        np.testing.assert_allclose(free['probabilities'], np.exp(z-logsumexp(z, axis=-1, keepdims=True)), atol=1e-14)

    def test_common_shift_and_class_permutation_invariance(self):
        ref = kl_limited_contrast(self.pos, self.neg, .2)
        shifted = kl_limited_contrast(self.pos+37, self.neg-91, .2)
        np.testing.assert_allclose(ref['probabilities'], shifted['probabilities'], atol=1e-12)
        permutation = [3, 0, 4, 1, 2]
        permuted = kl_limited_contrast(self.pos[:, permutation], self.neg[:, permutation], .2)
        np.testing.assert_allclose(permuted['probabilities'], ref['probabilities'][:, permutation], atol=1e-12)
        np.testing.assert_allclose(permuted['alpha'], ref['alpha'], atol=1e-12)

    def test_identical_branches_preserve_every_native_class(self):
        pos = np.eye(5)*4
        out = kl_limited_contrast(pos, pos, .05)
        self.assertEqual(out['probabilities'].argmax(-1).tolist(), list(range(5)))
        np.testing.assert_allclose(out['kl'], 0, atol=1e-14)

    def test_monotonicity_identity(self):
        d = self.pos-self.neg
        a, step = .4, 1e-5
        z = self.pos+a*d
        p = np.exp(z-logsumexp(z, axis=-1, keepdims=True))
        analytic = a*((p*d*d).sum(-1)-(p*d).sum(-1)**2)
        numeric = (self.divergence(self.pos+(a+step)*d, self.pos)-
                   self.divergence(self.pos+(a-step)*d, self.pos))/(2*step)
        np.testing.assert_allclose(numeric, analytic, atol=2e-8, rtol=2e-7)
        self.assertTrue(np.all(numeric >= -1e-10))

    def test_batch_is_sample_local_and_extreme_logits_finite(self):
        batch = kl_limited_contrast(self.pos[:8], self.neg[:8], .2)
        for i in range(8):
            single = kl_limited_contrast(self.pos[i], self.neg[i], .2)
            np.testing.assert_allclose(batch['probabilities'][i], single['probabilities'], atol=1e-14)
        out = kl_limited_contrast(self.pos*1000, self.neg*1000, .2)
        self.assertTrue(np.isfinite(out['probabilities']).all())
        self.assertTrue(np.all(out['kl'] <= .2+1e-9))


if __name__ == '__main__':
    unittest.main()
