import unittest
import torch
from .evidence import combine_native_logits


class EvidenceCompositionTests(unittest.TestCase):
    def test_original_arithmetic_unchanged(self):
        torch.manual_seed(71)
        positive=torch.randn(8,5)
        negative=torch.randn(8,5)
        for weight in [.5,1.,2.]:
            self.assertTrue(torch.equal(combine_native_logits(positive,negative,weight),
                                        (1+weight)*positive-weight*negative))

    def test_no_evidence_and_zero_weight_preserve_reference(self):
        torch.manual_seed(9)
        baseline=torch.randn(8,10)
        positive=torch.randn(8,10)
        negative=torch.randn(8,10)
        self.assertTrue(torch.equal(combine_native_logits(positive,positive,2.,baseline),baseline))
        self.assertTrue(torch.equal(combine_native_logits(positive,negative,0.,baseline),baseline))

    def test_probability_ratio_and_all_five_classes(self):
        baseline=torch.zeros(5,5,dtype=torch.float64)
        positive=torch.eye(5,dtype=torch.float64)
        negative=-positive
        logits=combine_native_logits(positive,negative,1.,baseline)
        torch.testing.assert_close(logits.argmax(-1),torch.arange(5))
        expected=baseline.softmax(-1)*(positive.softmax(-1)/negative.softmax(-1))
        expected=expected/expected.sum(-1,keepdim=True)
        torch.testing.assert_close(logits.softmax(-1),expected,rtol=1e-14,atol=1e-14)
        shifted=combine_native_logits(positive+7.,negative-3.,1.,baseline+2.)
        torch.testing.assert_close(shifted.softmax(-1),logits.softmax(-1),rtol=1e-14,atol=1e-14)


if __name__=='__main__':unittest.main()
