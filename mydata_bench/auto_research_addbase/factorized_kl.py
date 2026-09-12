"""Class-symmetric KL cap using both actual native counterfactuals."""
import numpy as np

from .kl_contrast import kl_limited_contrast


def factorized_kl(positive, visual_negative, task_negative, budget):
    arrays = [np.asarray(x,dtype=np.float32) for x in [positive,visual_negative,task_negative]]
    if any(x.shape!=arrays[0].shape or x.shape[-1]!=5 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Three equal finite native five-class vectors required')
    positive, visual, task = arrays
    # Exactly the source runtime's float32 negative-mean arithmetic.
    mean = .5*(visual+task)
    return kl_limited_contrast(positive,mean,budget,cap=2.)
