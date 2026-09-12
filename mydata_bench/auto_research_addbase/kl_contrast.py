"""Label-free KL-constrained contrast of the complete native class vector."""
import numpy as np


def logsoftmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def kl_limited_contrast(positive, negative, budget, cap=2., iterations=64):
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim < 1 or positive.shape[-1] != 5:
        raise ValueError('Requires equal five-class native vectors')
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError('Native logits must be finite')
    if not np.isfinite(budget) or not np.isfinite(cap) or budget < 0 or cap < 0 or iterations < 1:
        raise ValueError('Finite nonnegative budget/cap and positive iteration count required')
    z0 = positive - positive.mean(axis=-1, keepdims=True)
    # Category-common shifts in either branch have no effect on the decision.
    d = positive - negative
    d = d - d.mean(axis=-1, keepdims=True)
    base_logp = logsoftmax(z0)

    def evaluate(alpha):
        z = z0 + alpha[..., None] * d
        logp = logsoftmax(z)
        p = np.exp(logp)
        # Clamp only negative roundoff of a mathematically nonnegative divergence.
        kl = np.maximum(0., (p*(logp-base_logp)).sum(axis=-1))
        return z, p, kl

    low = np.zeros(positive.shape[:-1])
    high = np.full_like(low, cap)
    cap_feasible = evaluate(high)[2] <= budget
    for _ in range(iterations):
        midpoint = (low+high)/2
        feasible = evaluate(midpoint)[2] <= budget
        low = np.where(feasible, midpoint, low)
        high = np.where(feasible, high, midpoint)
    alpha = np.where(cap_feasible, cap, low)
    # Fix the zero-budget boundary exactly, including numerical ties.
    if budget == 0:
        alpha = np.zeros_like(alpha)
    z, probabilities, kl = evaluate(alpha)
    return dict(logits=z, probabilities=probabilities, alpha=alpha, kl=kl)
