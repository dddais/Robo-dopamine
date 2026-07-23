"""RoboRewardBench attention-mask intervention experiments.

The package deliberately separates target grounding, head discovery, causal
intervention, and post-hoc scoring.  In particular, reward labels are not part
of the model-facing data structure and cannot enter the GRM prompt by accident.
"""

