#!/usr/bin/env python3
"""Direct-script entry point for RoboRewardBench attention experiments.

Example:
    python mydata_bench/run_attention_eval.py prepare \
        --config mydata_bench/configs/attention_10_grm_forward_after.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydata_bench.attention_eval.cli import main


if __name__ == "__main__":
    main()
