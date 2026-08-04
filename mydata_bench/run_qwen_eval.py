#!/usr/bin/env python3
"""Direct-script entry point for Qwen3-VL RoboRewardBench baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydata_bench.qwen_eval.cli import main


if __name__ == "__main__":
    main()
