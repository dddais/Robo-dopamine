#!/usr/bin/env python3
"""Direct entry point for RoboReward-8B checkpoint-native attention runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydata_bench.roboreward_eval.attention import main


if __name__ == "__main__":
    main()
