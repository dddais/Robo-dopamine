#!/usr/bin/env python3
"""Direct entry point for Qwen3-VL attention ranking and steering."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydata_bench.qwen_eval.attention_cli import main


if __name__ == "__main__":
    main()
