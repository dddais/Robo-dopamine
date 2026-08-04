#!/usr/bin/env python3
"""Direct-script entry point for native discrete RoboReward evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rewardbench.roboreward_eval.cli import main


if __name__ == "__main__":
    main()

