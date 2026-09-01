#!/usr/bin/env python3
"""Direct entry point for the incremental gaze-head debug matrix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydata_bench.gaze_debug import main


if __name__ == "__main__":
    main()
