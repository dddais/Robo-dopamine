#!/usr/bin/env python3
"""Direct-script entry point for freezing a source-reward cohort."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydata_bench.cohorts import main


if __name__ == "__main__":
    main()
