"""Select the same experiment subset for scheduling and scoring."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .common import MODELS


def add_selection_arguments(parser):
    parser.add_argument('--models', nargs='+', choices=MODELS)
    parser.add_argument('--exclude-configs', nargs='+', default=[], metavar='NAME',
                        help='Configuration filename stems to skip, e.g. qwen_official')


def select_configs(root, models=None, exclude_configs=()):
    entries = [(path, yaml.safe_load(Path(path).read_text()))
               for path in json.loads((Path(root) / 'matrix.json').read_text())]
    excluded = set(exclude_configs)
    unknown = excluded - {Path(path).stem for path, _ in entries}
    if unknown:
        raise ValueError(f'Unknown excluded configurations: {", ".join(sorted(unknown))}')
    selected = [(path, cfg) for path, cfg in entries
                if (models is None or cfg['model'] in models) and Path(path).stem not in excluded]
    if not selected:
        raise ValueError('No configurations selected')
    return selected
