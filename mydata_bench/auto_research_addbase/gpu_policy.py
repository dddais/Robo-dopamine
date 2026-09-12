"""User restriction from 2026-09-11: physical GPU devices 0 and 1 only."""
import os


ALLOWED_GPUS = (0, 1)


def require_allowed_visibility():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = visible.split(',')
    if not devices or any(x not in {'0', '1'} for x in devices):
        raise RuntimeError('Research requires explicit CUDA_VISIBLE_DEVICES containing only physical GPU 0 or 1')
    if len(set(devices)) != len(devices):
        raise RuntimeError('Duplicate CUDA device visibility')
    return tuple(map(int, devices))
