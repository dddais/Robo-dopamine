"""Reject obsolete SOLE artifacts; only the corrected implementation is supported."""
import json
from pathlib import Path


SOLE_PROTOCOL_VERSION = 'sole_official_v2'


def is_official_sole(cfg):
    return cfg.get('model') == 'sole' and cfg.get('protocol') == 'official'


def validate_protocol_config(cfg):
    if is_official_sole(cfg) and cfg.get('sole_protocol_version') != SOLE_PROTOCOL_VERSION:
        raise ValueError(
            'SOLE official inference requires sole_protocol_version=sole_official_v2. '
            'Use mydata_bench/configs/v2_crossmodel_addbase/sole_official.yaml '
            'and regenerate baseline, ranking and steering. '
            'Obsolete unversioned SOLE results are not supported.'
        )


def protocol_metadata(cfg):
    return {'sole_protocol_version': SOLE_PROTOCOL_VERSION} if is_official_sole(cfg) else {}


def validate_records(cfg, records, source):
    if not is_official_sole(cfg):
        return
    validate_protocol_config(cfg)
    for record in records:
        if record.get('sole_protocol_version') != SOLE_PROTOCOL_VERSION:
            raise ValueError(f'Incompatible SOLE cache in {source}; regenerate it with {SOLE_PROTOCOL_VERSION}')


def load_analysis(path):
    """Reporting must not re-publish obsolete SOLE metrics from saved summaries."""
    result = json.loads(Path(path).read_text())
    validate_protocol_config(result['config'])
    return result


def validate_output_directory(cfg, output):
    """Reject a historical directory before writing a config or loading weights."""
    validate_protocol_config(cfg)
    if not is_official_sole(cfg):
        return
    output = Path(output)
    config_path = output / 'run_config.json'
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if not is_official_sole(saved) or saved.get('sole_protocol_version') != SOLE_PROTOCOL_VERSION:
            raise ValueError(f'{output} contains historical results; remove obsolete SOLE artifacts before running')
    elif any(output.glob('predictions/*.jsonl')) or any(output.glob('steps/*.jsonl')) or any(output.glob('ranking*.json*')):
        raise ValueError(f'{output} contains unversioned artifacts; remove obsolete SOLE artifacts before running')
