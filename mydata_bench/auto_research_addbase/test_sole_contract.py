"""Exercise the exploratory SOLE ranking writer through production resume."""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .runtime import ResearchRuntime
from mydata_bench.addbase_eval.run import rank
from mydata_bench.top_eval.versioning import SOLE_PROTOCOL_VERSION, validate_records


def test_research_ranking_writes_version_and_resumes_with_step_six_feedback():
    runtime = ResearchRuntime.__new__(ResearchRuntime)
    runtime.cfg = dict(model='sole',protocol='official',sole_protocol_version=SOLE_PROTOCOL_VERSION,
        batch_size=2,scopes=['last_frame','all_frames'],skip_early_layers=8)
    samples = [{'example_id':str(i)} for i in range(2)]
    seen = []
    def prepare(samples,step,previous):
        seen.append((step,previous))
        return {},[{}]*2,[10]*2,['prompt']*2
    runtime.prepare=prepare
    state={'seen':set(range(36)), 'raw':{s:np.zeros((2,36,32)) for s in runtime.cfg['scopes']},
           'visual':np.zeros((2,36,32))}
    runtime.controller=SimpleNamespace(rank=lambda *_:state,clear=lambda:None)
    runtime.model=lambda **_:None
    baseline={s['example_id']:dict(status='ok',step_count=7,previous_percentage_text='12.340',
        sole_protocol_version=SOLE_PROTOCOL_VERSION) for s in samples}
    with tempfile.TemporaryDirectory() as tmp:
        initial=rank(runtime,samples,baseline,Path(tmp))
        resumed=rank(runtime,samples,baseline,Path(tmp))
        assert initial==resumed
        validate_records(runtime.cfg,resumed.values(),'resumed rankings')
    assert seen==[(7,['12.340','12.340'])]


def test_research_checks_feedback_length_and_reused_protocol_before_preparing():
    runtime=ResearchRuntime.__new__(ResearchRuntime)
    runtime.cfg={'model':'sole','protocol':'image_text'}
    with pytest.raises(ValueError,match='batch length'):
        runtime.prepare([{},{}],previous=['0'])
    with pytest.raises(ValueError,match='empty'):
        runtime.prepare([])
    runtime.cfg['protocol']='official'
    with pytest.raises(ValueError,match='sole_protocol_version'):
        runtime.prepare([{}])
