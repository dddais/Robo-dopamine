"""Replay old/current discrete control flow with identical synthetic native heads."""
import hashlib
import json
import time
from types import ModuleType, SimpleNamespace

import torch

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .empirical_profile import sha
from .evidence import EvidenceRuntime


SNAPSHOT = OUT / 'code_snapshots/1789139915452844222_pid3294289.json'


def runtime(cls, model, mode, alpha, anchored):
    obj = object.__new__(cls)
    obj.cfg = dict(model=model, method='binding_transport', bias=0 if mode=='task' else 4,
        contrast_weight=alpha, contrast_negative_mode=mode, negative_strength=4, negative_task_strength=4,
        contrast_reference='baseline' if anchored else 'positive')
    obj.prepare = lambda *args: ({}, [{'query':0} for _ in range(5)], [0]*5, ['literal']*5)
    obj.audit = lambda m: m
    obj.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s, **kw: [int(s)-1]))
    state = dict(branch=None, calls=[])
    def steer(maps, heads, strength, scope, region):
        state['branch'] = (obj.cfg['method'], strength)
        return dict(diagnostics={'0': dict(method=obj.cfg['method'], strength=strength, heads=[0])})
    def clear(): state['branch'] = None
    obj.controller = SimpleNamespace(steer=steer, clear=clear)
    generator = torch.Generator().manual_seed(2501)
    keys = [None, ('binding_transport',4), ('mass_transport',-4), ('binding_task_suppression',4),
            ('binding_transport',0), ('binding_task_suppression',0)]
    cells = {k:torch.randn(5,1,5,generator=generator)*3 for k in keys}
    def forward(**kw):
        state['calls'].append(state['branch'])
        return SimpleNamespace(logits=cells[state['branch']].clone())
    obj.model = forward
    return obj, state


def main():
    old = json.loads(SNAPSHOT.read_text())['sources']['mydata_bench/auto_research_addbase/evidence.py']
    if hashlib.sha256(old['source'].encode()).hexdigest() != old['sha256']:
        raise ValueError('Startup source snapshot integrity failure')
    module = ModuleType('mydata_bench.auto_research_addbase._old_evidence_replay')
    module.__package__ = 'mydata_bench.auto_research_addbase'
    exec(compile(old['source'], str(SNAPSHOT), 'exec'), module.__dict__)
    checks = []
    for model in ['qwen','roboreward']:
        for mode, anchored in [('visual',False),('visual',True),('task',False),('visual_and_task',False)]:
            for alpha in [.5,1.,2.]:
                for condition in ['baseline','all_frames:target:1']:
                    values = []
                    for cls in [module.EvidenceRuntime, EvidenceRuntime]:
                        obj,state = runtime(cls,model,mode,alpha,anchored)
                        rows = obj.predict([dict(example_id=str(i)) for i in range(5)], condition,
                                           {'all_frames':{'ranking':[dict(layer=0,head=0)]}})
                        for row in rows: row.pop('duration_seconds_per_batch')
                        values.append((rows,state,obj.cfg['method']))
                    if values[0] != values[1]:
                        raise ValueError('Discrete outputs or actual branch invocation flow changed')
                    checks.append(dict(model=model, mode=mode, anchored=anchored, alpha=alpha,
                                       condition=condition, exact_synthetic_replay=True, n=5))
    destination = OUT / 'audit' / time.strftime('meter_port_discrete_source_replay_%Y%m%d_%H%M%S.json')
    create_json(destination, dict(status='pass', labels_read=False, checks=checks,
        sources_sha256={str(SNAPSHOT):sha(SNAPSHOT), 'mydata_bench/auto_research_addbase/evidence.py':
                       sha('mydata_bench/auto_research_addbase/evidence.py')},
        interpretation='48 old/current synthetic runtime replays, 240 paired rows. Native logits, probabilities, '
                       'branch calls and metadata exactly match, except timing. CPU control-flow regression; '
                       'this is not a real checkpoint forward or an independent efficacy replication.'))
    print(destination)


if __name__ == '__main__': main()
