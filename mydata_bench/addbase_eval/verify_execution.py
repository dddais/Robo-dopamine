"""Small real-checkpoint audit; writes only to an explicitly separate report.

Official inputs and heads are evaluated from the vendored upstream source.
SOLE decoding is compared under our declared HF greedy/512 setting, not vLLM
sampling. This is a smoke audit, not a replacement for the full experiment.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
import torch
from torch import nn
import transformers
import yaml

from .prepare import ROOT, create_json
from .runtime import Runtime


REF = ROOT/'mydata_bench/addbase_eval/references'


def extract_source(path, names, namespace=None):
    """Compile inspected definitions only; never initialize upstream servers."""
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            nodes.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    ns = dict(np=np, cv2=cv2, torch=torch, nn=nn, Image=Image, json=json,
              process_vision_info=process_vision_info,
              logger=SimpleNamespace(info=lambda *_: None))
    ns.update(namespace or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), ns)
    return ns


def official_meter_inputs(processor, samples):
    root = REF/'robometer/robometer/data/collators'
    ns = extract_source(root/'utils.py', {'convert_frames_to_pil_images'})
    methods = {'_prepare_frames_for_conversation', '_add_vision_content_to_list',
               '_process_conversation', '_process_progress_batch'}
    ns = extract_source(root/'rbm_heads.py', methods, ns)
    collator = type('OfficialCollatorMethods', (), {k: ns[k] for k in methods})()
    collator.processor = processor
    collator.base_model_id = 'Qwen/Qwen3-VL-4B-Instruct'
    collator.use_multi_image = collator.use_per_frame_progress_token = collator.inference = True
    collator.shuffle_progress_frames = False
    collator.max_length = 32768
    examples = [SimpleNamespace(trajectory=SimpleNamespace(
        frames=[Image.open(p).convert('RGB') for p in s['image_paths']],
        frames_shape=None, target_progress=None, task=s['task']), resample_attempts=0) for s in samples]
    return {k:v for k,v in collator._process_progress_batch(examples).items() if torch.is_tensor(v)}


def official_sole_inputs(processor, samples, step, previous):
    ns = extract_source(REF/'rewardgen/rewardgen/sole.py', {
        'system_prompt_template', 'question_template', 'problem_key',
        'user_question_template_external_view', 'resize_with_padding',
        'create_composite_frame', 'make_conversation_image'})
    texts, images = [], []
    for sample, prior in zip(samples, previous):
        frames = [np.asarray(Image.open(sample['image_paths'][i]).convert('RGB')) for i in [0,step-1,step]]
        composite = ns['create_composite_frame'](None,frames[0],None,frames[1],None,frames[2],view_type='external')
        h,w = smart_resize(*composite.shape[:2],factor=28,min_pixels=3136,max_pixels=12845056)
        images.append(Image.fromarray(composite).resize((w,h)))
        question = ns['user_question_template_external_view'].format(task_description=sample['task'],prev_progress=prior)
        messages = json.loads(ns['make_conversation_image']({'question':question}))
        texts.append(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    return processor(text=texts,images=images,return_tensors='pt',padding=True,add_special_tokens=False)


def assert_inputs_equal(actual, reference):
    assert set(actual) == set(reference), (set(actual), set(reference))
    for key, value in actual.items():
        expected = reference[key].to(device=value.device, dtype=value.dtype)
        if not torch.equal(value, expected):
            raise AssertionError(f'Official input tensor mismatch: {key}')


def official_meter_readout(runtime, hidden, ids):
    root = REF/'robometer/robometer/models'
    ns = extract_source(root/'heads.py', {'PredictionHeadsMixin'})
    methods = {'squeeze_last_safe','_extract_hidden_state_from_token',
               '_apply_heads_to_hidden_states','_process_token_extraction'}
    ns = extract_source(root/'rbm.py', methods, ns)
    cls = type('OfficialReadout', (ns['PredictionHeadsMixin'],), {k:ns[k] for k in methods if k.startswith('_')})
    ref = cls(hidden_dim=hidden.shape[-1],model_config=SimpleNamespace(
        progress_loss_type='discrete',progress_discrete_bins=10)).to(hidden.device,dtype=hidden.dtype).eval()
    ref.processor = runtime.processor
    ref.base_model_id = 'Qwen/Qwen3-VL-4B-Instruct'
    ref.load_state_dict({k:v for k,v in runtime.model.state_dict().items()
                        if k.split('.')[0] in {'progress_head','success_head','preference_head'}},strict=True)
    with torch.inference_mode():
        progress, success, _ = ref._process_token_extraction(hidden, ids, 'progress')
    # Keep known logits typed. The upstream heuristic has a sum==1 edge bug.
    values = (progress['A'].float().softmax(-1)*torch.linspace(0,1,10,device=hidden.device)).sum(-1)
    return values, success['A'].float().sigmoid(), success['A'].sigmoid().float()


def run_meter(runtime, samples, folder):
    inputs, maps, _, _ = runtime.prepare(samples)
    assert_inputs_equal(inputs, official_meter_inputs(runtime.processor, samples))
    with torch.inference_mode():
        hidden = runtime.model(**inputs,use_cache=False).last_hidden_state
        actual, success, _ = runtime.model.read_progress(hidden,inputs['input_ids'],runtime.prog_id)
        ref, ref_success, bf16_success = official_meter_readout(runtime,hidden,inputs['input_ids'])
    torch.testing.assert_close(torch.tensor(actual,device=ref.device),ref,rtol=0,atol=0)
    torch.testing.assert_close(torch.tensor(success,device=ref.device),ref_success,rtol=0,atol=0)
    runtime.collect(samples)
    baseline = runtime.predict(samples,'baseline')
    assert [r['progress'] for r in baseline] == [r[-1] for r in actual]
    rankings = {scope:json.loads((folder/f'ranking_{scope}.json').read_text()) for scope in runtime.cfg['scopes']}
    steered = runtime.predict(samples,'last_frame:target:32',rankings)
    again = runtime.predict(samples,'baseline')
    assert [r['progress'] for r in again] == [r['progress'] for r in baseline]
    assert [r['success_probability'] for r in again] == [r['success_probability'] for r in baseline]
    differences = {}
    for name, rows in [('baseline',baseline),('last_frame_target_32',steered)]:
        from .run import latest
        archived = latest(folder/'predictions'/f'{name}.jsonl')
        differences[name] = [{key:abs(row[key]-archived[row['example_id']][key])
                              for key in ['progress','success_probability']} for row in rows]
    single_differences = {}
    for condition in ['baseline','last_frame:target:32']:
        name = condition.replace(':','_')
        archived = latest(folder/'predictions'/f'{name}.jsonl')
        single_rows = [runtime.predict([s],condition,rankings)[0] for s in samples]
        single_differences[name] = [{key:abs(row[key]-archived[row['example_id']][key])
                                    for key in ['progress','success_probability']} for row in single_rows]
    saved_cfg = json.loads((folder/'run_config.json').read_text())
    frozen = json.loads(Path(saved_cfg['inputs']).read_text())
    original_differences = {}
    for condition in ['baseline','last_frame:target:32']:
        name = condition.replace(':','_')
        population = frozen if condition=='baseline' else [s for s in frozen if s['cohort']]
        selected_ids = {s['example_id'] for s in samples}
        archived = latest(folder/'predictions'/f'{name}.jsonl')
        original_differences[name] = []
        for start in range(0,len(population),saved_cfg['batch_size']):
            batch = population[start:start+saved_cfg['batch_size']]
            if not any(s['example_id'] in selected_ids for s in batch):
                continue
            for row in runtime.predict(batch,condition,rankings):
                if row['example_id'] in selected_ids:
                    original_differences[name].append({key:abs(row[key]-archived[row['example_id']][key])
                                                      for key in ['progress','success_probability']})
    return {'official_input_tensors_exact':True,'official_heads_fp32_readout_exact':True,
            'ranking_and_steering_leave_baseline_unchanged':True,
            'upstream_bf16_sigmoid_max_rounding_difference':float((ref_success-bf16_success).abs().max()),
            'archive_absolute_differences':differences,
            'archive_batch_size_1_absolute_differences':single_differences,
            'archive_saved_batch_size':saved_cfg['batch_size'],
            'archive_saved_batch_absolute_differences':original_differences,
            'baseline':[{'example_id':r['example_id'],'progress':r['progress'],'success':r['success_probability']} for r in baseline],
            'steering':[{'example_id':r['example_id'],'progress':r['progress'],'success':r['success_probability'],
                         'diagnostics':r['attention_diagnostics']} for r in steered]}


def run_sole(runtime, samples):
    from .run import _predict_condition, rank
    import tempfile
    # Run the real production seven-step baseline, ranking, and steering loops.
    with tempfile.TemporaryDirectory(prefix='sole_execution_audit_') as tmp:
        output = Path(tmp)
        base = _predict_condition(runtime,samples,'baseline',None,output)
        assert all(r['status']=='ok' and r['step_count']==7 for r in base.values()), base
        steps = [json.loads(line) for line in (output/'steps/baseline.jsonl').read_text().splitlines()]
        for step in range(1,8):
            previous = [next(r['previous_percentage_text'] for r in steps
                             if r['example_id']==s['example_id'] and r['step']==step) for s in samples]
            actual, _, _, _ = runtime.prepare(samples,step,previous)
            reference = official_sole_inputs(runtime.processor,samples,step,previous)
            assert_inputs_equal(actual,reference)
            # Direct model generation bypasses Runtime.predict and its parser.
            with torch.inference_mode():
                out = runtime.model.generate(**{k:v.to(actual[k].device,dtype=actual[k].dtype) for k,v in reference.items()},
                    do_sample=False,max_new_tokens=512,temperature=None,top_p=None,top_k=None,use_cache=True,
                    logits_to_keep=1,pad_token_id=runtime.processor.tokenizer.pad_token_id)
            raw = runtime.processor.batch_decode(out[:,actual['input_ids'].shape[1]:],skip_special_tokens=True)
            for sample, text in zip(samples,raw):
                expected = next(r['raw_output'] for r in steps if r['example_id']==sample['example_id'] and r['step']==step)
                assert text == expected, f'Direct official-input decode differs at step {step}'
        rankings = rank(runtime,samples,base,output)
        target = _predict_condition(runtime,samples,'last_frame:target:8',rankings,output)
        assert all(r['status']=='ok' and r['step_count']==7 for r in target.values()), target
        # Check selected layers actually ran through cached autoregressive decode.
        for row in target.values():
            diag = row['attention_diagnostics']
            assert diag and all(v['prefill_calls']==1 and v['decode_calls']>0 for v in diag.values()), diag
        again = runtime.predict(samples,'baseline',step=7,previous=[base[s['example_id']]['previous_percentage_text'] for s in samples])
        assert all(r['raw_output']==base[r['example_id']]['raw_output'] for r in again)
        return {'seven_steps_official_input_tensors_and_greedy_text_exact':True,
                'steering_seven_steps_with_own_feedback_completed':True,'steering_leaves_baseline_unchanged':True,
                'ranking_sample_count':len(samples),
                'baseline':[{k:r[k] for k in ['example_id','progress_curve','raw_output']} for r in base.values()],
                'steering':[{k:r[k] for k in ['example_id','progress_curve','raw_output','attention_diagnostics']} for r in target.values()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=['meter','sole'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    name = 'meter_official' if args.model=='meter' else 'sole_official'
    cfg = yaml.safe_load((ROOT/f'mydata_bench/configs/v2_crossmodel_addbase/{name}.yaml').read_text())
    samples = [s for s in json.loads(Path(cfg['inputs']).read_text()) if s['ranking']][:2]
    runtime = Runtime(cfg)
    result = run_meter(runtime,samples,Path(cfg['output_dir'])) if args.model=='meter' else run_sole(runtime,samples)
    files = [REF/'robometer/robometer/data/collators/rbm_heads.py',REF/'robometer/robometer/models/rbm.py',
             REF/'robometer/robometer/models/heads.py',REF/'rewardgen/rewardgen/sole.py']
    create_json(args.output,{'model':args.model,'sample_ids':[s['example_id'] for s in samples],
        'scope':'two-sample execution audit; no full benchmark rerun','torch':torch.__version__,
        'transformers':transformers.__version__,'loading':runtime.model.loading_audit,
        'reference_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'results':result})
    print(f'Execution audit passed: {args.output}',flush=True)


if __name__=='__main__':
    main()
