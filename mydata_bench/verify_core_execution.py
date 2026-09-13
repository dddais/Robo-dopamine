"""Real-checkpoint smoke audit of GRM and shared Qwen/RoboReward paths.

Run in a separate output directory. This does not rerun or replace a benchmark.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
import yaml

from .addbase_eval.verify_execution import extract_source, assert_inputs_equal
from .attention_eval.masking import Head
from .attention_eval.runtime import AttentionRuntime, sample_images
from .qwen_eval.attention import QwenAttentionRuntime
from .qwen_eval.runner import Qwen3VLBaseline
from .qwen_eval.protocols import ROBOREWARDBENCH_NATIVE
from .roboreward_eval.runner import NativeRoboReward, ROBOREWARD_PROMPT

ROOT = Path(__file__).resolve().parents[1]


def first_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [next(r for r in rows if r['example_id'].startswith(split+'/')) for split in ['fail','suc']]


def check_diagnostics(result):
    diag = result['hook_diagnostics']
    layers = diag['per_layer']
    assert diag['hook_active'] and layers
    assert all(d['prefill_applied_calls'] == 1 and d['applied_calls'] > 0
               and d['selected_other_disjoint'] for d in layers.values()), diag
    return layers


def direct_decode(runtime, inputs):
    with torch.inference_mode():
        out = runtime.model.generate(**inputs, do_sample=False, temperature=None,
            top_p=None, top_k=None, max_new_tokens=runtime.config['max_new_tokens'],
            use_cache=True, pad_token_id=runtime.processor.tokenizer.pad_token_id)
    return runtime.processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()


def grm_audit():
    cfg = yaml.safe_load((ROOT/'mydata_bench/configs/v2/attention_10_grm_forward_after.yaml').read_text())['attention_eval']
    cfg['device_map'] = 'cuda:0'
    runtime = AttentionRuntime(cfg)
    samples = first_rows(Path(cfg['output_dir'])/'eligible.jsonl')
    ns = extract_source(ROOT/'examples/inference.py', {'SYSTEM_PROMPT','GRMInference'})
    capture = {}
    def generate(requests, **kwargs):
        capture['request'] = requests[0]
        return [SimpleNamespace(outputs=[SimpleNamespace(text='0.0')])]
    official = ns['GRMInference'].__new__(ns['GRMInference'])
    official.processor = runtime.processor
    official.model = SimpleNamespace(generate=generate)
    official.sampling_params = None
    rows = []
    for sample in samples:
        inputs, spans = runtime.prepare(sample)
        paths = sample_images(sample,cfg['blank_goal'],cfg)
        official.inference_batch([{'image':paths,'task':sample['task']}])
        request = capture['request']
        reference = runtime.processor(text=[request['prompt']],images=request['multi_modal_data']['image'],return_tensors='pt')
        assert_inputs_equal(inputs,reference)
        selected, image_positions, _ = runtime.target_positions(sample,spans)
        base = runtime.generate(sample,selected_positions=selected,image_positions=image_positions)
        assert base['raw_output'] == direct_decode(runtime,inputs)
        mass = runtime.collect_mass(sample)
        assert np.isfinite(mass['raw_mass']).all() and np.shape(mass['raw_mass']) == (36,32)
        # Exercise the actual hook even when its bias is zero.
        from .attention_eval.masking import resolve_negative_positions
        other, _ = resolve_negative_positions(spans,selected,'target_span')
        with runtime.steering_hooks([Head(20,0)],selected,other,0.,{},'all'):
            assert direct_decode(runtime,inputs) == base['raw_output']
        steered = runtime.generate(sample,heads=[Head(20,0)],selected_positions=selected,
            image_positions=image_positions,bias=6.,query_scope='all',negative_scope='target_span')
        diag = check_diagnostics(steered)
        assert runtime.generate(sample,selected_positions=selected,image_positions=image_positions)['raw_output'] == base['raw_output']
        rows.append({'example_id':sample['example_id'],'official_prompt_and_images_exact':True,
            'official_prompt_HF_input_tensors_exact':True,'direct_greedy_text_exact':True,
            'zero_bias_hook_text_exact':True,'ranking_shape':[36,32],
            'baseline_restored':True,'baseline':base['raw_output'],'steered':steered['raw_output'],
            'diagnostics':diag,'image_grids':inputs['image_grid_thw'].tolist()})
    return rows


def qwen_audit(model):
    cfg = yaml.safe_load((ROOT/f'mydata_bench/configs/v2_crossmodel/attention_{"07_qwen" if model=="qwen" else "03_roboreward"}_text_images.yaml').read_text())['attention_steer']
    cfg['device_map'] = 'cuda:0'
    runtime = QwenAttentionRuntime(cfg)
    samples = first_rows(Path(cfg['output_dir'])/'cohort_inputs.jsonl')
    rows = []
    # Both baseline runners use their production input methods with this same
    # eager model. Native video uses the attention run's explicit eight-frame cap.
    for proto,order in [('roborewardbench_image_sequence','text_then_images'),
                        ('roborewardbench_interleaved_image_sequence','interleaved'),
                        (ROBOREWARDBENCH_NATIVE,'video_then_text')]:
        runtime.protocol=proto; runtime.content_order=order
        runtime.processor.video_processor.max_frames=8
        for sample in samples:
            prepared = runtime.prepare(sample)
            selected = runtime.target_positions(sample,prepared)
            base = runtime.generate(sample,prepared=prepared)
            baseline = Qwen3VLBaseline.__new__(Qwen3VLBaseline)
            baseline.processor=runtime.processor; baseline.protocol=proto
            baseline.model=runtime.model
            def capture_decode(inputs):
                assert_inputs_equal(prepared.inputs,inputs)
                return direct_decode(runtime,prepared.inputs)
            baseline._decode=capture_decode
            payload={'protocol':proto,'task':sample['task'],'image':sample['image_paths'],
                'content_order':order,'sampling_record':sample['image_sampling_record'],
                'media_order':[],'video_path':sample['video_path'],
                'prompt':ROBOREWARD_PROMPT.format(task=sample['task'])}
            text, _ = baseline.infer(payload)
            assert text == base['raw_output']
            native_runner_checked = False
            if model=='roboreward' and order!='interleaved':
                native=NativeRoboReward.__new__(NativeRoboReward)
                native.processor=runtime.processor; native.torch=torch
                native.input_representation='video' if proto==ROBOREWARDBENCH_NATIVE else 'independent_images'
                native.preprocessor_mode='checkpoint_default';native.content_order=order
                native.do_sample=False;native.max_new_tokens=cfg['max_new_tokens']
                def checked_generate(**kwargs):
                    actual={k:kwargs[k] for k in prepared.inputs}
                    assert_inputs_equal(prepared.inputs,actual)
                    return runtime.model.generate(**kwargs)
                native.model=SimpleNamespace(device=runtime.model.device,config=runtime.model.config,generate=checked_generate)
                paths=[sample['video_path']] if proto==ROBOREWARDBENCH_NATIVE else sample['image_paths']
                native_text,_=native.infer(sample['task'],paths,{'video_input_protocol':'checkpoint_native_mp4_v1'})
                assert native_text==base['raw_output']
                native_runner_checked=True
            mass=runtime.collect_mass(sample)
            assert np.isfinite(mass['raw_mass']).all() and np.shape(mass['raw_mass'])==(36,32)
            with runtime.steering_hooks([Head(20,0)],selected,prepared.visual_positions,0.,'all','target_span',prepared.spans,{}):
                assert direct_decode(runtime,prepared.inputs)==base['raw_output']
            steered=runtime.generate(sample,prepared=prepared,heads=[Head(20,0)],selected_positions=selected,
                bias=6.,query_scope='all',negative_scope='target_span')
            diag=check_diagnostics(steered)
            assert runtime.generate(sample,prepared=prepared)['raw_output']==base['raw_output']
            # All-frame mapping must also cover every sampled temporal plane.
            runtime.temporal_scope='all_frames'
            runtime.target_positions(sample,prepared)
            runtime.temporal_scope='last_frame'
            rows.append({'example_id':sample['example_id'],'protocol':proto,'content_order':order,
                'baseline_runner_input_tensors_and_greedy_text_exact':True,
                'roboreward_dedicated_baseline_runner_checked':native_runner_checked,
                'zero_bias_hook_text_exact':True,'baseline_restored':True,'ranking_shape':[36,32],
                'baseline':base['raw_output'],'steered':steered['raw_output'],
                'diagnostics':diag,'alignment':prepared.video_metadata})
            print(model,proto,sample['example_id'],'PASS',flush=True)
    return rows


def grm_incremental_audit():
    cfg=yaml.safe_load((ROOT/'mydata_bench/configs/v2/attention_12_grm_incremental_after_official.yaml').read_text())['attention_eval']
    cfg['device_map']='cuda:0'
    sample=first_rows(Path(cfg['output_dir'])/'eligible.jsonl')[0]
    cfg['output_dir']=str(ROOT/'results/mydata_bench/audit_20260913/grm_incremental_cache')
    runtime=AttentionRuntime(cfg)
    plan=runtime.incremental_plan(sample)
    original=runtime.generate
    calls=[]
    def checked_generate(*args,**kwargs):
        result=original(*args,**kwargs)
        if kwargs.get('bias'):
            check_diagnostics(result)
        calls.append(bool(kwargs.get('bias')))
        return result
    runtime.generate=checked_generate
    base=runtime.generate_incremental(sample,plan)
    steered=runtime.generate_incremental(sample,plan,heads=[Head(20,0)],bias=6.)
    again=runtime.generate_incremental(sample,plan)
    assert base['incremental_steps']==again['incremental_steps']
    assert calls==[False]*len(plan)+[True]*len(plan)+[False]*len(plan)
    for result in [base,steered]:
        value=None
        for hop in result['incremental_steps']:
            score=hop['hop_score']
            value=score if value is None else value+(1-value)*score if score>=0 else value+value*score
            assert value==hop['accumulated_progress_unclipped']
        assert result['progress']==min(1.,max(0.,value))
    return {'example_id':sample['example_id'],'hop_count':len(plan),
        'all_hops_steered_and_baseline_restored':True,'independent_recurrence_exact':True,
        'baseline':{k:base[k] for k in ['progress','sampled_frame_indices','incremental_steps']},
        'steered':{k:steered[k] for k in ['progress','incremental_steps','hook_diagnostics']}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['grm','grm_incremental','qwen','roboreward'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    results=grm_audit() if args.model=='grm' else grm_incremental_audit() if args.model=='grm_incremental' else qwen_audit(args.model)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'model':args.model,'torch':torch.__version__,
        'transformers':transformers.__version__,'scope':'two samples per protocol, or one full episode for GRM incremental; arbitrary single head checks execution, not performance; no full benchmark rerun',
        'results':results},indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
