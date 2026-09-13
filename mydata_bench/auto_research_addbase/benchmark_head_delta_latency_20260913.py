"""Bounded, label-free batch-one timing of the frozen round33 checkpoints.

Writes new timing artifacts only. Existing efficacy results/checkpoints stay intact.
PNG frames and ROI tracks are pre-existing; online grounding and video decoding
are outside the measured scope. No training, scoring, or parameter selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from .prepare import OUT
from .head_delta_inference import final_record, inference_config
from .head_delta_worker import fixed_gates, ranking, sha
from .head_delta_reft import HeadDeltaRuntime


def summarize(values):
    return dict(n=len(values), median=float(statistics.median(values)),
                mean=float(statistics.mean(values)),
                p95=float(np.percentile(values, 95)), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(OUT.resolve()) or destination.exists():
        raise ValueError('Require a fresh output inside the current research session')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in ['0', '1']:
        raise ValueError('Exactly one permitted physical GPU is required')
    adapter_path, adapter = final_record(args.model)
    gate_path, gates = fixed_gates(args.model)
    source_hashes = {str(p): sha(p) for p in [adapter_path, gate_path]}
    split_path = OUT / 'splits.json'
    input_path = OLD / 'inputs.json'
    ids = set(json.loads(split_path.read_text())['validation'])
    unique = {}
    for sample in json.loads(input_path.read_text()):
        if sample['example_id'] in ids:
            unique.setdefault(sample['video_sha256'], sample)
    ordered = [unique[key] for key in sorted(unique)]
    indexes = np.linspace(0, len(ordered)-1, 12, dtype=int).tolist()
    samples = [ordered[index] for index in indexes]
    cfg = inference_config(args.model, 'video_text')
    cfg['batch_size'] = 1
    scope = 'last_frame' if args.model == 'qwen' else 'all_frames'
    condition = f'{scope}:target:32'
    ranks = ranking(args.model, 'video_text')
    loading_start = time.perf_counter()
    runtime = HeadDeltaRuntime(cfg)
    runtime.controller.fixed_values = gates['gate_values']
    runtime.controller.A = torch.tensor(adapter['A'], dtype=torch.float32, device=runtime.model.device)
    runtime.controller.B = torch.tensor(adapter['B'], dtype=torch.float32, device=runtime.model.device)
    torch.cuda.synchronize()
    loading_seconds = time.perf_counter() - loading_start
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Frozen inference only')
    original_prepare = runtime.prepare
    timing = {}

    def timed_prepare(*positional, **keyword):
        result = original_prepare(*positional, **keyword)
        torch.cuda.synchronize()
        timing['prepared_at'] = time.perf_counter()
        timing['sequence_tokens'] = int(result[0]['input_ids'].shape[-1])
        return result

    runtime.prepare = timed_prepare
    # Warm frame/ROI caches for both conditions, without reading labels.
    for sample in samples:
        runtime.active_ranking_prefix = 'ANSWER: '
        original_prepare([sample])
    runtime.active_ranking_prefix = ''
    for sample in samples[:2]:
        for mode in ['baseline', condition]:
            runtime.predict([sample], mode, ranks)
    torch.cuda.synchronize()
    records = []
    signatures = {}
    replay_equal = True
    for repeat in range(3):
        for index, sample in enumerate(samples):
            order = ['baseline', condition] if (repeat+index) % 2 == 0 else [condition, 'baseline']
            pair_inputs = []
            for mode in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                row, = runtime.predict([sample], mode, ranks)
                torch.cuda.synchronize()
                finish = time.perf_counter()
                if row['status'] != 'ok' or row['actual_forward_branches'] != 1:
                    raise ValueError('Valid single-forward output required')
                audit = row['token_audit']
                pair_inputs.append(audit['input_ids_sha256'])
                key = (sample['example_id'], mode)
                logits = row['native_class_logits_positive']
                if key in signatures:
                    replay_equal = replay_equal and signatures[key] == logits
                signatures[key] = logits
                records.append(dict(example_id=sample['example_id'], video_sha256=sample['video_sha256'],
                    repeat=repeat, mode='baseline' if mode == 'baseline' else 'target',
                    sequence_tokens=timing['sequence_tokens'], input_ids_sha256=pair_inputs[-1],
                    prepare_seconds=timing['prepared_at']-start,
                    model_and_readout_seconds=finish-timing['prepared_at'],
                    request_seconds=finish-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            if len(set(pair_inputs)) != 1:
                raise ValueError('Baseline and target input mismatch')
        print(args.model, 'completed timing repetition', repeat+1, '/ 3', flush=True)
    summary = {}
    for mode in ['baseline', 'target']:
        subset = [r for r in records if r['mode'] == mode]
        summary[mode] = {key: summarize([r[key] for r in subset]) for key in
            ['prepare_seconds', 'model_and_readout_seconds', 'request_seconds', 'peak_allocated_bytes']}
    for name, digest in source_hashes.items():
        if sha(Path(name)) != digest:
            raise ValueError('An existing checkpoint changed during timing')
    report = dict(status='complete', model=args.model, protocol='video_text', scope=scope, k=32,
        batch_size=1, independent_video_groups=12, repeats=3, labels_read=False,
        ordering='Alternating baseline/target order within each matched sample pair',
        sample_rule='12 evenly spaced sorted validation video SHA groups; first input per group; no labels',
        cuda_visible_devices=visible, gpu_name=torch.cuda.get_device_name(0),
        gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
        model_loading_seconds=loading_seconds, repeated_logits_exact=replay_equal,
        sequence_tokens=summarize([r['sequence_tokens'] for r in records]),
        includes='Warm PNG/track access, preprocessing, host-to-device transfer, model forward and research readout/diagnostics',
        excludes='Model load from request timing; video capture/decoding; ROI detection/tracking generation; network/queue; streaming cache implementation',
        source_hashes=source_hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary=summary, records=records)
    create_json(destination, report)
    print(json.dumps(dict(model=args.model, destination=str(destination), summary=summary,
                         repeated_logits_exact=replay_equal), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
