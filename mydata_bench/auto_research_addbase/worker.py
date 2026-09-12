"""An independently resumable GPU worker, not an autonomous model agent."""
import argparse
import json
import time
from pathlib import Path
from mydata_bench.addbase_eval.run import append, latest, batches, predict_condition, rank
from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from .prepare import OUT, CONFIGS
from .runtime import ResearchRuntime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--protocols',nargs='+',required=True)
    parser.add_argument('--population',default='discovery')
    parser.add_argument('--methods',nargs='+',default=['mass_transport'])
    parser.add_argument('--strengths',nargs='+',type=float,default=[2,4,8])
    parser.add_argument('--ks',nargs='+',type=int,default=[8,32,64])
    parser.add_argument('--scopes',nargs='+',default=['all_frames','last_frame'])
    parser.add_argument('--controls',nargs='+',default=['target'])
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--variant',default='')
    parser.add_argument('--ranking-prefix',default='')
    parser.add_argument('--contrast-weight',type=float)
    parser.add_argument('--contrast-kl-budget',type=float)
    parser.add_argument('--contrast-reference',choices=['positive','baseline'],default='positive')
    parser.add_argument('--negative-strength',type=float,default=4)
    parser.add_argument('--contrast-negative-mode',choices=['visual','visual_and_task','task','joint_interaction','bias_task_anchor','task_content_block','resolution','head_output','learned_gates'],default='visual')
    parser.add_argument('--learned-gate-file',type=Path)
    parser.add_argument('--content-block-probe',action='store_true')
    parser.add_argument('--resolution-high-max-pixels',type=int)
    parser.add_argument('--negative-task-strength',type=float,default=4)
    parser.add_argument('--research-query-scope',choices=['all','text','readout'],default='all')
    parser.add_argument('--task-binding-fraction',type=float)
    parser.add_argument('--task-binding-distribution',choices=['attention','uniform'],default='attention')
    parser.add_argument('--visual-mass-partition',choices=['global','temporal_planes'],default='global')
    parser.add_argument('--context-radius',type=float)
    parser.add_argument('--ranking-strategy',choices=['raw_mass','dual_mass'],default='raw_mass')
    parser.add_argument('--frozen-ranking-root',type=Path)
    args=parser.parse_args()
    if args.learned_gate_file is not None and args.contrast_negative_mode!='learned_gates':
        raise ValueError('Learned gates require their separate supervised variant')
    learned_record=None
    if args.contrast_negative_mode=='learned_gates':
        expected_file=OUT/'learned_attention_gates_v1/training'/args.model/'final_gates.json'
        if (args.variant!='learned_attention_gates_v1' or args.model not in {'qwen','roboreward'}
                or args.learned_gate_file is None or args.learned_gate_file.resolve()!=expected_file.resolve()
                or args.methods!=['learned_gate_bias'] or args.strengths!=[6.] or args.contrast_weight!=0
                or args.contrast_reference!='positive' or args.task_binding_fraction is not None
                or args.task_binding_distribution!='attention' or args.contrast_kl_budget is not None
                or args.batch_size!=8 or args.ranking_prefix!='ANSWER: ' or args.frozen_ranking_root is None
                or args.frozen_ranking_root.resolve()!=(OUT/'functional_selections'/f'stage8_{args.model}_v1').resolve()
                or args.visual_mass_partition!='global' or args.context_radius is not None
                or args.research_query_scope!='all' or args.ranking_strategy!='raw_mass'):
            raise ValueError('Require final fixed-step learned gates, original heads and native five-class readout')
        import hashlib
        learned_record=json.loads(expected_file.read_text())
        if (learned_record['status']!='complete' or learned_record['model']!=args.model
                or learned_record['variant']!='learned_attention_gates_v1'
                or learned_record['forward_backward_steps']!=4200 or learned_record['optimizer_steps']!=525
                or learned_record['trained_parameter_count']!=56 or learned_record['backbone_requires_grad_count']!=0
                or learned_record['validation_labels_used'] or learned_record['selection_rule']!='Final fixed step only'
                or learned_record['policy_sha256']!=hashlib.sha256((OUT/'selection_learned_attention_gates_v1.json').read_bytes()).hexdigest()):
            raise ValueError('Trained gate artifact does not match the frozen learning protocol')
    if args.contrast_negative_mode=='head_output' and (
            args.variant!='head_output_contrast_norm_a1' or args.model not in {'qwen','roboreward'}
            or args.methods!=['binding_transport'] or args.strengths!=[4.] or args.contrast_weight!=0
            or args.negative_strength!=4 or args.negative_task_strength!=4 or args.task_binding_fraction!=.5
            or args.task_binding_distribution!='uniform' or args.contrast_kl_budget is not None
            or args.ranking_prefix!='ANSWER: ' or args.frozen_ranking_root is None or args.batch_size!=8
            or args.visual_mass_partition!='global' or args.context_radius is not None
            or args.research_query_scope!='all' or args.ranking_strategy!='raw_mass'
            or args.contrast_reference!='positive'):
        raise ValueError('Head-output contrast requires its frozen local formula and no logit contrast')
    if args.resolution_high_max_pixels is not None and args.contrast_negative_mode!='resolution':
        raise ValueError('Resolution budgets only apply to the independent resolution variant')
    if args.contrast_negative_mode=='resolution' and (
            args.variant!='resolution_evidence_a1' or args.resolution_high_max_pixels!=200704
            or args.model not in {'qwen','roboreward'} or args.methods!=['bias'] or args.strengths!=[6.]
            or args.contrast_weight!=1 or args.contrast_reference!='positive' or args.task_binding_fraction is not None
            or args.task_binding_distribution!='attention' or args.contrast_kl_budget is not None
            or args.ranking_prefix!='ANSWER: ' or args.frozen_ranking_root is None or args.batch_size!=8
            or args.visual_mass_partition!='global' or args.context_radius is not None
            or args.research_query_scope!='all' or args.ranking_strategy!='raw_mass'):
        raise ValueError('Resolution contrast requires the fixed high/low budgets, original bias6 and weight1')
    if args.content_block_probe and (args.contrast_negative_mode!='task_content_block' or args.limit!=8
            or args.variant!='task_content_block_evidence_a1_probe'):
        raise ValueError('Instruction perturbation probe requires its separate eight-example smoke variant')
    if args.contrast_negative_mode=='task_content_block' and (
            args.variant not in ['task_content_block_evidence_a1','task_content_block_evidence_a1_probe']
            or args.model not in {'qwen','roboreward'} or args.methods!=['bias'] or args.strengths!=[6.]
            or args.contrast_weight!=1 or args.contrast_reference!='positive' or args.task_binding_fraction is not None
            or args.task_binding_distribution!='attention' or args.contrast_kl_budget is not None
            or args.ranking_prefix!='ANSWER: ' or args.frozen_ranking_root is None
            or args.visual_mass_partition!='global' or args.context_radius is not None
            or args.research_query_scope!='all' or args.ranking_strategy!='raw_mass'):
        raise ValueError('Content-block mode requires frozen original-bias6, weight1 and explicitly aligned task keys')
    if args.contrast_kl_budget is not None and (
            not args.variant or args.model not in {'qwen','roboreward'} or args.contrast_kl_budget not in [.05,.2,.8]
            or args.contrast_negative_mode!='visual_and_task' or args.contrast_weight!=2
            or args.contrast_reference!='positive' or args.strengths!=[4.] or args.negative_strength!=4
            or args.negative_task_strength!=4 or args.task_binding_fraction!=.5
            or args.task_binding_distribution!='uniform' or args.methods!=['binding_transport']
            or args.visual_mass_partition!='global' or args.frozen_ranking_root is None):
        raise ValueError('KL mode requires the registered three native branches, cap2 and fixed heads')
    if args.visual_mass_partition != 'global':
        if (not args.variant or args.model not in {'qwen','roboreward'} or args.methods != ['binding_transport']
                or args.task_binding_distribution != 'uniform' or args.task_binding_fraction is None
                or args.contrast_weight is None or args.contrast_negative_mode != 'visual'
                or args.contrast_reference != 'positive'):
            raise ValueError('Temporal-plane variant requires explicit two-branch uniform binding on a discrete model')
    if 'binding_task_suppression' in args.methods:
        raise ValueError('Task suppression is an internal counterfactual branch, not a positive intervention')
    if args.contrast_negative_mode not in {'visual','task_content_block','resolution','learned_gates'}:
        allowed_model=(args.model in {'qwen','roboreward'} or
                       (args.model=='meter' and args.contrast_negative_mode=='visual_and_task'))
        if (not args.variant or not allowed_model or args.contrast_weight is None
                or args.contrast_reference!='positive' or args.methods!=['binding_transport']
                or args.task_binding_distribution!='uniform' or args.task_binding_fraction is None
                or not 0<args.negative_task_strength<=8):
            raise ValueError('Factorized counterfactuals require a separate uniform-binding native-class variant')
    elif args.negative_task_strength!=4:
        raise ValueError('Task-negative strength only applies to the explicit factorized variant')
    if args.contrast_negative_mode=='task' and (args.strengths != [0.] or args.visual_mass_partition!='global'):
        raise ValueError('Pure task-domain contrast must explicitly bypass visual steering with sole strength zero')
    if args.contrast_negative_mode in {'joint_interaction','bias_task_anchor'} and (
            args.strengths != [4.] or args.negative_strength != 4 or args.negative_task_strength != 4
            or args.task_binding_fraction != .5 or args.contrast_weight != 1
            or args.visual_mass_partition != 'global' or args.frozen_ranking_root is None):
        raise ValueError('This branch design requires its registered cells, weight1 and frozen heads')
    if args.task_binding_distribution!='attention' and (not args.variant or args.task_binding_fraction is None):
        raise ValueError('Changed instruction distribution requires a separate variant and binding fraction')
    if args.contrast_reference!='positive' and (not args.variant or args.contrast_weight is None):
        raise ValueError('Baseline-anchored contrast requires a separate variant and explicit weight')
    if args.context_radius is not None and (not args.variant or not 0<args.context_radius<=1):
        raise ValueError('Spatial context requires a separate variant and a radius in (0,1]')
    if 'context_transport' in args.methods and args.context_radius is None:
        raise ValueError('Spatial context requires an explicit frozen radius')
    # Import optional inference modules now so a long-lived worker does not
    # acquire later source edits when it reaches another protocol.
    from . import binding, dual_rank, evidence, context, interaction, bias_task_anchor, factorized_kl_runtime, task_content_block, resolution_evidence, head_output_contrast, learned_attention_gates
    from .provenance import snapshot
    startup=snapshot(vars(args))
    split=json.loads((OUT/'splits.json').read_text())
    all_samples=json.loads((OLD/'inputs.json').read_text())
    requested=set(split[args.population])
    samples=[s for s in all_samples if s['example_id'] in requested]
    if args.limit:samples=samples[:args.limit]
    runtime=None
    for protocol in args.protocols:
        cfg=json.loads((CONFIGS/f'{args.model}_{protocol}.json').read_text())
        if args.ranking_prefix and (not args.variant or args.model not in {'qwen','roboreward'}):
            raise ValueError('Answer-query ranking requires a separate variant and a discrete model')
        if args.variant:
            cfg['output_dir']+='_'+args.variant
            cfg['ranking_prefix']=args.ranking_prefix
            if args.contrast_negative_mode=='task_content_block':
                cfg.update(require_task_positions=True,content_block_probe=args.content_block_probe,method='bias')
            if args.contrast_negative_mode=='learned_gates':
                cfg.update(require_task_positions=True,method='learned_gate_bias',
                    learned_gate_file=str(args.learned_gate_file.resolve()),
                    learned_gate_sha256=hashlib.sha256(args.learned_gate_file.read_bytes()).hexdigest())
            if args.contrast_negative_mode=='resolution':
                if cfg['max_pixels']!=50176:raise ValueError('Original low resolution differs from the frozen budget')
                cfg.update(resolution_high_max_pixels=args.resolution_high_max_pixels,method='bias')
            if args.contrast_kl_budget is not None:cfg['contrast_kl_budget']=args.contrast_kl_budget
            if args.research_query_scope!='all':cfg['research_query_scope']=args.research_query_scope
            if args.task_binding_fraction is not None:cfg['task_binding_fraction']=args.task_binding_fraction
            if args.task_binding_distribution!='attention':cfg['task_binding_distribution']=args.task_binding_distribution
            if args.visual_mass_partition!='global':cfg['visual_mass_partition']=args.visual_mass_partition
            if args.context_radius is not None:cfg['context_radius']=args.context_radius
            if args.ranking_strategy!='raw_mass':
                if args.task_binding_fraction is None:raise ValueError('Dual ranking requires task token alignment')
                cfg['ranking_strategy']=args.ranking_strategy
            if args.frozen_ranking_root:
                if not args.frozen_ranking_root.resolve().is_relative_to(OUT.resolve()):
                    raise ValueError('Frozen rankings must belong to this research session')
                cfg['frozen_ranking_source']=str(args.frozen_ranking_root/f'{args.model}_{protocol}')
            if args.contrast_weight is not None:
                cfg.update(readout='evidence_contrast',contrast_weight=args.contrast_weight,
                           negative_strength=args.negative_strength)
                if args.contrast_reference!='positive':cfg['contrast_reference']=args.contrast_reference
                if args.contrast_negative_mode!='visual':
                    cfg.update(contrast_negative_mode=args.contrast_negative_mode,negative_task_strength=args.negative_task_strength)
            create_json(CONFIGS/f'{args.model}_{protocol}_{args.variant}.json',cfg)
        elif args.contrast_weight is not None:
            raise ValueError('Evidence readout requires a separate variant and matched baseline')
        elif args.research_query_scope!='all':
            raise ValueError('Changed query scope requires an explicitly named separate variant')
        elif args.task_binding_fraction is not None:
            raise ValueError('Task binding requires a separate variant')
        elif args.ranking_strategy!='raw_mass':
            raise ValueError('Changed ranking requires a separate variant')
        elif args.frozen_ranking_root:
            raise ValueError('Frozen supervised rankings require a separate variant')
        if args.batch_size:cfg['batch_size']=args.batch_size
        folder=Path(cfg['output_dir'])/f'{args.population}{"_smoke" if args.limit else ""}'
        create_json(folder/'runtime_config.json',cfg)
        create_json(folder/'requested_ids.json',[s['example_id'] for s in samples])
        create_json(folder/'run_intents'/f'{startup["run_id"]}.json',startup)
        if runtime is None:
            if cfg.get('contrast_negative_mode')=='learned_gates':
                runtime=learned_attention_gates.LearnedGateRuntime(cfg)
                runtime.controller.fixed_values=learned_record['gate_values']
                runtime.controller.values(runtime.model.device)
            elif cfg.get('contrast_negative_mode')=='head_output':
                runtime=head_output_contrast.HeadOutputRuntime(cfg)
            elif cfg.get('contrast_negative_mode')=='resolution':
                runtime=resolution_evidence.ResolutionRuntime(cfg)
            elif cfg.get('contrast_negative_mode')=='task_content_block':
                runtime=task_content_block.TaskContentRuntime(cfg)
            elif cfg.get('contrast_kl_budget') is not None:
                runtime=factorized_kl_runtime.FactorizedKLRuntime(cfg)
            elif cfg.get('contrast_negative_mode')=='bias_task_anchor':
                runtime=bias_task_anchor.BiasTaskAnchorRuntime(cfg)
            elif cfg.get('contrast_negative_mode')=='joint_interaction':
                runtime=interaction.InteractionRuntime(cfg)
            elif cfg.get('readout')=='evidence_contrast':
                from .evidence import EvidenceRuntime
                runtime=EvidenceRuntime(cfg)
            else:runtime=ResearchRuntime(cfg)
        else:runtime.cfg.clear();runtime.cfg.update(cfg)
        create_json(folder/'loading_audit.json',runtime.model.loading_audit)
        baseline=predict_condition(runtime,samples,'baseline',None,folder)
        rank_folder=Path(cfg['output_dir'])/'ranking'
        if args.frozen_ranking_root:
            rankings={scope:json.loads((Path(cfg['frozen_ranking_source'])/f'ranking_{scope}.json').read_text())
                      for scope in ['last_frame','all_frames']}
            for scope,value in rankings.items():create_json(rank_folder/f'ranking_{scope}.json',value)
        elif args.model in {'meter','sole'} and not args.variant:
            rankings={scope:json.loads((OLD/f'{args.model}_{protocol}'/f'ranking_{scope}.json').read_text())
                      for scope in ['last_frame','all_frames']}
            for scope,value in rankings.items():create_json(rank_folder/f'ranking_{scope}.json',value)
        else:
            rank_samples=[s for s in all_samples if s['ranking']]
            ranking_baseline=(predict_condition(runtime,rank_samples,'baseline',None,rank_folder)
                              if args.model=='sole' and protocol=='official' else None)
            if args.ranking_strategy=='dual_mass':
                from .dual_rank import dual_rank
                rankings=dual_rank(runtime,rank_samples,ranking_baseline,rank_folder)
            else:rankings=rank(runtime,rank_samples,ranking_baseline,rank_folder)
        for method in args.methods:
            for strength in args.strengths:
                runtime.cfg.update(method=method,bias=strength)
                output=folder/f'{method}_s{strength:g}'
                create_json(output/'intervention_config.json',dict(runtime.cfg))
                for scope in args.scopes:
                    for k in args.ks:
                        for kind in args.controls:
                            predict_condition(runtime,samples,f'{scope}:{kind}:{k}',rankings,output)
        append(folder/'worker_events.jsonl',[{'event':'complete','time':time.time(),'arguments':startup['arguments']}])

if __name__=='__main__':main()
