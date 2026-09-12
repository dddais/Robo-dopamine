"""Verify real anchored forwards against recorded branches, without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .derive_anchor import derived_row
from .branch_ablation import native_branch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiments',nargs='+',required=True)
    args=parser.parse_args()
    checked=[];sources={}
    def source(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path)
    for experiment in args.experiments:
        folder=OUT/'experiments'/experiment/'discovery_smoke'
        cfg=json.loads((folder/'runtime_config.json').read_text())
        if cfg.get('contrast_reference')!='baseline':raise ValueError('Not a baseline anchor experiment')
        ids=json.loads((folder/'requested_ids.json').read_text())
        if len(ids)!=8:raise ValueError('Use the frozen matching-eight-sample smoke')
        if not (folder/'worker_events.jsonl').exists():raise ValueError('Smoke is not complete')
        current_base=source(folder/'predictions/baseline.jsonl')
        original=OUT/'experiments'/f"{cfg['model']}_{cfg['protocol']}_evidence_a1"/'discovery'
        original_cfg=json.loads((original/'runtime_config.json').read_text())
        if cfg['batch_size']!=original_cfg['batch_size']:raise ValueError('Batch size differs')
        original_base=source(original/'predictions/baseline.jsonl')
        for path in sorted(folder.glob('mass_transport_s*/predictions/*_target_*.jsonl')):
            current=source(path)
            if set(current)!=set(ids):raise ValueError('Partial actual forward condition')
            prior=source(original/path.parents[1].name/'predictions'/path.name)
            maxima={'own_reconstruction':0.,'derived_progress':0.,'derived_success_probability':0.,
                    'baseline_logits':0.,'positive_logits':0.,'negative_logits':0.,'success_logits':0.}
            budget=int(path.stem.rsplit('_',1)[1])
            for eid in ids:
                row=current[eid];base=current_base[eid]
                if row['status']!='ok' or base['status']!='ok':raise ValueError('Forward failed')
                for old,new in [(original_base[eid],base),(prior[eid],row)]:
                    if old['token_audit']['input_ids_sha256']!=new['token_audit']['input_ids_sha256']:
                        raise ValueError('Matching-batch input IDs differ')
                reference=np.asarray(row['native_class_logits_reference'],dtype=np.float32)
                if not np.array_equal(reference,np.asarray(base['native_class_logits_positive'],dtype=np.float32)):
                    raise ValueError('Reference is not the same-batch unsteered baseline')
                if row['reference_attention_diagnostics']:raise ValueError('Reference was steered')
                for diagnostics in ['attention_diagnostics','negative_attention_diagnostics']:
                    heads={(int(layer),head) for layer,diag in row[diagnostics].items() for head in diag['heads']}
                    if len(heads)!=budget:raise ValueError('Wrong positive or negative head budget')
                for name,old,new in [('baseline_logits',original_base[eid]['native_class_logits_positive'],base['native_class_logits_positive']),
                                     ('positive_logits',prior[eid]['native_class_logits_positive'],row['native_class_logits_positive']),
                                     ('negative_logits',prior[eid]['native_class_logits_negative'],row['native_class_logits_negative'])]:
                    maxima[name]=max(maxima[name],float(np.max(np.abs(np.asarray(old)-np.asarray(new)))))
                derived=derived_row(original_base[eid],prior[eid],cfg['model'],cfg['contrast_weight'])
                own=derived_row(base,row,cfg['model'],cfg['contrast_weight'])
                for field in ['progress']+(['success_probability'] if cfg['model']=='meter' else []):
                    maxima['own_reconstruction']=max(maxima['own_reconstruction'],abs(row[field]-own[field]))
                    maxima['derived_'+field]=max(maxima['derived_'+field],abs(row[field]-derived[field]))
                if cfg['model']=='meter':
                    if row['success_logit_reference']!=base['success_logit_positive']:
                        raise ValueError('Native success anchor differs from baseline')
                    for branch in ['positive','negative']:
                        maxima['success_logits']=max(maxima['success_logits'],abs(row['success_logit_'+branch]-prior[eid]['success_logit_'+branch]))
            if max(maxima.values())>=1e-5:raise ValueError(f'Actual-forward mismatch requires investigation: {experiment}/{path.name}: {maxima}')
            checked.append({'experiment':experiment,'condition':path.stem,'method':path.parents[1].name,
                            'n':len(ids),'batch_size':cfg['batch_size'],'budget_per_steered_branch':budget,
                            'reference_is_unsteered':True,'all_classes_preserved':10 if cfg['model']=='meter' else 5,
                            'maximum_differences':maxima})
    destination=OUT/'audit'/time.strftime('anchor_actual_forward_%Y%m%d_%H%M%S.json')
    create_json(destination,{'status':'pass','arguments':vars(args),'checks':checked,'sources':sources,
                             'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'labels_read':False,'tolerance':1e-5})
    print(destination)
    print(json.dumps(checked,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
