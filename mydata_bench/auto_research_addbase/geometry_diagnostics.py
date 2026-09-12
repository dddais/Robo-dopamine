"""Measure ROI quantization and same-video target-mask collisions, read-only."""
import json
from pathlib import Path
import numpy as np
from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from .prepare import OUT


def main():
    labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    results={}
    for protocol in ['image_text','text_image','video_text','official']:
        source=OLD/f'meter_{protocol}/predictions/baseline.jsonl'
        rows={}
        with source.open() as f:
            for line in f:
                r=json.loads(line);audit=r.get('token_audit',{})
                if not audit.get('target'):continue
                positions={p:i for i,p in enumerate(audit['visual'])}
                rows[r['example_id']]={s:{positions[p] for p in audit['target'][s]} for s in ['last_frame','all_frames']}
                rows[r['example_id']]['visual_keys']=len(positions)
        entry={'n':len(rows),'source':str(source),
               'visual_keys_median':float(np.median([r['visual_keys'] for r in rows.values()])),'scopes':{}}
        for scope in ['last_frame','all_frames']:
            pairs=[]
            for fid,row in rows.items():
                label=labels[fid]
                if label['split']!='fail' or label['source_suc_id'] not in rows:continue
                sid=label['source_suc_id']
                assert label['video_sha256']==labels[sid]['video_sha256']
                a,b=rows[sid][scope],row[scope]
                pairs.append({'suc_id':sid,'fail_id':fid,'jaccard':len(a&b)/len(a|b),
                              'identical':a==b,'suc_keys':len(a),'fail_keys':len(b)})
            sizes=[len(r[scope]) for r in rows.values()]
            entry['scopes'][scope]={'target_key_count_quantiles':np.quantile(sizes,[0,.25,.5,.75,1]).tolist(),
                 'pairs':len(pairs),'identical_masks':sum(p['identical'] for p in pairs),
                 'jaccard_ge_0p75':sum(p['jaccard']>=.75 for p in pairs),
                 'jaccard_mean':float(np.mean([p['jaccard'] for p in pairs])),
                 'one_or_two_key_targets':sum(n<=2 for n in sizes),'pair_details':pairs}
        results[protocol]=entry
        print(protocol,'visual',entry['visual_keys_median'])
        for s,r in entry['scopes'].items():print(s,{k:v for k,v in r.items() if k!='pair_details'})
    create_json(OUT/'analysis/roi_quantization_v1.json',results)

if __name__=='__main__':main()
