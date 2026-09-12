"""Reconstruct complete native summary statistics without a new label lookup."""
import json
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .head_delta_worker import BASE, sha, verify_sources


def close(a,b):
    if not np.isclose(a,b,rtol=0,atol=1e-12):raise ValueError(f'Summary identity failed: {a} != {b}')


def verify_summary(record):
    n=record['n'];hist=record['ordinal_prediction_distributions']
    if n!=record['expected'] or record['invalid'] or record['missing_ids'] or record['invalid_ids']:
        raise ValueError('Complete native cohort required')
    if set(hist)!=set(['all','suc','fail']) or hist['suc']['n']+hist['fail']['n']!=n:
        raise ValueError('Native class population does not reconstruct')
    for label,h in hist.items():
        if set(h['counts'])!=set(map(str,range(1,6))) or sum(h['counts'].values())!=h['n']:
            raise ValueError('All five native classes must be retained')
        if h['n']:
            for k,count in h['counts'].items():close(h['rates'][k],count/h['n'])
    for k in map(str,range(1,6)):
        if hist['all']['counts'][k]!=hist['suc']['counts'][k]+hist['fail']['counts'][k]:
            raise ValueError('Class histogram sums differ')
    if n:
        error=sum((5-k)*hist['suc']['counts'][str(k)]+(k-1)*hist['fail']['counts'][str(k)] for k in range(1,6))
        close(record['mae'],error/n)
    if record['accuracy']['0.125/0.875']!=record['accuracy']['0.2/0.8']:
        raise ValueError('Native five-class endpoints should give identical threshold counts')
    correct={'suc':hist['suc']['counts']['5'],'fail':hist['fail']['counts']['1']}
    correct['all']=correct['suc']+correct['fail']
    for label in correct:
        a=record['accuracy']['0.125/0.875'][label]
        if a['correct']!=correct[label] or a['expected']!=hist[label]['n'] or a['valid']!=hist[label]['n']:
            raise ValueError('Endpoint accuracy does not reconstruct from native class histograms')
        if a['expected']:close(a['rate_all_expected'],a['correct']/a['expected'])
    pair=record['pairwise'];counts={key:0 for key in ['<0','0','1','2','3','4']}
    if len(pair['pairs'])!=pair['n']:raise ValueError('Pair count differs')
    for row in pair['pairs']:
        delta=row['ordinal_delta'];close(delta,4*row['continuous_delta'])
        if int(delta)!=delta or not -4<=delta<=4:raise ValueError('Invalid native ordinal pair gap')
        counts['<0' if delta<0 else str(int(delta))]+=1
    if counts!=pair['ordinal_difference_counts']:raise ValueError('Six-bin paired distribution differs')
    if pair['n']:
        close(pair['continuous_mean_delta'],sum(p['continuous_delta'] for p in pair['pairs'])/pair['n'])
        close(pair['continuous_positive_rate'],sum(counts[str(k)] for k in range(1,5))/pair['n'])


def main():
    path=BASE/'coverage.json';decision=json.loads(path.read_text());verify_sources(decision['sources_sha256'])
    sources={str(path):sha(path)};checks=[]
    for population,artifact in decision['artifacts'].items():
        root=Path(artifact['checkpoint']);detail_path=root/'details.json';point_path=root/'points.json'
        sources[str(detail_path)]=sha(detail_path);sources[str(point_path)]=sha(point_path)
        details=json.loads(detail_path.read_text());points=json.loads(point_path.read_text())
        lookup={f"{p['experiment']}/{p['method']}/{p['condition']}/{p['field']}":p for p in points if p['threshold']=='0.125/0.875'}
        if set(details)!=set(lookup) or len(details)!=sum(len(e['point']['ks']) for e in decision['inputs']):
            raise ValueError('Every frozen target condition must be in the complete report')
        for key,row in details.items():
            point=lookup[key];baseline=row['baseline'];intervention=row['intervention']
            for aggregate,by_task in [(baseline,row['baseline_by_task']),(intervention,row['by_task'])]:
                verify_summary(aggregate)
                for task in by_task.values():verify_summary(task)
                if sum(t['n'] for t in by_task.values())!=aggregate['n']:raise ValueError('Task sample support does not sum')
                close(sum(t['mae']*t['n'] for t in by_task.values())/aggregate['n'],aggregate['mae'])
                for label in ['all','suc','fail']:
                    a=aggregate['accuracy']['0.125/0.875'][label]
                    for count in ['correct','expected','valid']:
                        if sum(t['accuracy']['0.125/0.875'][label][count] for t in by_task.values())!=a[count]:
                            raise ValueError('Task-specific accuracy counts do not sum')
            if set(row['by_task'])!=set(row['baseline_by_task']):raise ValueError('Task membership differs')
            changes=[]
            for task,value in row['by_task'].items():
                before=row['baseline_by_task'][task]
                if value['n']!=before['n']:raise ValueError('Task sample count changed between arms')
                changes.append(value['mae']-before['mae'])
            pairs=lambda a:{(p['suc_id'],p['fail_id']) for p in a['pairwise']['pairs']}
            if pairs(baseline)!=pairs(intervention):raise ValueError('Actual pair membership differs between arms')
            close(point['delta_mae'],intervention['mae']-baseline['mae'])
            close(point['delta_task_macro_mae'],float(np.mean(changes)))
            for label in ['all','suc','fail']:
                a=intervention['accuracy']['0.125/0.875'][label]['rate_all_expected']
                b=baseline['accuracy']['0.125/0.875'][label]['rate_all_expected']
                close(point[f'delta_{label}'],a-b)
            checks.append(dict(population=population,condition=key,n=baseline['n'],
                suc_n=baseline['ordinal_prediction_distributions']['suc']['n'],fail_n=baseline['ordinal_prediction_distributions']['fail']['n'],
                tasks_improved=sum(x<0 for x in changes),tasks_worse=sum(x>0 for x in changes),tasks_unchanged=sum(x==0 for x in changes),
                delta_micro_mae=point['delta_mae'],delta_macro_mae=point['delta_task_macro_mae'],
                micro_macro_sign_reversal=point['delta_mae']*point['delta_task_macro_mae']<0,
                native_middle_predictions=sum(intervention['ordinal_prediction_distributions']['all']['counts'][str(k)] for k in [2,3,4]),
                paired_observations=baseline['pairwise']['n'],class_histogram_mae_accuracy_pair_bins_and_task_reconstruction=True))
    verify_sources(sources);dest=OUT/'audit'/time.strftime('head_delta_statistical_structure_%Y%m%d_%H%M%S.json')
    create_json(dest,dict(status='pass',new_label_lookup=False,sources_sha256=sources,checks=checks,
        distinct_target_configurations=sum(len(e['point']['ks']) for e in decision['inputs']),overlapping_populations=3,
        interpretation='Arithmetic and support checks on all frozen native target reports. Configurations and overlapping populations are not independent replicates. This does not correct adaptive selection bias.'))
    print(dest,flush=True)


if __name__=='__main__':main()
