"""Generate reviewable tables and scientific plots after independent scoring."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import numpy as np
from .prepare import OUT


def number(value):return 'NA' if value is None else f'{value:.4f}'
def pct(value):return 'NA' if value is None else f'{value:.2%}'
def accuracy(data,threshold,split):
    r=data.get('accuracy',{}).get(threshold,{}).get(split,{})
    return f"{r.get('correct',0)}/{r.get('expected',0)} ({pct(r.get('rate_all_expected'))})"
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+[
            '| '+' | '.join(map(str,row))+' |' for row in rows]


def create_text(path,text):
    data=text.encode('utf-8')
    if path.exists():
        if path.read_bytes()!=data:raise FileExistsError(f'Use a new report suffix: {path}')
        return
    with path.open('xb') as f:f.write(data)


def population_task_tables(experiments):
    """Export explicit full/cohort/holdout denominators, including empty valid sets."""
    inputs=json.loads((OUT/'inputs.json').read_text())
    labels=json.loads((OUT/'labels_for_scoring_only.json').read_text())
    expected=Counter()
    for sample in inputs:
        label=labels[sample['example_id']]
        for population in ['full']+(['cohort'] if sample['cohort'] else [])+(['holdout'] if sample['cohort'] and sample['holdout'] else []):
            expected[(population,label['subset'],'all')]+=1
            expected[(population,label['subset'],label['split'])]+=1
    metrics=[];distributions=[]
    for name,result in experiments.items():
        for condition,record in result['conditions'].items():
            for population,data in record.items():
                for task,t in data['by_task'].items():
                    assert t['expected']==expected[(population,task,'all')]
                    bounds=t.get('mae_all_expected_bounds',[0.,4.])
                    row={'experiment':name,'condition':condition,'population':population,'task':task,
                         'expected':t['expected'],'valid':t['n'],'invalid':t['invalid'],'mae_valid':t.get('mae'),
                         'mae_all_expected_lower':bounds[0],'mae_all_expected_upper':bounds[1]}
                    for threshold in ['0.125/0.875','0.2/0.8']:
                        for split in ['all','suc','fail']:
                            n=expected[(population,task,split)]
                            acc=t.get('accuracy',{}).get(threshold,{}).get(split,{})
                            if acc:assert acc['expected']==n
                            correct=acc.get('correct',0)
                            row[f'expected_{split}']=n
                            row[f'correct_{threshold}_{split}']=correct
                            row[f'acc_fixed_{threshold}_{split}']=correct/n if n else None
                    metrics.append(row)
                    for binning,field in [('equal_width_progress','prediction_distributions'),('ordinal_reward','ordinal_prediction_distributions')]:
                        if field not in t and t['n']>0:continue  # Older checkpoints did not export both definitions.
                        for split in ['all','suc','fail']:
                            dist=t.get(field,{}).get(split,{})
                            n=expected[(population,task,split)];valid=dist.get('n',0)
                            counts={str(k):dist.get('counts',{}).get(str(k),0) for k in range(1,6)}
                            assert sum(counts.values())==valid and valid<=n
                            distributions.append({'experiment':name,'condition':condition,'population':population,
                                'task':task,'split':split,'binning':binning,'expected':n,'valid':valid,'invalid':n-valid,
                                **{f'prediction_{k}':v for k,v in counts.items()},
                                **{f'rate_valid_{k}':v/valid if valid else None for k,v in counts.items()},
                                **{f'rate_fixed_{k}':v/n if n else None for k,v in counts.items()}})
    return metrics,distributions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis-name',default='analysis_v1')
    parser.add_argument('--record-suffix',default='');args=parser.parse_args()
    folder=OUT/args.analysis_name
    index=json.loads((folder/'index.json').read_text())
    holm=json.loads((folder/'holm.json').read_text())
    experiments={name:json.loads((folder/(name+'.json')).read_text()) for name in index['experiments']}
    baseline_rows=[];candidate_rows=[];holdout_rows=[];control_rows=[];task_rows=[];distribution_rows=[];pair_rows=[]
    for name,result in experiments.items():
        conditions=result['conditions']
        full=conditions['baseline']['full']
        baseline_rows.append([name,f"{full['n']}/{full['expected']}",number(full.get('mae')),
                              str(full.get('mae_all_expected_bounds')),
                              *[accuracy(full,'0.125/0.875',s) for s in ['all','suc','fail']],
                              *[accuracy(full,'0.2/0.8',s) for s in ['all','suc','fail']]])
        lines=[f'# {name} 实验记录','',
               '本表从原始结果重新统计。准确率以固定期望样本数为分母，解析失败/不可用条件不计正确；MAE 为有效输出的描述性平均，另给全部样本误差上下界。',
               'MAE 奖励档边界为 .125/.375/.625/.875；另有 .2/.4/.6/.8 等宽进度分布。两种分布分别保存，不混用。',
               '官方 SOLE 预测为绝对进度，前一步来自同一条件自己的模型输出。','',
               '## Baseline 全量','']
        lines+=table(['配置','有效/期望','MAE','全体 MAE 界','acc 严格 all','suc','fail','acc 宽 all','suc','fail'],[baseline_rows[-1]])
        for population in ['cohort','holdout']:
            rows=[]
            for condition,record in conditions.items():
                data=record[population];change=data.get('paired_change',{})
                row=[condition,f"{data['n']}/{data['expected']}",number(data.get('mae')),
                     *[accuracy(data,'0.125/0.875',s) for s in ['all','suc','fail']],
                     *[accuracy(data,'0.2/0.8',s) for s in ['all','suc','fail']],
                     number(change.get('mae_delta')),str(change.get('mae_delta_ci95','NA')),
                     number(holm[population]['adjusted_p'].get(name+'/'+condition))]
                rows.append(row)
                if ':target:' in condition:
                    (candidate_rows if population=='cohort' else holdout_rows).append([name]+row)
            lines+=['',f'## {population} 同组比较','']+table(
                    ['条件','有效/期望','MAE','严格 all','suc','fail','宽 all','suc','fail','配对 ΔMAE','95% CI','Holm p'],rows)
        lines+=['','## 四条件共同有效样本上的控制比较','']
        controls=[]
        for key,record in result.get('matched_controls',{}).items():
            data=record['cohort'];scope,k=key.split(':')
            names=['baseline']+[f'{scope}:{kind}:{k}' for kind in ['target','wrong_region','low_rank']]
            row=[key,f"{data['common_n']}/{data['expected_population']}"]+[number(data['conditions'][c].get('mae')) for c in names]
            controls.append(row);control_rows.append([name]+row)
        lines+=table(['范围/k','四组共同 n / cohort','baseline MAE','target MAE','wrong MAE','low-rank MAE'],controls)
        if result.get('secondary_success_head'):
            lines+=['','## Robometer success head（次要描述指标，主指标仍为 progress）','']
            secondary=[]
            for condition,record in result['secondary_success_head'].items():
                data=record['cohort']
                secondary.append([condition,f"{data['n']}/{data['expected']}",number(data.get('mae')),
                                  *[accuracy(data,'0.125/0.875',s) for s in ['all','suc','fail']],
                                  *[accuracy(data,'0.2/0.8',s) for s in ['all','suc','fail']]])
            lines+=table(['条件','有效/期望','success MAE','严格 all','suc','fail','宽 all','suc','fail'],secondary)
        lines+=['','## 同视频 suc−fail 配对','']
        pair_table=[]
        for condition,record in conditions.items():
            data=record['cohort'];pairs=data.get('pairwise',{})
            row=[condition,pairs.get('n',0),number(pairs.get('continuous_mean_delta')),
                 pct(pairs.get('continuous_negative_rate')),pct(pairs.get('continuous_strong_positive_ge_0p5_rate')),
                 *[pairs.get('ordinal_difference_counts',{}).get(k,0) for k in ['<0','0','1','2','3','4']]]
            pair_table.append(row);pair_rows.append([name]+row)
            for task,t in data.get('by_task',{}).items():
                task_rows.append({'experiment':name,'condition':condition,'task':task,'expected':t['expected'],'valid':t['n'],
                                  'mae':t.get('mae'),**{f'acc_{threshold}_{split}':t.get('accuracy',{}).get(threshold,{}).get(split,{}).get('rate_all_expected')
                                  for threshold in ['0.125/0.875','0.2/0.8'] for split in ['all','suc','fail']}})
                for split,dist in t.get('prediction_distributions',{}).items():
                    distribution_rows.append({'experiment':name,'condition':condition,'task':task,'split':split,
                                              'valid':dist['n'],**{f'prediction_{k}':v for k,v in dist['counts'].items()}})
        lines+=table(['条件','配对 n','平均连续差','负差率','差≥.5率','离散负','0','1','2','3','4'],pair_table)
        lines+=['','各 task 的准确率和预测分布见统一分析目录的 `task_metrics.csv`、`task_distributions.csv`（cohort）；',
                '`task_metrics_all_populations.csv`、`task_distributions_all_populations.csv` 明确列出 full/cohort/holdout 的固定分母、正确数及无效数；分布表的 binning 列区分 ordinal_reward 与 equal_width_progress。',
                'suc/fail 全部五档分布、连续差每 10% 分档、逐个 suc/fail 配对及解析失败 ID 均保存在同名分析 JSON 中。',
                'Head 完整 ranking、原始 attention mass、逐样本 prompt、key 映射与 hook 证据位于本实验目录。']
        create_text(OUT/name/f'exp_record{args.record_suffix}.md','\n'.join(lines)+'\n')
    title='新增 baseline 全矩阵数值表' if len(experiments)==12 else f'阶段数值表（仅 {len(experiments)}/12 个配置，不是最终总结）'
    overview=['# '+title,'',
              '该文件为逐项数据表；解释和最终结论见 mydata_bench/exp_plan_addbase_summary.md。',
              '所有准确率按固定期望分母计数；NA 表示不可计算。','', '## 全量 baseline','']
    overview+=table(['配置','有效/期望','MAE','全体 MAE 界','严格 all','suc','fail','宽 all','suc','fail'],baseline_rows)
    for title,rows in [('同 cohort target 条件',candidate_rows),('留出视频组 target 条件',holdout_rows)]:
        overview+=['',f'## {title}','']+table(['配置','条件','有效/期望','MAE','严格 all','suc','fail','宽 all','suc','fail','配对 ΔMAE','95% CI','Holm p'],rows)
    overview+=['','## 四条件共同有效样本控制比较','']+table(['配置','范围/k','共同 n / cohort','baseline','target','wrong','low-rank'],control_rows)
    overview+=['','## 同视频配对','']+table(['配置','条件','配对 n','平均连续差','负差率','差≥.5率','离散负','0','1','2','3','4'],pair_rows)
    create_text(folder/'full_tables.md','\n'.join(overview)+'\n')
    all_task_rows,all_distribution_rows=population_task_tables(experiments)
    for filename,rows in [('task_metrics.csv',task_rows),('task_distributions.csv',distribution_rows),
                         ('task_metrics_all_populations.csv',all_task_rows),('task_distributions_all_populations.csv',all_distribution_rows)]:
        f=io.StringIO(newline='')
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        create_text(folder/filename,f.getvalue())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=[f'{scope}:target:{k}' for scope in ['last_frame','all_frames'] for k in [8,32,64]]
    names=list(experiments)
    fig,axes=plt.subplots(1,2,figsize=(15,8),constrained_layout=True)
    for ax,pop in zip(axes,['cohort','holdout']):
        data=np.array([[experiments[n]['conditions'].get(c,{}).get(pop,{}).get('paired_change',{}).get('mae_delta',np.nan)
                        for c in keys] for n in names])
        finite=np.abs(data[np.isfinite(data)]);limit=max(1,float(finite.max())) if len(finite) else 1
        im=ax.imshow(data,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_yticks(range(len(names)),names);ax.set_xticks(range(6),['last k8','last k32','last k64','all k8','all k32','all k64'],rotation=40,ha='right')
        ax.set_title(f'{pop}: paired MAE change (negative = lower error)')
        for i,n in enumerate(names):
            for j,c in enumerate(keys):
                if not np.isfinite(data[i,j]):continue
                change=experiments[n]['conditions'][c][pop]['paired_change']
                suffix='' if change['complete'] else '†'
                ax.text(j,i,f'{data[i,j]:.2f}{suffix}',ha='center',va='center',fontsize=8,
                        color='white' if abs(data[i,j])>limit*.6 else 'black')
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.supxlabel('† Matched valid subset; incomplete coverage. Confidence intervals and fixed-denominator accuracy are in the tables.')
    if not (folder/'paired_mae_changes.png').exists():fig.savefig(folder/'paired_mae_changes.png',dpi=180)
    if not (folder/'paired_mae_changes.svg').exists():fig.savefig(folder/'paired_mae_changes.svg')
    plt.close(fig)
    print('Wrote experiment records, full tables, task CSVs, and plots',flush=True)


if __name__=='__main__':main()
