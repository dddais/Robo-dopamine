"""Top-head identities and exact set overlaps, including historical models."""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
from .prepare import ROOT,OUT,create_json


def read_ranking(path):
    d=json.loads(path.read_text())
    rows=d.get('ranking') or d.get('rankings',{}).get('mean')
    if not rows:return None
    skip=int(d.get('skip_early_layers',8))
    rows=[r for r in rows if int(r['layer'])>=skip]
    pairs=[(int(r['layer']),int(r['head'])) for r in rows]
    if len(set(pairs))!=len(pairs):raise ValueError(f'Duplicate heads: {path}')
    return {'source':str(path),'skip_early_layers':skip,'pairs':pairs,'num_layers':d.get('num_layers'),
            'num_heads':d.get('num_heads'),'n':d.get('n',d.get('n_discovery_samples')),
            'metric':d.get('ranking_score',d.get('ranking_score_kind',
                        'mean_excess_mass_then_raw_mass' if rows and 'mean_excess_mass' in rows[0] else 'source_order'))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='head_overlap_v1');args=parser.parse_args()
    output=OUT/args.output_name;output.mkdir(parents=True,exist_ok=True)
    rankings={}
    for path in sorted(OUT.glob('*/ranking_*frame*.json')):
        if '_pilot' in path.parent.name:continue
        rankings[f'{path.parent.name}/{path.stem.removeprefix("ranking_")}']=read_ranking(path)
    for folder in ['experiments_v2_corssmodel','experiments_v2']:
        for path in sorted((ROOT/'results/mydata_bench'/folder).glob('attention_*/consensus_ranking.json')):
            if any(m in path.parent.name for m in ['roboreward','qwen']):
                rankings[f'historical/{path.parent.name}']=read_ranking(path)
        for path in sorted((ROOT/'results/mydata_bench'/folder).glob('attention_*grm*/in_domain_ranking.json')):
            rankings[f'historical/{path.parent.name}']=read_ranking(path)
            data=json.loads(path.read_text())
            raw=sorted(data['ranking'],key=lambda r:(-r['mean_raw_mass'],r['layer'],r['head']))
            companion=dict(rankings[f'historical/{path.parent.name}'])
            companion['pairs']=[(int(r['layer']),int(r['head'])) for r in raw if int(r['layer'])>=8]
            companion['metric']='mean_raw_mass_reranked_read_only_from_historical_values'
            rankings[f'historical/{path.parent.name}/reranked_raw']=companion
    rankings={k:v for k,v in rankings.items() if v and len(v['pairs'])>=64}
    create_json(output/'rankings.json',rankings)
    lines=['# Ranking head 编号与重合度','',
           '以 (layer, head) 零基编号计算精确集合交集；跨不同训练权重/隐藏维度的编号重合不表示功能等价。',
           '历史 ranking 的发现集、输入协议、时域或 query 位置可能不同，因此仅作描述性比较。','',
           'GRM 原发布顺序按 mean_excess_mass 优先排序；另提供 /reranked_raw 行，以历史 mean_raw_mass 重新排序，不改写历史结果。','',
           '| 配置 | 具体 top 8 |','| --- | --- |']
    for name,r in rankings.items():
        lines.append('| '+name+' | '+', '.join(f'L{l}H{h}' for l,h in r['pairs'][:8])+' |')
    rows=[]
    for (a,ra),(b,rb) in combinations(rankings.items(),2):
        if a.startswith('historical/') and b.startswith('historical/'):continue
        for k in [8,32,64]:
            sa=set(ra['pairs'][:k]);sb=set(rb['pairs'][:k]);intersection=sa&sb
            rows.append({'first':a,'second':b,'k':k,'overlap_count':len(intersection),
                         'overlap_fraction':len(intersection)/k,'jaccard':len(intersection)/len(sa|sb)})
    with (output/'overlap.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    create_json(output/'overlap.json',rows)
    lines += ['', '## 新增两模型的同协议、同时域比较','',
              '| 协议 / 时域 | top 8 | top 32 | top 64 |','| --- | ---: | ---: | ---: |']
    for protocol in ['video_text','text_video','image_text','text_image','interleaved','official']:
        for scope in ['last_frame','all_frames']:
            a=f'meter_{protocol}/{scope}';b=f'sole_{protocol}/{scope}'
            if a not in rankings or b not in rankings:continue
            ra=rankings[a]['pairs'];rb=rankings[b]['pairs']
            values=[f'{len(set(ra[:k])&set(rb[:k]))}/{k} ({len(set(ra[:k])&set(rb[:k]))/k:.1%})' for k in [8,32,64]]
            lines.append('| '+protocol+' / '+scope+' | '+' | '.join(values)+' |')
    with (output/'ranking_overlap.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('Rankings',len(rankings),'overlap rows',len(rows))


if __name__=='__main__':main()
