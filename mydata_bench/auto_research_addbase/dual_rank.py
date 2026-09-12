"""Budget-matched union ranking of visual and instruction attention heads."""
import numpy as np
from mydata_bench.addbase_eval.run import rank, latest
from mydata_bench.addbase_eval.prepare import create_json


def alternating_union(visual, instruction):
    """Alternate sources, skipping duplicates; every prefix is a nested head set."""
    visual_ids={(row['layer'],row['head']) for row in visual}
    instruction_ids={(row['layer'],row['head']) for row in instruction}
    if len(visual_ids)!=len(visual) or len(instruction_ids)!=len(instruction) or visual_ids!=instruction_ids:
        raise ValueError('Dual lists must contain the same unique head identities')
    positions=[0,0]
    sources=[visual,instruction]
    seen=set()
    result=[]
    while len(result)<len(visual):
        for source,items in enumerate(sources):
            while positions[source]<len(items):
                item=items[positions[source]]
                positions[source]+=1
                key=(item['layer'],item['head'])
                if key in seen:continue
                seen.add(key)
                result.append(dict(item,selection_source=['visual','instruction'][source]))
                break
    return result


def dual_rank(runtime,samples,baseline,output):
    originals=rank(runtime,samples,baseline,output/'raw_visual')
    observations=list(latest(output/'raw_visual/ranking_observations.jsonl').values())
    if any('task_mass' not in row for row in observations):
        raise ValueError('Dual ranking requires actual task attention observations')
    task=np.mean([row['task_mass'] for row in observations],axis=0)
    result={}
    for scope,original in originals.items():
        visual=[dict(row,visual_score=row['score'],task_score=float(task[row['layer'],row['head']]))
                for row in original['ranking']]
        instruction=sorted(visual,key=lambda row:(-row['task_score'],row['layer'],row['head']))
        artifact=dict(original,ranking=alternating_union(visual,instruction),
                      ranking_score='alternating_raw_visual_and_task_mass',
                      top_k_budget='k total unique heads, not k per source',
                      task_selection='literal instruction tokens; no labels or model reward scores')
        create_json(output/f'ranking_{scope}.json',artifact)
        result[scope]=artifact
    return result
