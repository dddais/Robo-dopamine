"""Spatial context profiles on accepted visual-token maps, without image edits."""
import numpy as np


def context_profile(mapping, scope, region, radius):
    if not 0 < radius <= 1:
        raise ValueError('Context radius must be in (0,1] normalized image coordinates')
    if region not in {'target','wrong'}:
        raise ValueError('Unknown spatial region')
    domain=set(mapping['target'][scope]) | set(mapping['negative'][scope])
    core=set(mapping['target'][scope])
    selected=set(mapping[region][scope])
    if not selected or not selected <= domain:
        raise ValueError('Unavailable or invalid context-control core')
    positions=[];weights=[];spans=[]
    for span in mapping['alignment'][scope]:
        start,end=span['start'],span['end']
        height,width=span['grid_thw'][1]//2,span['grid_thw'][2]//2
        if end-start!=height*width:
            raise ValueError('Context profile expects one merged temporal span per grid')
        ids=sorted(p for p in domain if start<=p<end)
        target_ids=sorted(p for p in core if start<=p<end)
        region_ids=sorted(p for p in selected if start<=p<end)
        if not ids:
            continue
        if not target_ids or len(region_ids)!=len(target_ids):
            raise ValueError('Each selected temporal span needs an equal-sized context core')
        def coords(indices):
            index=np.asarray(indices)-start
            return np.stack(((index//width+.5)/height,(index%width+.5)/width),axis=-1)
        locations=coords(ids)
        def squared_distance(core_ids):
            return ((locations[:,None]-coords(core_ids)[None])**2).sum(-1).min(-1)
        distance=squared_distance(target_ids)
        values=np.exp(-distance/(2*radius**2))
        if region=='wrong':
            # Preserve the complete gain multiset, including any edge truncation.
            # Only its spatial arrangement changes; this is a matched control,
            # not an assertion of an untruncated Gaussian at the new location.
            order=np.lexsort((np.asarray(ids),squared_distance(region_ids)))
            shuffled=np.empty_like(values)
            shuffled[order]=np.sort(values)[::-1]
            values=shuffled
        selected_values=[float(values[i]) for i,p in enumerate(ids) if p in selected]
        if not all(value==1. for value in selected_values):
            raise ValueError('Context core must retain exactly unit gain')
        positions.extend(ids);weights.extend(values.tolist())
        spans.append({'start':start,'end':end,'core_count':len(region_ids),
                      'domain_count':len(ids),'kernel_sum':float(values.sum()),
                      'noncore_weight_sum':float(values.sum()-len(region_ids))})
    if set(positions)!=domain or len(positions)!=len(set(positions)):
        raise ValueError('Spatial profile failed to cover exactly the accepted domain')
    return positions,weights,{
        'radius_fraction':radius,'coordinate_units':'x/grid_width and y/grid_height',
        'region':region,'control':'radial_rank_permutation_preserving_gain_multiset' if region=='wrong' else None,
        'kernel':'exp(-nearest_core_squared_distance/(2*radius**2))',
        'spans':spans,'domain_count':len(positions),'core_count':len(selected),
        'kernel_min':min(weights),'kernel_max':max(weights)}
