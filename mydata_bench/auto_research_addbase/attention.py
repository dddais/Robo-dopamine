"""Mass-conserving visual attention transport, with the old bias as a control."""
import torch
from mydata_bench.addbase_eval.attention import AttentionController
from .context import context_profile
from .temporal import temporal_domains, partitioned_redistribute


def redistribute(weights, domain, target, strength):
    """Nonnegative, normalized transport; zero visible domain is a no-op.

    All inputs broadcast on the key axis. Domain-external weights are identical.
    Subtracting strength before exponentiating prevents overflow for strong gains.
    """
    part = weights * domain
    mass = part.sum(-1, keepdim=True)
    tilted = part * torch.exp(strength * (target.to(weights.dtype) - 1))
    denominator = tilted.sum(-1, keepdim=True)
    redistributed = tilted * (mass / denominator.clamp_min(torch.finfo(weights.dtype).tiny))
    return torch.where(domain, redistributed, weights)


def mixture_redistribute(weights, domain, target, fraction):
    """Move a bounded fraction of domain mass to its visible target evidence.

    Each non-target key retains exactly (1-fraction) of its old weight. Domain
    mass and every domain-external key are unchanged. If the target is causally
    invisible there is no intervention, avoiding future-key leakage.
    """
    if not 0 <= fraction <= 1: raise ValueError('Mixture fraction must be in [0,1]')
    part=weights*domain
    selected=part*target
    selected_mass=selected.sum(-1,keepdim=True)
    destination=selected*(part.sum(-1,keepdim=True)/selected_mass.clamp_min(torch.finfo(weights.dtype).tiny))
    change=fraction*(destination-part)
    return weights+torch.where(selected_mass>0,change,torch.zeros_like(change))


def uniform_redistribute(weights, domain, target, visible, fraction):
    """Inject domain mass uniformly into explicitly visible instruction tokens.

    Visibility comes from the causal/padding mask, not rounded probabilities:
    an allowed key that underflowed to zero can receive evidence; a masked key
    cannot. An empty visible target leaves the entire row unchanged.
    """
    if not 0 <= fraction <= 1:
        raise ValueError('Mixture fraction must be in [0,1]')
    selected = domain & target & visible
    count = selected.sum(-1, keepdim=True)
    part = weights * domain
    destination = selected.to(weights.dtype) * (part.sum(-1, keepdim=True) / count.clamp_min(1))
    change = fraction * (destination - part)
    return weights + torch.where(count > 0, change, torch.zeros_like(change))


class ResearchController(AttentionController):
    def __init__(self, layers, cfg):
        self.cfg = cfg
        super().__init__(layers)

    def rank(self, maps, queries, num_layers=36, num_heads=32):
        state=super().rank(maps,queries,num_layers,num_heads)
        if self.cfg.get('ranking_strategy')=='dual_mass':
            import numpy as np
            state['task_mass']=np.zeros((len(maps),num_layers,num_heads))
        return state

    def collect_task_mass(self,query,key,attention_mask,scaling,layer):
        state=self.state
        b,h,q,d=query.shape
        k=key.shape[-2]
        repeated=key.repeat_interleave(h//key.shape[1],dim=1)
        selected=torch.stack([query[i,:,j,:] for i,j in enumerate(state['queries'])])
        scores=torch.einsum('bhd,bhkd->bhk',selected.float(),repeated.float())*(scaling or d**-0.5)
        for i,j in enumerate(state['queries']):
            if attention_mask is None:scores[i,:,j+1:]=-torch.inf
            else:
                mask=attention_mask[i if attention_mask.shape[0]>1 else 0,:,j,:k]
                if mask.dtype==torch.bool:scores[i].masked_fill_(~mask,-torch.inf)
                else:scores[i]+=mask.float()
        weights=torch.nan_to_num(scores.softmax(-1),nan=0.)
        for i,mapping in enumerate(state['maps']):
            state['task_mass'][i,layer]=weights[i,:,mapping['task_positions']].sum(-1).cpu().numpy()

    def forward(self, module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        state = self.state
        layer = self.layer_ids.get(id(module))
        method = self.cfg.get('method', 'bias')
        if state is not None and state['kind']=='rank' and 'task_mass' in state and layer is not None:
            self.collect_task_mass(query,key,attention_mask,scaling,layer)
        if (method == 'bias' or state is None or state['kind'] != 'steer'
                or layer not in state['heads']):
            return super().forward(module, query, key, value, attention_mask,
                                   dropout=dropout, scaling=scaling, **kwargs)
        if method not in {'mass_transport','mixture_transport','positive_bias','binding_transport','context_transport','binding_task_suppression'}: raise ValueError(method)
        if dropout: raise ValueError('Research controller requires deterministic inference')
        result, weights_out = self.original(module, query, key, value, attention_mask,
                                            dropout=dropout, scaling=scaling, **kwargs)
        strength = float(state['bias'])
        pure_task = self.cfg.get('contrast_negative_mode')=='task'
        if pure_task and (strength != 0 or method not in {'binding_transport','binding_task_suppression'}):
            raise ValueError('Task-domain contrast requires zero visual strength and a text-domain branch')
        if strength == 0 and method not in {'binding_transport','binding_task_suppression'}: return result, weights_out
        b, h, q, d = query.shape
        k = key.shape[-2]
        heads = state['heads'][layer]
        index_key=('kv_heads',layer,str(query.device))
        if index_key not in state['cache']:
            state['cache'][index_key]=torch.tensor([n // (h // key.shape[1]) for n in heads],device=query.device)
        kv_heads=state['cache'][index_key]
        kh = key.index_select(1, kv_heads).float()
        vh = value.index_select(1, kv_heads).float()
        base_key=('domains_base',str(query.device))
        if base_key not in state['cache']:
            end=max(max(m['visual']) for m in state['maps'])+1
            domain=torch.zeros((b,1,1,end),dtype=torch.bool,device=query.device)
            target=torch.zeros_like(domain,dtype=torch.float32 if method=='context_transport' else torch.bool)
            for i,mapping in enumerate(state['maps']):
                ids=sorted(set(mapping['target'][state['scope']]) | set(mapping['negative'][state['scope']]))
                domain[i,0,0,ids]=True
                if method=='context_transport':
                    positions,values,audit=context_profile(mapping,state['scope'],state['region'],self.cfg['context_radius'])
                    target[i,0,0,positions]=torch.tensor(values,dtype=target.dtype,device=query.device)
                    mapping.setdefault('context_kernel_audit',{})[state['scope']]=audit
                else:target[i,0,0,mapping[state['region']][state['scope']]]=True
            state['cache'][base_key]=(domain,target)
        mask_key=('domains',k,str(query.device))
        if mask_key not in state['cache']:
            domain,target=state['cache'][base_key]
            if k>domain.shape[-1]:
                padding=(0,k-domain.shape[-1])
                domain=torch.nn.functional.pad(domain,padding)
                target=torch.nn.functional.pad(target,padding)
            state['cache'][mask_key]=(domain[...,:k],target[...,:k])
        domain,target=state['cache'][mask_key]
        temporal_partition = self.cfg.get('visual_mass_partition', 'global') == 'temporal_planes'
        if temporal_partition:
            if method not in {'mass_transport', 'binding_transport'}:
                raise ValueError('Temporal partition is defined for the frozen two-branch uniform operator')
            plane_key = ('temporal_planes', k, str(query.device))
            if plane_key not in state['cache']:
                groups = [temporal_domains(mapping, state['scope']) for mapping in state['maps']]
                planes = [torch.zeros_like(domain) for _ in range(max(map(len, groups)))]
                for i, sample_groups in enumerate(groups):
                    for j, positions in enumerate(sample_groups):
                        planes[j][i, 0, 0, [p for p in positions if p < k]] = True
                state['cache'][plane_key] = planes
            plane_masks = state['cache'][plane_key]
        if method in {'binding_transport','binding_task_suppression'}:
            text_key=('text_domains',k,str(query.device))
            if text_key not in state['cache']:
                text_domain=torch.zeros((b,1,1,k),dtype=torch.bool,device=query.device)
                task_target=torch.zeros_like(text_domain)
                for i,mapping in enumerate(state['maps']):
                    text_domain[i,0,0,mapping['prompt_text_positions']]=True
                    task_target[i,0,0,mapping['task_positions']]=True
                if (text_domain&domain).any():raise ValueError('Text and visual binding domains overlap')
                state['cache'][text_key]=(text_domain,task_target)
            text_domain,task_target=state['cache'][text_key]
        # Original SDPA output is retained, including on unselected heads. Only
        # the analytic change caused by redistribution is added to selected heads.
        result = result.clone()
        query_scope=self.cfg.get('research_query_scope','all')
        first_query=(max(0,min(m['query'] for m in state['maps'])-(k-q))
                     if query_scope=='readout' else 0)
        for start in range(first_query, q, 128):
            stop = min(start+128,q)
            scores = torch.matmul(query[:,heads,start:stop].float(),kh.transpose(-1,-2)) * (scaling or d**-0.5)
            uniform_task = method=='binding_transport' and self.cfg.get('task_binding_distribution','attention')=='uniform'
            visible = torch.ones_like(scores,dtype=torch.bool) if uniform_task else None
            if attention_mask is None:
                if q > 1:
                    qi = torch.arange(k-q+start,k-q+stop,device=query.device)[:,None]
                    ki = torch.arange(k,device=query.device)[None,:]
                    scores.masked_fill_(ki > qi, -torch.inf)
                    if uniform_task:visible = visible & (ki <= qi)
            else:
                mask = attention_mask[...,:k]
                if mask.shape[1] > 1: mask = mask[:,heads]
                if mask.shape[-2] > 1: mask = mask[...,start:stop,:]
                if mask.dtype == torch.bool: scores.masked_fill_(~mask,-torch.inf)
                else: scores += mask.float()
                if uniform_task:
                    allowed = mask if mask.dtype==torch.bool else mask > torch.finfo(mask.dtype).min
                    visible = visible & allowed
            p = torch.nan_to_num(scores.softmax(-1), nan=0.0)
            if method in {'mass_transport','binding_transport','context_transport','binding_task_suppression'}:
                if pure_task:
                    shifted=p
                else:
                    shifted=(partitioned_redistribute(p,plane_masks,target,strength,redistribute)
                             if temporal_partition else redistribute(p,domain,target,strength))
                if method=='binding_task_suppression':
                    suppressed=redistribute(shifted,text_domain,task_target,-self.cfg['negative_task_strength'])
                    if pure_task:
                        task_mass=(shifted*text_domain*task_target).sum(-1,keepdim=True)
                        text_mass=(shifted*text_domain).sum(-1,keepdim=True)
                        shifted=torch.where((task_mass>0)&(task_mass<text_mass),suppressed,shifted)
                    else:
                        shifted=suppressed
                if method=='binding_transport':
                    if uniform_task:
                        shifted=uniform_redistribute(shifted,text_domain,task_target,visible,self.cfg['task_binding_fraction'])
                    else:
                        shifted=mixture_redistribute(shifted,text_domain,task_target,self.cfg['task_binding_fraction'])
            elif method=='mixture_transport':shifted=mixture_redistribute(p,domain,target,strength)
            else:shifted=torch.nan_to_num((scores+strength*target).softmax(-1),nan=0.0)
            delta = torch.matmul(shifted-p,vh).transpose(1,2).to(result.dtype)
            if query_scope!='all':
                positions=torch.arange(k-q+start,k-q+stop,device=query.device)
                allowed=torch.ones((b,stop-start),dtype=torch.bool,device=query.device)
                for i,mapping in enumerate(state['maps']):
                    if query_scope=='readout':allowed[i]=positions>=mapping['query']
                    elif query_scope=='text':
                        visual=torch.tensor(mapping['visual'],device=query.device)
                        allowed[i]=~torch.isin(positions,visual)
                    else:raise ValueError(query_scope)
                delta*=allowed[:,:,None,None]
            result[:,start:stop,heads,:] += delta
        diag = state['diagnostics'].setdefault(str(layer), {'method':method,'heads':heads,
                'strength':strength,'all_query_rows':self.cfg.get('research_query_scope','all')=='all',
                'research_query_scope':self.cfg.get('research_query_scope','all'),'causal_mask_preserved':True,
                'domain_mass_preserved':method!='positive_bias','prefill_calls':0,'decode_calls':0})
        diag['prefill_calls' if q>1 else 'decode_calls'] += 1
        if pure_task:
            diag.update(visual_intervention='bypassed',visual_weights_unchanged_locally=True)
        if temporal_partition:
            diag.update(visual_mass_partition='temporal_planes',
                plane_counts=[len(temporal_domains(mapping,state['scope'])) for mapping in state['maps']],
                plane_definition='accepted merged token time plane; not necessarily one physical frame',
                per_plane_mass_preserved_by_operator=True)
        if method in {'binding_transport','binding_task_suppression'}:
            diag.update(text_domain_mass_preserved=True,task_token_counts=[len(m['task_positions']) for m in state['maps']])
        if method=='binding_task_suppression':
            diag.update(task_logit_strength=-self.cfg['negative_task_strength'],task_binding_distribution='exponential_suppression')
        if method=='binding_transport':
            diag.update(task_binding_fraction=self.cfg['task_binding_fraction'])
            if self.cfg.get('task_binding_distribution')=='uniform':
                diag.update(task_binding_distribution='uniform',explicit_mask_visibility=True)
        if method=='context_transport':
            diag.update(context_radius=self.cfg['context_radius'],context_region=state['region'],
                        continuous_spatial_gain=True)
        return result, weights_out
