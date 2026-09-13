"""Four-model adapter; existing protocols and checkpoints remain untouched."""
import hashlib
import json
import time
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from mydata_bench.addbase_eval.runtime import Runtime
from mydata_bench.addbase_eval.protocols import prompt_payload, align_input
from mydata_bench.roboreward_eval.runner import ROBOREWARD_PROMPT, parse_native_score
from mydata_bench.qwen_eval.protocols import INTERLEAVED_REWARD_PROMPT
from mydata_bench.top_eval.protocol import parse_progress
from mydata_bench.top_eval.versioning import protocol_metadata, validate_protocol_config
from .attention import ResearchController


class ResearchRuntime(Runtime):
    def __init__(self, cfg):
        from .gpu_policy import require_allowed_visibility
        require_allowed_visibility()
        model = cfg['model']
        surrogate = dict(cfg, model='sole' if model in {'qwen','roboreward'} else model)
        super().__init__(surrogate)
        ALL_ATTENTION_FUNCTIONS.register('sdpa', self.controller.original)
        self.cfg = cfg
        self.controller = ResearchController(self.layers, self.cfg)

    def payload(self, sample, step, previous):
        if self.cfg['model'] in {'meter','sole'}:
            return prompt_payload(sample,self.cfg,step,previous)
        p = prompt_payload(sample,dict(self.cfg,model='sole'),step,previous)
        proto = self.cfg['protocol']
        prompt = {'type':'text','text':ROBOREWARD_PROMPT.format(task=sample['task'])}
        if proto == 'interleaved':
            parts = INTERLEAVED_REWARD_PROMPT.format(task=sample['task']).split('<image>')
            content=[]
            for part in parts[:-1]: content.extend([{'type':'text','text':part},{'type':'image'}])
            content.append({'type':'text','text':parts[-1]})
        else:
            visual = [{'type':'video'}] if p['videos'] else [{'type':'image'} for _ in p['images']]
            content = visual+[prompt] if proto in {'video_text','image_text'} else [prompt]+visual
        p['messages']=[{'role':'user','content':content}]
        return p

    def prepare(self,samples,step=None,previous=None):
        # Workers may reuse this runtime while switching protocols.
        validate_protocol_config(self.cfg)
        if not samples:
            raise ValueError('Cannot prepare an empty sample batch')
        if previous is None:
            previous = [0]*len(samples)
        if len(previous) != len(samples):
            raise ValueError('Previous predictions must match the sample batch length')
        for sample in samples:
            if len(sample['image_paths']) != 8 or len(sample['sampling']['selected_source_indices']) != 8:
                raise ValueError('Frozen evaluation requires exactly eight sampled frames per example')
        payloads=[self.payload(s,step,p) for s,p in zip(samples,previous)]
        texts=[self.processor.apply_chat_template(p['messages'],tokenize=False,
               add_generation_prompt=self.cfg['model']!='meter',add_vision_id=self.cfg['model']=='meter',
               enable_thinking=False) for p in payloads]
        if getattr(self,'active_ranking_prefix',''):
            texts=[text+self.active_ranking_prefix for text in texts]
        args={'text':texts,'padding':True,'return_tensors':'pt','do_resize':False}
        args.update(payloads[0].get('processor_kwargs', {}))
        images=[im for p in payloads for im in p['images']]
        videos=[v for p in payloads for v in p['videos']]
        if images:args['images']=images
        if videos:args.update(videos=videos,video_metadata=[m for p in payloads for m in p['video_metadata']],do_sample_frames=False)
        batch=self.processor(**args)
        maps=[];queries=[];offset=0
        for i,(s,p) in enumerate(zip(samples,payloads)):
            n=len(p['videos']) if videos else len(p['images'])
            grids=batch['video_grid_thw' if videos else 'image_grid_thw'][offset:offset+n].tolist();offset+=n
            ids=batch['input_ids'][i].tolist()
            m=align_input(ids,grids,p,s,self.cfg)
            m['query']=max(j for j,v in enumerate(ids) if v==self.prog_id) if self.cfg['model']=='meter' else len(ids)-1
            m['prompt_sha256']=hashlib.sha256(texts[i].encode()).hexdigest()
            m['input_ids_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if self.cfg.get('task_binding_fraction') is not None or self.cfg.get('require_task_positions', False):
                from .binding import instruction_positions
                m['task_positions'],m['prompt_text_positions'],m['task_span_audit']=instruction_positions(
                    self.processor.tokenizer,ids,s['task'],batch['attention_mask'][i].tolist(),m['visual'])
            maps.append(m);queries.append(m['query'])
        inputs={k:v.to(device=self.model.device,dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                for k,v in batch.items() if torch.is_tensor(v)}
        return inputs,maps,queries,texts

    def collect(self,samples,step=None,previous=None):
        self.active_ranking_prefix=self.cfg.get('ranking_prefix','')
        try:inputs,maps,queries,texts=self.prepare(samples,step,previous)
        finally:self.active_ranking_prefix=''
        state=self.controller.rank(maps,queries)
        try:
            with torch.inference_mode():
                extra={} if self.cfg['model']=='meter' else {'logits_to_keep':1}
                self.model(**inputs,use_cache=False,**extra)
            if len(state['seen']) != 36:
                raise RuntimeError('Incomplete head observations')
            return [{'example_id':s['example_id'],'status':'ok',
                     **protocol_metadata(self.cfg),
                     **({'task_mass':state['task_mass'][i].tolist()} if 'task_mass' in state else {}),
                     'raw_mass':{k:v[i].tolist() for k,v in state['raw'].items()},
                     'visual_mass':state['visual'][i].tolist(),'query':queries[i],
                     'query_kind':('final_prog_token' if self.cfg['model']=='meter' else
                                   'answer_format_prefix' if self.cfg.get('ranking_prefix') else 'last_prompt'),
                     'token_audit':self.audit(maps[i]),'prompt':texts[i]} for i,s in enumerate(samples)]
        finally:self.controller.clear()

    def predict(self,samples,condition,rankings=None,step=None,previous=None):
        if self.cfg['model'] in {'meter','sole'}:
            return super().predict(samples,condition,rankings,step,previous)
        inputs,maps,queries,texts=self.prepare(samples,step,previous)
        state=None
        if condition!='baseline':
            scope,kind,k=condition.split(':');k=int(k)
            ranked=rankings[scope]['ranking']
            heads=ranked[-k:] if kind=='low_rank' else ranked[:k]
            if kind=='wrong_region' and any(not m['wrong'][scope] for m in maps):
                raise ValueError('Wrong-region control unavailable: insufficient disjoint cells')
            state=self.controller.steer(maps,heads,self.cfg['bias'],scope,'wrong' if kind=='wrong_region' else 'target')
        start=time.monotonic()
        try:
            with torch.inference_mode():
                output=self.model.generate(**inputs,do_sample=False,max_new_tokens=self.cfg['max_new_tokens'],
                       use_cache=True,logits_to_keep=1,pad_token_id=self.processor.tokenizer.pad_token_id)
            generated=output[:,inputs['input_ids'].shape[1]:]
            raw=self.processor.batch_decode(generated,skip_special_tokens=True)
            rows=[]
            for i,text in enumerate(raw):
                try:
                    score=parse_native_score(text)
                    parsed={'status':'ok','progress':(score-1)/4,'reward':score}
                except ValueError as exc:parsed={'status':'parse_error','progress':None,'parse_error':str(exc)}
                rows.append(dict(parsed,raw_output=text,example_id=samples[i]['example_id'],condition=condition,
                    generated_tokens=int((generated[i]!=self.processor.tokenizer.pad_token_id).sum()),
                    duration_seconds_per_batch=time.monotonic()-start,token_audit=self.audit(maps[i]),
                    prompt=texts[i],attention_diagnostics=state['diagnostics'] if state else {}))
            return rows
        finally:self.controller.clear()
