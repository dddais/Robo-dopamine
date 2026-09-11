from __future__ import annotations
import hashlib
import json
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mydata_bench.meter_eval.model import RobometerModel
from mydata_bench.top_eval.protocol import parse_progress
from .attention import AttentionController
from .protocols import PROG, SPECIAL_TOKENS, prompt_payload, align_input


class Runtime:
    def __init__(self, cfg):
        if cfg['model'] not in {'meter','sole'} or cfg['protocol'] not in {
                'video_text','text_video','image_text','text_image','interleaved','official'}:
            raise ValueError('Unknown model or input protocol')
        expected={'num_frames':8,'query_scope':'all','ranking_score':'raw_mass',
                  'negative_scope':'selected_temporal_frames','decoding':'greedy'}
        for key,value in expected.items():
            if cfg.get(key)!=value:raise ValueError(f'This frozen implementation requires {key}={value!r}')
        self.cfg = cfg
        torch.manual_seed(cfg['seed'])
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(cfg['processor_path'],local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        if cfg['model'] == 'meter':
            for token in SPECIAL_TOKENS:
                if token not in self.processor.tokenizer.get_vocab():
                    self.processor.tokenizer.add_special_tokens({'additional_special_tokens':[token]})
            self.model = RobometerModel.load(cfg['model_path'])
            if len(self.processor.tokenizer) != self.model.config.text_config.vocab_size:
                raise ValueError('Checkpoint/tokenizer vocab mismatch')
            self.prog_id = self.processor.tokenizer.convert_tokens_to_ids(PROG)
            self.layers = self.model.model.language_model.layers
        else:
            self.model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
                cfg['model_path'],dtype=torch.bfloat16,device_map='cuda:0',
                attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
            if loading['missing_keys'] or loading['unexpected_keys'] or loading['mismatched_keys']:
                raise ValueError(f'Non-exact SOLE checkpoint loading: {loading}')
            self.model.eval()
            self.model.loading_audit = loading
            self.layers = self.model.model.language_model.layers
        self.controller = AttentionController(self.layers)

    def prepare(self, samples, step=None, previous=None):
        previous = previous or [0]*len(samples)
        payloads = [prompt_payload(s,self.cfg,step,p) for s,p in zip(samples,previous)]
        texts = [self.processor.apply_chat_template(p['messages'],tokenize=False,
                 add_generation_prompt=self.cfg['model']=='sole', add_vision_id=self.cfg['model']=='meter',
                 enable_thinking=False) for p in payloads]
        args = {'text': texts,'padding':True,'return_tensors':'pt','do_resize':False}
        images = [im for p in payloads for im in p['images']]
        videos = [v for p in payloads for v in p['videos']]
        if images: args['images'] = images
        if videos:
            args.update(videos=videos,video_metadata=[m for p in payloads for m in p['video_metadata']],do_sample_frames=False)
        batch = self.processor(**args)
        maps, queries, grid_offset = [], [], 0
        for i,(s,p) in enumerate(zip(samples,payloads)):
            n = len(p['videos']) if videos else len(p['images'])
            grids = batch['video_grid_thw' if videos else 'image_grid_thw'][grid_offset:grid_offset+n].tolist()
            grid_offset += n
            ids = batch['input_ids'][i].tolist()
            m = align_input(ids,grids,p,s,self.cfg)
            m['prompt_sha256'] = hashlib.sha256(texts[i].encode()).hexdigest()
            m['input_ids_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            m['query'] = (max(j for j,v in enumerate(ids) if v==self.prog_id) if self.cfg['model']=='meter' else len(ids)-1)
            maps.append(m);queries.append(m['query'])
        inputs = {k:v.to(device=self.model.device,dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                  for k,v in batch.items() if torch.is_tensor(v)}
        return inputs,maps,queries,texts

    def collect(self,samples,step=None,previous=None):
        inputs,maps,queries,texts = self.prepare(samples,step,previous)
        state = self.controller.rank(maps,queries)
        try:
            with torch.inference_mode():
                extra = {'logits_to_keep':1} if self.cfg['model']=='sole' else {}
                self.model(**inputs,use_cache=False,**extra)
            if len(state['seen']) != 36: raise RuntimeError('Incomplete head observations')
            rows = []
            for i,s in enumerate(samples):
                rows.append({'example_id':s['example_id'],'status':'ok','raw_mass':{k:v[i].tolist() for k,v in state['raw'].items()},
                             'visual_mass':state['visual'][i].tolist(),'query':queries[i],
                             'query_kind':'final_prog_token' if self.cfg['model']=='meter' else 'last_prompt',
                             'token_audit':self.audit(maps[i]), 'prompt':texts[i]})
            return rows
        finally:
            self.controller.clear()

    @staticmethod
    def audit(m):
        return {k:v for k,v in m.items() if k != 'records'}

    def predict(self,samples,condition,rankings=None,step=None,previous=None):
        inputs,maps,queries,texts = self.prepare(samples,step,previous)
        state = None
        if condition != 'baseline':
            scope,kind,k = condition.split(':');k=int(k)
            if kind=='wrong_region' and any(not m['wrong'][scope] for m in maps):
                raise ValueError('Wrong-region control unavailable: insufficient disjoint cells; baseline and target remain valid')
            ranked = rankings[scope]['ranking']
            heads = ranked[-k:] if kind=='low_rank' else ranked[:k]
            state = self.controller.steer(maps,heads,self.cfg['bias'],scope,'wrong' if kind=='wrong_region' else 'target')
        started = time.monotonic()
        try:
            with torch.inference_mode():
                if self.cfg['model']=='meter':
                    output = self.model(**inputs,use_cache=False)
                    values,success,positions = self.model.read_progress(output.last_hidden_state,inputs['input_ids'],self.prog_id)
                    rows = [{'status':'ok','progress':p[-1],'per_frame_progress':p,'success_probability':s[-1],
                             'per_frame_success':s,'progress_token_positions':pos} for p,s,pos in zip(values,success,positions)]
                else:
                    output = self.model.generate(**inputs,do_sample=False,max_new_tokens=self.cfg['max_new_tokens'],
                                  use_cache=True,logits_to_keep=1,pad_token_id=self.processor.tokenizer.pad_token_id)
                    generated = output[:,inputs['input_ids'].shape[1]:]
                    raw = self.processor.batch_decode(generated,skip_special_tokens=True)
                    rows = [{'raw_output':r, 'generated_tokens':int((tok!=self.processor.tokenizer.pad_token_id).sum()),
                             **parse_progress(r)} for r,tok in zip(raw,generated)]
            for i,row in enumerate(rows):
                row.update(example_id=samples[i]['example_id'],condition=condition,step=step,
                           previous_percentage=(previous[i] if previous else None),
                           duration_seconds_per_batch=time.monotonic()-started,
                           token_audit=self.audit(maps[i]),prompt=texts[i],
                           attention_diagnostics=(state['diagnostics'] if state else {}))
            return rows
        finally:
            self.controller.clear()
