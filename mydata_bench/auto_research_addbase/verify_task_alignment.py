"""Real processor/tokenizer tests across all 22 input/model protocols, on CPU."""
import json
from types import SimpleNamespace
import torch
from transformers import AutoProcessor
from mydata_bench.addbase_eval.protocols import SPECIAL_TOKENS, PROG
from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from .prepare import OUT, CONFIGS
from .runtime import ResearchRuntime


def main():
    source=[s for s in json.loads((OLD/'inputs.json').read_text()) if s['cohort']]
    samples=[min(source,key=lambda s:len(s['task'])),max(source,key=lambda s:len(s['task']))]
    results=[]
    for model in ['meter','sole','qwen','roboreward']:
        protocols=['video_text','text_video','image_text','text_image','interleaved']+(['official'] if model in {'meter','sole'} else [])
        config=json.loads((CONFIGS/f'{model}_{protocols[0]}.json').read_text())
        runtime=ResearchRuntime.__new__(ResearchRuntime)
        runtime.processor=AutoProcessor.from_pretrained(config['processor_path'],local_files_only=True)
        runtime.processor.tokenizer.padding_side='left'
        runtime.model=SimpleNamespace(device=torch.device('cpu'),dtype=torch.float32)
        if model=='meter':
            for token in SPECIAL_TOKENS:
                if token not in runtime.processor.tokenizer.get_vocab():
                    runtime.processor.tokenizer.add_special_tokens({'additional_special_tokens':[token]})
            runtime.prog_id=runtime.processor.tokenizer.convert_tokens_to_ids(PROG)
        for protocol in protocols:
            runtime.cfg=json.loads((CONFIGS/f'{model}_{protocol}.json').read_text())
            runtime.cfg['task_binding_fraction']=.5
            official=model=='sole' and protocol=='official'
            inputs,maps,queries,texts=runtime.prepare(samples,7 if official else None,[0,0] if official else None)
            for i,(sample,m) in enumerate(zip(samples,maps)):
                picked=runtime.processor.tokenizer.decode(inputs['input_ids'][i,m['task_positions']].tolist(),skip_special_tokens=False)
                assert sample['task'] in picked
                assert not set(m['task_positions'])&set(m['visual'])
                assert set(m['task_positions'])<=set(m['prompt_text_positions'])
                results.append({'model':model,'protocol':protocol,'example_id':sample['example_id'],
                                'query':queries[i],'task_span':m['task_span_audit'],'decoded_selected_tokens':picked})
            print(model,protocol,'passed',flush=True)
    create_json(OUT/'audit/task_alignment_v1.json',{'status':'passed','protocols':22,'examples':len(results),'results':results})

if __name__=='__main__':main()
