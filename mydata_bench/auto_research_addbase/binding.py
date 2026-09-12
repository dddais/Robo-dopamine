"""Locate the actual instruction tokens without inserting any new prompt text."""


def instruction_positions(tokenizer, ids, task, valid, visual):
    decoded=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=tokenizer(decoded,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=ids:
        raise ValueError('Instruction alignment requires exact tokenizer round-trip')
    if decoded.count(task)!=1:
        raise ValueError('Expected exactly one literal task instruction in model input')
    start=decoded.index(task);end=start+len(task)
    positions=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<end and b>start and b>a]
    visual=set(visual)
    domain=[i for i,flag in enumerate(valid) if flag and i not in visual]
    if not positions or not set(positions)<=set(domain):
        raise ValueError('Task instruction does not align to valid text tokens')
    return positions,domain,{'roundtrip_exact':True,'literal_task_matches':1,
                           'task':task,'character_span':[start,end],'token_count':len(positions)}
