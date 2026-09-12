"""ASCD-inspired sample-local contrast of native, unrestricted reward classes.

No labels, thresholds, pair identities, or endpoint remapping enter inference.
Discrete models retain all five native reward alternatives. Robometer retains
all ten trained progress bins and its native success logit.
"""
import time
import torch
from .runtime import ResearchRuntime


def combine_native_logits(positive, negative, weight, reference=None):
    """Preserve the existing two-branch arithmetic unless an anchor is explicit."""
    if reference is None:
        return (1 + weight) * positive - weight * negative
    return reference + weight * (positive - negative)


def mean_native_counterfactuals(visual_negative, task_negative):
    """Equal scalar weight for every class; this is an explicit derived mean."""
    if visual_negative.shape!=task_negative.shape:
        raise ValueError('Counterfactual native class shapes differ')
    return .5*(visual_negative+task_negative)


class EvidenceRuntime(ResearchRuntime):
    def predict(self,samples,condition,rankings=None,step=None,previous=None):
        if self.cfg['model'] not in {'meter','qwen','roboreward'}:
            raise ValueError('Native class contrast is not defined for SOLE CoT percentages')
        factorized=self.cfg.get('contrast_negative_mode','visual')=='visual_and_task'
        task_contrast=self.cfg.get('contrast_negative_mode','visual')=='task'
        allowed_factorized=self.cfg['model'] in {'qwen','roboreward','meter'}
        if (factorized or task_contrast) and ((not allowed_factorized if factorized else self.cfg['model'] not in {'qwen','roboreward'}) or
                           self.cfg.get('contrast_reference','positive')!='positive'):
            raise ValueError('Factorized contrast requires a supported native readout and positive anchor')
        if task_contrast and condition!='baseline' and self.cfg['bias']!=0:
            raise ValueError('Task-domain contrast must bypass both visual interventions')
        self.active_ranking_prefix='' if self.cfg['model']=='meter' else 'ANSWER: '
        try:inputs,maps,queries,texts=self.prepare(samples,step,previous)
        finally:self.active_ranking_prefix=''
        started=time.monotonic()
        heads=None;scope=None;region='target'
        if condition!='baseline':
            scope,kind,k=condition.split(':');k=int(k)
            ranked=rankings[scope]['ranking']
            heads=ranked[-k:] if kind=='low_rank' else ranked[:k]
            region='wrong' if kind=='wrong_region' else 'target'
            if region=='wrong' and any(not m['wrong'][scope] for m in maps):
                raise ValueError('Wrong-region control unavailable: insufficient disjoint cells')
        if self.cfg['model']!='meter':
            ids=[self.processor.tokenizer.encode(str(i),add_special_tokens=False) for i in range(1,6)]
            if any(len(x)!=1 for x in ids):raise ValueError('Native reward digits must be single tokens')
            class_ids=[x[0] for x in ids]

        def branch(strength=None,method=None):
            old_method=self.cfg['method']
            state=None
            if strength is not None:
                self.cfg['method']=method or old_method
                state=self.controller.steer(maps,heads,strength,scope,region)
            try:
                with torch.inference_mode():
                    if self.cfg['model']=='meter':
                        hidden=self.model(**inputs,use_cache=False).last_hidden_state
                        h=torch.stack([hidden[i,j] for i,j in enumerate(queries)])
                        progress=self.model.progress_head(h).float()
                        success=self.model.success_head(h).float().squeeze(-1)
                    else:
                        out=self.model(**inputs,use_cache=False,logits_to_keep=1)
                        progress=out.logits[:,-1,class_ids].float()
                        success=None
                return progress,success,state['diagnostics'] if state else {}
            finally:
                self.controller.clear();self.cfg['method']=old_method

        reference=reference_success=None
        task_negative=negative_mean=task_negative_success=negative_success_mean=None
        task_negative_diagnostics={}
        anchored=self.cfg.get('contrast_reference','positive')=='baseline'
        if condition=='baseline':
            positive,positive_success,positive_diagnostics=branch()
            negative=negative_success=None;negative_diagnostics={}
            combined=positive;success_combined=positive_success
        else:
            if anchored:
                reference,reference_success,reference_diagnostics=branch()
                if reference_diagnostics:raise ValueError('Baseline anchor must be unsteered')
            positive,positive_success,positive_diagnostics=branch(self.cfg['bias'])
            alpha=self.cfg['contrast_weight']
            if alpha==0:
                negative=negative_success=None;negative_diagnostics={}
                combined=reference if anchored else positive
                success_combined=reference_success if anchored else positive_success
            else:
                negative,negative_success,negative_diagnostics=(
                    branch(0.,'binding_task_suppression') if task_contrast else
                    branch(-self.cfg['negative_strength'],'mass_transport'))
                if factorized:
                    task_negative,task_negative_success,task_negative_diagnostics=branch(self.cfg['bias'],'binding_task_suppression')
                    negative_mean=mean_native_counterfactuals(negative,task_negative)
                    if positive_success is not None:
                        negative_success_mean=mean_native_counterfactuals(negative_success,task_negative_success)
                combined=combine_native_logits(positive,negative_mean if factorized else negative,alpha,reference)
                success_combined=(combine_native_logits(positive_success,negative_success_mean if factorized else negative_success,alpha,reference_success)
                                  if positive_success is not None else None)
        probabilities=combined.softmax(-1)
        rows=[]
        for i,sample in enumerate(samples):
            row={'example_id':sample['example_id'],'condition':condition,'status':'ok',
                 'readout':'native_bin_attention_contrast' if self.cfg['model']=='meter' else 'five_way_answer_likelihood_attention_contrast',
                 'native_class_logits_positive':positive[i].cpu().tolist(),
                 'native_class_logits_negative':negative[i].cpu().tolist() if negative is not None else None,
                 'native_class_probabilities':probabilities[i].cpu().tolist(),
                 'contrast_weight':self.cfg['contrast_weight'] if condition!='baseline' else 0,
                 'negative_strength':self.cfg['negative_strength'] if negative is not None else None,
                 'prompt':texts[i],'token_audit':self.audit(maps[i]),
                 'attention_diagnostics':positive_diagnostics,'negative_attention_diagnostics':negative_diagnostics,
                 'duration_seconds_per_batch':time.monotonic()-started}
            if factorized:
                row.update(contrast_negative_mode='visual_and_task',
                    native_class_logits_negative_task=task_negative[i].cpu().tolist() if task_negative is not None else None,
                    native_class_logits_negative_mean=negative_mean[i].cpu().tolist() if negative_mean is not None else None,
                    negative_mean_is_derived=negative_mean is not None,
                    task_negative_attention_diagnostics=task_negative_diagnostics,
                    negative_task_strength=self.cfg['negative_task_strength'] if task_negative is not None else None,
                    actual_forward_branches=3 if task_negative is not None else 1,
                    composition='(1+alpha)*positive-alpha*(visual_negative+task_negative)/2' if task_negative is not None else 'single uncontrasted forward')
            if task_contrast:
                row.update(contrast_negative_mode='task',negative_branch_kind='task' if negative is not None else None,
                    negative_strength=self.cfg['negative_task_strength'] if negative is not None else None,
                    negative_strength_domain='task',negative_task_strength=self.cfg['negative_task_strength'] if negative is not None else None,
                    native_class_logits_negative_task=negative[i].cpu().tolist() if negative is not None else None,
                    task_negative_is_actual=True if negative is not None else None,
                    visual_intervention='bypassed',positive_visual_strength=0.,negative_visual_strength=0.,
                    actual_forward_branches=2 if negative is not None else 1,
                    composition='(1+alpha)*task_positive-alpha*task_negative' if negative is not None else 'single uncontrasted forward')
            if anchored:
                row.update(contrast_reference='unsteered_baseline',
                           native_class_logits_reference=reference[i].cpu().tolist() if reference is not None else None,
                           reference_attention_diagnostics={},
                           composition='baseline + weight * (positive - negative)' if condition!='baseline' else 'baseline')
            if self.cfg['model']=='meter':
                bins=torch.linspace(0,1,10,device=probabilities.device)
                row['progress']=float((probabilities[i]*bins).sum())
                row['success_probability']=float(success_combined[i].sigmoid())
                row['success_logit_positive']=float(positive_success[i])
                row['success_logit_negative']=float(negative_success[i]) if negative_success is not None else None
                if factorized:
                    row['success_logit_negative_task']=float(task_negative_success[i]) if task_negative_success is not None else None
                    row['success_logit_negative_mean']=float(negative_success_mean[i]) if negative_success_mean is not None else None
                if anchored:row['success_logit_reference']=float(reference_success[i]) if reference_success is not None else None
            else:
                row['reward']=int(probabilities[i].argmax())+1
                row['progress']=(row['reward']-1)/4
                row['candidate_token_ids']=class_ids
                row['raw_output']=None  # This is a likelihood readout, not fabricated generated text.
            rows.append(row)
        return rows
