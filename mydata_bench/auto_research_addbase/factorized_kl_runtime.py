"""Three real model forwards, followed by the frozen sample-local KL solver."""
import time
import numpy as np

from .evidence import EvidenceRuntime
from .factorized_kl import factorized_kl


class FactorizedKLRuntime(EvidenceRuntime):
    def predict(self,samples,condition,rankings=None,step=None,previous=None):
        cfg=self.cfg;budget=cfg.get('contrast_kl_budget')
        if (cfg['model'] not in {'qwen','roboreward'} or budget not in [.05,.2,.8]
                or cfg.get('contrast_negative_mode')!='visual_and_task' or cfg.get('contrast_weight')!=2
                or cfg.get('contrast_reference','positive')!='positive' or cfg.get('negative_strength')!=4
                or cfg.get('negative_task_strength')!=4 or cfg.get('task_binding_fraction')!=.5
                or cfg.get('task_binding_distribution')!='uniform' or cfg.get('visual_mass_partition','global')!='global'
                or (condition!='baseline' and cfg['bias']!=4)):
            raise ValueError('The registered three-branch KL configuration differs')
        started=time.monotonic()
        rows=super().predict(samples,condition,rankings,step,previous)
        result=None
        if condition!='baseline':
            vectors=[np.asarray([r[field] for r in rows],dtype=np.float32) for field in
                ['native_class_logits_positive','native_class_logits_negative','native_class_logits_negative_task']]
            result=factorized_kl(*vectors,budget)
        for i,row in enumerate(rows):
            row.update(readout='five_way_answer_likelihood_factorized_kl',contrast_cap=2.,kl_budget=budget,
                fixed_cap_native_class_probabilities=list(row['native_class_probabilities']),
                model_branch_duration_seconds_per_batch=row['duration_seconds_per_batch'])
            if result is not None:
                row.update(contrast_weight=float(result['alpha'][i]),kl_divergence_from_positive=float(result['kl'][i]),
                    native_class_probabilities=result['probabilities'][i].tolist(),
                    native_class_logits_combined=result['logits'][i].tolist(),kl_solver_iterations=64,
                    reward=int(result['probabilities'][i].argmax())+1,
                    composition='KL-limited positive versus actual visual/task negative mean')
                row['progress']=(row['reward']-1)/4
            else:
                row.update(kl_divergence_from_positive=0.,kl_solver_iterations=0,
                    native_class_logits_combined=list(row['native_class_logits_positive']))
            row['duration_seconds_per_batch']=time.monotonic()-started
        return rows
