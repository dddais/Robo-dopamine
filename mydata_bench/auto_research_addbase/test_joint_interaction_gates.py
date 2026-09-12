import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .joint_interaction import CELLS, VARIANT, complete, verify_rows


class InteractionGateTests(unittest.TestCase):
    def row(self):
        cells = {k:(np.arange(5,dtype=np.float32)+i).tolist() for i,k in enumerate(CELLS)}
        arrays = {k:np.asarray(v,dtype=np.float32) for k,v in cells.items()}
        effect = arrays['pp']-arrays['mp']-arrays['pm']+arrays['mm']; z = arrays['pp']+effect
        p = np.exp(z-z.max()); p /= p.sum()
        diagnostics = {}
        for cell,(strength,method) in CELLS.items():
            diagnostics[cell] = {'0':dict(heads=[0],method=method,strength=strength,
                domain_mass_preserved=True,text_domain_mass_preserved=True,causal_mask_preserved=True,
                all_query_rows=True,prefill_calls=1,task_binding_fraction=.5,
                task_binding_distribution='uniform' if method == 'binding_transport' else 'exponential_suppression',
                task_logit_strength=-4)}
        return dict(status='ok',example_id='x',condition='all_frames:target:1',
            contrast_negative_mode='joint_interaction',readout='five_way_answer_likelihood_joint_interaction',
            candidate_token_ids=[1,2,3,4,5],native_class_logits_positive=cells['pp'],
            native_cell_logits=cells,native_interaction_logits=effect.tolist(),
            native_class_logits_combined=z.tolist(),native_class_probabilities=p.tolist(),
            reward=5,progress=1.,actual_forward_branches=4,interaction_weight=1,
            composition='pp+(pp-mp-pm+mm)',cell_attention_diagnostics=diagnostics)

    def test_complete_native_formula_accepted(self):
        row = self.row()
        self.assertEqual(verify_rows({'x':row},['x'],{(0,0)}),{'x':row})

    def test_missing_cells_tampered_predictions_and_bad_heads_rejected(self):
        mutations = [lambda r:r.update(actual_forward_branches=3),
            lambda r:r['native_cell_logits'].pop('mm'),
            lambda r:r['native_class_probabilities'].__setitem__(0,.9),
            lambda r:r.update(reward=1,progress=0.),
            lambda r:r['cell_attention_diagnostics']['mp']['0'].update(method='mass_transport'),
            lambda r:r['cell_attention_diagnostics']['mm']['0'].update(heads=[1]),
            lambda r:r.update(status='error')]
        for change in mutations:
            row = self.row(); change(row)
            with self.assertRaises((ValueError,KeyError)):
                verify_rows({'x':row},['x'],{(0,0)})
        with self.assertRaises(ValueError): verify_rows({'x':self.row()},['x','missing'],{(0,0)})

    def test_baseline_must_have_one_unsteered_forward(self):
        row = self.row(); z = np.asarray(row['native_class_logits_positive'])
        p = np.exp(z-z.max()); p /= p.sum()
        row.update(condition='baseline',actual_forward_branches=1,interaction_weight=0,
            native_cell_logits={},cell_attention_diagnostics={},native_interaction_logits=None,
            native_class_logits_combined=z.tolist(),native_class_probabilities=p.tolist())
        verify_rows({'x':row},['x'],baseline=True)
        row['actual_forward_branches'] = 4
        with self.assertRaises(ValueError): verify_rows({'x':row},['x'],baseline=True)

    def test_old_complete_event_does_not_complete_an_extension(self):
        # The only temporary data are synthetic fixtures; no research outputs touched.
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'worker_events.jsonl').write_text(json.dumps(dict(event='complete',arguments=dict(
                ks=[8],scopes=['all_frames'],controls=['target'],variant=VARIANT)))+'\n')
            (root/'requested_ids.json').write_text('["x"]')
            self.assertFalse(complete(root,[8,32],['all_frames'],1))
            self.assertFalse(complete(root,[8],['all_frames'],1))


if __name__ == '__main__':
    unittest.main()
