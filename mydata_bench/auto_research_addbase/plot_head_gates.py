"""Final-only parameter/optimization diagnostics for the head-gate capacity study."""
import csv
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .head_gate_worker import BASE, sha, verify_sources


def main():
    audit_file = BASE/'fixed_training_audit_v1.json'
    audit = json.loads(audit_file.read_text())
    if audit['status'] != 'pass': raise ValueError('Only both complete final audited checkpoints may be plotted')
    verify_sources(audit['sources_sha256'])
    sources = {str(audit_file): sha(audit_file)}; records = []; traces = []; diagnostics = []
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    def read(path, lines=False):
        sources[str(path)] = sha(path)
        return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())
    for row, model in enumerate(['qwen', 'roboreward']):
        root = BASE/'training'/model
        checkpoint = read(root/'final_gates.json'); values = np.asarray(checkpoint['gate_values'])
        coverage = read(root/'training_head_coverage.json')
        active = np.zeros((28, 32), dtype=bool)
        for l, h in coverage['training_head_union']: active[l-8, h] = True
        for j, domain in enumerate(['Visual', 'Instruction']):
            ax = axes[row, j]
            im = ax.imshow(values[:, :, j], origin='lower', aspect='auto', extent=(-.5, 31.5, 7.5, 35.5), vmin=0, vmax=1, cmap='viridis')
            ys, xs = np.nonzero(~active)
            ax.scatter(xs, ys+8, marker='.', s=4, c='white', alpha=.5)
            ax.set(xlabel='Head index', ylabel='Layer index', title=f'{model}: {domain} sigmoid gates')
            fig.colorbar(im, ax=ax, fraction=.04)
        for l in range(28):
            for h in range(32):
                records.append(dict(model=model, layer=l+8, head=h, visual_gate=float(values[l,h,0]),
                    task_gate=float(values[l,h,1]), selected_by_any_training_condition=bool(active[l,h])))
        diagnostics.append(dict(model=model, trainable_parameters=1792, unique_training_heads=int(active.sum()),
            regularizer_only_heads=int((~active).sum()),
            visual_gate_sum_on_training_union=float(values[:,:,0][active].sum()),
            task_gate_sum_on_training_union=float(values[:,:,1][active].sum()),
            regularizer_only_gate_range=[float(values[~active].min()),float(values[~active].max())]))
        ax = axes[row, 2]
        for label, path in [('Layer gates (56)', OUT/'learned_attention_gates_v1/training'/model/'optimizer_events.jsonl'),
                            ('Head gates (1792)', root/'optimizer_events.jsonl')]:
            events = read(path, True)
            if len(events) != 525: raise ValueError('Incomplete fixed trajectory')
            losses = np.asarray([e['mean_loss'] for e in events])
            smoothed = [float(losses[max(0,i-24):i+1].mean()) for i in range(len(losses))]
            ax.plot(range(1,526), smoothed, label=label)
            traces += [dict(model=model, parameterization=label, update=e['optimizer_steps'],
                mean_loss=e['mean_loss'], trailing25_loss=s) for e,s in zip(events,smoothed)]
        ax.set(xlabel='Optimizer update', ylabel='Balanced five-class CE + gate penalty', title=f'{model}: same fixed training schedule')
        ax.legend(); ax.grid(alpha=.2)
        for boundary in [175,350]: ax.axvline(boundary, color='gray', ls=':', lw=.8)
    fig.suptitle('Round32: fixed final1792 attention scalars per model; backbone/output weights frozen')
    fig.supxlabel('White dots: head never selected in training (regularizer only). Discovery70 reused across layouts and epochs; these are not efficacy plots.')
    verify_sources(sources)
    dest = OUT/'analysis'/time.strftime('figures_head_gates_fixed_training_%Y%m%d_%H%M%S'); dest.mkdir()
    fig.savefig(dest/'head_gates_and_training.png', dpi=150)
    fig.savefig(dest/'head_gates_and_training.pdf'); plt.close(fig)
    for name, rows in [('gates',records),('training_trace',traces)]:
        create_json(dest/f'{name}.json',rows)
        with (dest/f'{name}.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    create_json(dest/'manifest.json',dict(sources_sha256=sources,diagnostics=diagnostics,
        interpretation='Final fixed-training diagnostics only. Does not assess held-out or resubstitution efficacy.'))
    print(dest,flush=True)


if __name__ == '__main__': main()
