"""Final fixed-step gate values and descriptive training optimization traces."""
import csv
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .learned_gate_worker import BASE
from .empirical_profile import sha, verify_sources


def main():
    audit = BASE/'fixed_training_audit_v1.json'
    a = json.loads(audit.read_text())
    if a['status'] != 'pass': raise ValueError('Require completed fixed training audit')
    verify_sources(a['sources_sha256'])
    destination = OUT/'analysis'/time.strftime('figures_learned_gates_fixed_training_%Y%m%d_%H%M%S')
    destination.mkdir(exist_ok=False)
    sources = {str(audit): sha(audit)}; points = []; traces = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for i, model in enumerate(['qwen', 'roboreward']):
        root = BASE/'training'/model
        p = root/'final_gates.json'; sources[str(p)] = sha(p); record = json.loads(p.read_text())
        values = np.asarray(record['gate_values'])
        for layer, (visual, task) in enumerate(values, 8):
            points.append(dict(model=model, layer=layer, visual_gate=float(visual), task_gate=float(task),
                visual_logit_bias=float(6*visual), task_logit_bias=float(4*task)))
        x = np.arange(8, 36)
        axes[i, 0].plot(x, values[:, 0], marker='o', markersize=3, label='Visual gate (cap 6)', color='#2563a6')
        axes[i, 0].plot(x, values[:, 1], marker='s', markersize=3, label='Instruction gate (cap 4)', color='#ad6525')
        axes[i, 0].axhline(1/(1+np.exp(2)), color='.5', linestyle=':', label='Shared initialization')
        axes[i, 0].set(xlabel='Language layer', ylabel='Learned sigmoid gate', ylim=(-.03, 1.03), title=model)
        axes[i, 0].grid(alpha=.2); axes[i, 0].legend(fontsize=8)
        log = root/'optimizer_events.jsonl'; sources[str(log)] = sha(log)
        events = [json.loads(line) for line in log.read_text().splitlines()]
        if len(events) != 525: raise ValueError('Only complete fixed 525-update traces may be plotted')
        loss = np.array([e['mean_loss'] for e in events])
        smoothed = np.array([loss[max(0, j-24):j+1].mean() for j in range(len(loss))])
        steps = np.arange(1, 526)
        axes[i, 1].plot(steps, loss, alpha=.2, color='#67778b', linewidth=.7, label='Actual eight-example update loss')
        axes[i, 1].plot(steps, smoothed, color='#334a64', label='Trailing 25 updates')
        for boundary in [175, 350]: axes[i, 1].axvline(boundary, color='.6', linestyle=':', linewidth=.8)
        axes[i, 1].set(xlabel='Optimizer update (fixed final 525)', ylabel='Class-balanced CE + gate penalty', title=model+' optimization')
        axes[i, 1].grid(alpha=.2); axes[i, 1].legend(fontsize=8)
        traces.extend(dict(model=model, update=e['optimizer_steps'], forward_backward_steps=e['forward_backward_steps'],
            epoch=e['epoch'], actual_update_loss=e['mean_loss'], trailing25_loss=float(smoothed[j])) for j, e in enumerate(events))
    fig.suptitle('56 global attention scalars per model; all backbone and output weights frozen')
    fig.supxlabel('Discovery70 supervision; layouts and epochs reuse the same examples. Parameter/optimization diagnostics are not efficacy evidence.', fontsize=9)
    fig.savefig(destination/'gates_and_training.png', dpi=170)
    fig.savefig(destination/'gates_and_training.pdf'); plt.close(fig)
    for name, rows in [('gates', points), ('training_trace', traces)]:
        create_json(destination/f'{name}.json', rows)
        with (destination/f'{name}.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    create_json(destination/'manifest.json', dict(sources_sha256=sources,
        interpretation='Complete prespecified optimization, no checkpoint selection. No validation or efficacy claim.'))
    print(destination)


if __name__ == '__main__': main()
