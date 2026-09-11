"""Plot logged key footprints over actual processor inputs, without model inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

from .prepare import OUT, create_json
from .protocols import prompt_payload


def read_example(folder, eid):
    found = None
    with (folder / 'predictions/baseline.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['example_id'] == eid:
                found = row
    assert found is not None and found['status'] == 'ok'
    return found


def draw(ax, img, row, record, scope, original_size, title):
    img = np.asarray(img)
    height, width = img.shape[:2]
    _, raw_h, raw_w = record['grid_thw']
    gh, gw = raw_h // 2, raw_w // 2
    start, end = record['start'], record['end']
    assert end - start == gh * gw
    rgba = np.zeros((gh * gw, 4), dtype=float)
    audit = row['token_audit']
    for key in audit['negative'][scope]:
        if start <= key < end:
            rgba[key - start] = [0, .6, 1, .28]
    for key in audit['target'][scope]:
        if start <= key < end:
            rgba[key - start] = [1, .05, .05, .48]
    ax.imshow(img, extent=(0, width, height, 0))
    ax.imshow(rgba.reshape(gh, gw, 4), extent=(0, width, height, 0), interpolation='nearest')
    ow, oh = original_size
    for x1, y1, x2, y2 in record['boxes']:
        ax.add_patch(Rectangle((x1 * width / ow, y1 * height / oh), (x2 - x1) * width / ow,
                               (y2 - y1) * height / oh, fill=False, edgecolor='yellow', linewidth=1.3))
    ax.set_title(title + f' | merged grid {gh}x{gw}', fontsize=10)
    ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis('off')


def save(fig, folder, stem):
    for suffix in ['png', 'svg']:
        path = folder / f'{stem}.{suffix}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='alignment_visualization_v1')
    args = parser.parse_args()
    folder = OUT / args.output_name
    folder.mkdir(exist_ok=False)
    sample = next(s for s in json.loads((OUT / 'inputs.json').read_text()) if s['ranking'])
    rows, payloads = {}, {}
    names = ['meter_interleaved', 'meter_official', 'sole_video_text', 'sole_official']
    for name in names:
        rows[name] = read_example(OUT / name, sample['example_id'])
        cfg = json.loads((OUT / name / 'run_config.json').read_text())
        payloads[name] = prompt_payload(sample, cfg, 7 if name == 'sole_official' else None,
                                        rows[name].get('previous_percentage') or 0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name in zip(axes[0], names[:2]):
        record = rows[name]['token_audit']['alignment']['last_frame'][0]
        draw(ax, payloads[name]['images'][-1], rows[name], record, 'last_frame',
             payloads[name]['span_specs'][-1]['size'], name + ' / terminal image')
    name = 'sole_video_text'
    record = rows[name]['token_audit']['alignment']['last_frame'][0]
    for ax, index, source in zip(axes[1], [6, 7], record['source_frames']):
        img = payloads[name]['videos'][0][index].permute(1, 2, 0).numpy()
        draw(ax, img, rows[name], record, 'last_frame', payloads[name]['span_specs'][-1]['size'],
             f'Native video / source frame {source}\nShared final-tubelet keys')
    legend = [Patch(color='red', alpha=.5, label='target keys (+6 when steered)'),
              Patch(color='deepskyblue', alpha=.4, label='negative-domain keys (-6)'),
              Patch(facecolor='none', edgecolor='yellow', label='logged bbox / two-frame bbox union')]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle(sample['example_id'] + '\n' + sample['task'], fontsize=11)
    fig.tight_layout(rect=(0, .055, 1, .93)); save(fig, folder, 'image_and_tubelet')
    name = 'sole_official'
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for ax, scope in zip(axes, ['last_frame', 'all_frames']):
        record = rows[name]['token_audit']['alignment'][scope][0]
        draw(ax, payloads[name]['images'][0], rows[name], record, scope,
             payloads[name]['span_specs'][0]['size'], f'SOLE official / {scope} / first, previous, current')
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('Logged spatial key footprints; grid cells can span padding or adjacent mosaic tiles', fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, .95)); save(fig, folder, 'sole_official_mosaic')
    footprints = {}
    for name in names:
        audit = rows[name]['token_audit']
        target = set(audit['target']['last_frame'])
        domain = target | set(audit['negative']['last_frame'])
        footprints[name] = {'last_target_cells': len(target), 'last_domain_cells': len(domain),
                            'target_fraction_of_selected_domain': len(target) / len(domain)}
    name = 'sole_official'
    spec = payloads[name]['span_specs'][0]
    sw, sh = spec['source_size']
    scale = 384 / max(sw, sh)
    rw, rh = int(sw * scale), int(sh * scale)
    ox, oy = (384 - rw) // 2, (384 - rh) // 2
    content = [[389 * tile + ox, oy, 389 * tile + ox + rw, oy + rh] for tile in range(3)]
    record = rows[name]['token_audit']['alignment']['last_frame'][0]
    gh, gw = record['grid_thw'][1] // 2, record['grid_thw'][2] // 2
    cw, ch = spec['size']
    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    audit = rows[name]['token_audit']
    mixed, noncurrent = [], []
    for key in sorted(set(audit['target']['last_frame']) | set(audit['negative']['last_frame'])):
        relative = key - record['start']; y, x = divmod(relative, gw)
        cell = [x * cw / gw, y * ch / gh, (x + 1) * cw / gw, (y + 1) * ch / gh]
        area = (cell[2] - cell[0]) * (cell[3] - cell[1])
        if overlap(cell, content[2]) < area - 1e-6:
            mixed.append(key)
        if any(overlap(cell, other) > 1e-6 for other in content[:2]):
            noncurrent.append(key)
    footprints[name]['selected_cells_not_wholly_inside_current_content'] = len(mixed)
    footprints[name]['selected_cells_also_intersecting_noncurrent_content'] = len(noncurrent)
    footprints[name]['noncurrent_intersection_key_positions'] = noncurrent
    create_json(folder / 'provenance.json', {'example_id': sample['example_id'],
                'selection': 'First example marked ranking in immutable inputs.json; no score-based selection.',
                'sources': {name: str(OUT / name / 'predictions/baseline.jsonl') for name in names},
                'purpose': 'Visualization of logged key selections, not independent re-grounding or an inference rerun.',
                'key_rule': 'Every merged visual cell intersecting a bbox/content rectangle is included. Boundary cells can contain pixels outside the rectangle.',
                'representative_footprints': footprints})
    print(folder)


if __name__ == '__main__':
    main()
