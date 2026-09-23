"""Original front-camera footage and forward progress, without attention capture."""
from __future__ import annotations

import copy
import csv
from numbers import Real
from pathlib import Path
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image

from mydata_bench.basic_method.common import create_json, file_hash, fingerprint
from mydata_bench.basic_method.grounding import eligibility
from .video_render import VideoWriter, comparison_frame, panel


def predict_progress_frame(runtime, sample, args, ranking, folder):
    """Generate the usual baseline/SAS scores without attaching a capture wrapper."""
    check = eligibility(sample, runtime.cfg, args.scope)
    condition = f'{args.scope}:target:{args.top_k}'
    baseline = runtime.predict(sample, 'baseline', ranking)
    if check['eligible']:
        sas = runtime.predict(sample, condition, ranking)
        for field in ('prompt_sha256', 'input_ids_sha256'):
            if sas['token_audit'][field] != baseline['token_audit'][field]:
                raise ValueError(f'Baseline/SAS progress inputs differ: {field}')
    else:
        sas = copy.deepcopy(baseline)
        sas.update(condition=condition, baseline_fallback=True, sas_applied=False,
                   fallback_reason=check['reason'], positive_bias=0., negative_bias=0.)
    index = sample['visualization_source_frame']
    image_path = sample['grm_image_paths'][5]
    record = {'source_index': index, 'source_time_seconds': index / sample['sampling']['source_fps'],
              'image_path': image_path, 'image_sha256': file_hash(image_path),
              'sample_id': fingerprint(sample), 'predictions': {'baseline': baseline, 'sas': sas},
              'grounding_check': check, 'sas_applied': check['eligible'], 'baseline_fallback': not check['eligible'],
              'positive_bias': runtime.cfg['bias'] if check['eligible'] else 0.,
              'negative_bias': -runtime.cfg['bias'] if check['eligible'] else 0., 'attention_captured': False}
    create_json(folder / 'progress_frames' / f'frame_{index:06d}.json', record)
    return record


def progress_series(frames):
    times = np.asarray([f['source_time_seconds'] for f in frames], dtype=float)
    if not len(times) or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Progress timestamps must be finite, nonnegative and strictly increasing')
    series = {}
    for name in ('baseline', 'sas'):
        values = []
        for frame in frames:
            row = frame['predictions'][name]
            value = row.get('progress')
            valid = (row.get('status') == 'ok' and isinstance(value, Real) and not isinstance(value, bool)
                     and np.isfinite(value) and 0 <= value <= 1)
            values.append(float(value) if valid else np.nan)
        series[name] = np.asarray(values)
    return times, series


class ProgressPlot:
    """Animate observed samples on fixed axes; do not smooth or force monotonicity."""

    def __init__(self, frames, size, duration):
        self.times, self.values = progress_series(frames)
        self.fallback = np.asarray([f['baseline_fallback'] for f in frames])
        self.fig, self.ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        self.fig.set_facecolor('#121821')
        self.ax.set_facecolor('#121821')
        self.fig.subplots_adjust(left=.13, right=.96, bottom=.16, top=.94)
        self.ax.set(xlim=(0, max(duration, self.times[-1], .1)), ylim=(-.04, 1.04),
                    xlabel='Source time (s)', ylabel='Task progress')
        self.ax.yaxis.set_major_formatter(PercentFormatter(1.))
        self.ax.tick_params(colors='#becadd', labelsize=9)
        self.ax.xaxis.label.set_color('#becadd')
        self.ax.yaxis.label.set_color('#becadd')
        for spine in self.ax.spines.values():
            spine.set_color('#354254')
        self.ax.grid(color='#354254', linewidth=.6, alpha=.65)
        self.lines = {name: self.ax.plot([], [], '-o', color=color, label=label, lw=2, ms=3)[0]
                      for name, color, label in [('baseline', '#5aadfa', 'GRM'), ('sas', '#eb9e56', 'GRM + SAS')]}
        self.fallback_points, = self.ax.plot([], [], 'o', color='#eb9e56', markerfacecolor='#121821', ms=6,
                                             label='SAS fallback' if self.fallback.any() else '_nolegend_', zorder=5)
        self.cursor = self.ax.axvline(0, color='#d9e1ed', lw=1, alpha=.6)
        self.ax.legend(loc='upper left', facecolor='#121821', edgecolor='#354254', labelcolor='#d9e1ed', fontsize=9)

    def draw(self, index, *, cursor=True):
        for name, line in self.lines.items():
            line.set_data(self.times[:index + 1], self.values[name][:index + 1])
        selected = self.fallback[:index + 1]
        self.fallback_points.set_data(self.times[:index + 1][selected], self.values['sas'][:index + 1][selected])
        self.cursor.set_xdata([self.times[index]] * 2)
        self.cursor.set_visible(cursor)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)


def render_progress_clip(source, manifest, output, args):
    """Render either saved attention-run predictions or a progress-only run."""
    from .video import rendering_options
    frames = manifest['frames']
    times, values = progress_series(frames)
    trajectory = manifest['trajectory']
    original = trajectory['source_videos']['front']
    if file_hash(original['path']) != original['sha256']:
        raise ValueError('Original front video differs from the saved run')
    for frame in frames:
        if file_hash(frame['image_path']) != frame['image_sha256']:
            raise ValueError('Saved source frame changed')
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the complete original video, with its original FPS and duration.
    shutil.copyfile(original['path'], output / 'original.mp4')
    with (output / 'progress_curve.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['source_index', 'source_time_seconds', 'grm_progress', 'sas_progress',
                                                   'grm_status', 'sas_status', 'baseline_fallback', 'fallback_reason'])
        writer.writeheader()
        for i, frame in enumerate(frames):
            writer.writerow({'source_index': frame['source_index'], 'source_time_seconds': times[i],
                             **{column: float(values[name][i]) if np.isfinite(values[name][i]) else ''
                                for column, name in [('grm_progress', 'baseline'), ('sas_progress', 'sas')]},
                             'grm_status': frame['predictions']['baseline'].get('status'),
                             'sas_status': frame['predictions']['sas'].get('status'),
                             'baseline_fallback': frame['baseline_fallback'],
                             'fallback_reason': frame.get('grounding_check', {}).get('reason')})
    with Image.open(frames[0]['image_path']) as im:
        size = im.size
    chart = ProgressPlot(frames, size, trajectory['source_duration_seconds'])
    fps = args.fps or len(frames) / trajectory['source_duration_seconds']
    writer = None
    try:
        chart.draw(len(frames) - 1, cursor=False)
        chart.fig.savefig(output / 'progress_curve.png', dpi=180)
        for i, frame in enumerate(frames):
            with Image.open(frame['image_path']) as im:
                pixels = np.asarray(im.convert('RGB'))
            if im.size != size:
                raise ValueError('Front video frame dimensions changed')
            graph = chart.draw(i)
            description = []
            for name, title in [('baseline', 'GRM'), ('sas', 'SAS')]:
                value = values[name][i]
                description.append(f'{title} {value:.1%}' if np.isfinite(value) else f'{title} invalid')
            if frame['baseline_fallback']:
                description.append('SAS: baseline fallback')
            cards = [panel(pixels, 'Front camera', f'Source frame {frame["source_index"]}  |  t={times[i]:.2f} s'),
                     panel(graph, 'Task progress', '  |  '.join(description), accent=(235, 158, 86))]
            subtitle = (f'{manifest["example_id"]}  |  {manifest["scope"]} top-{manifest["top_k"]}'
                        f'  |  sampled {i + 1}/{len(frames)}  |  t={times[i]:.2f}s')
            composed = comparison_frame(cards, manifest['task'], subtitle,
                                         frame['source_index'] / max(1, trajectory['source_frame_count'] - 1),
                                         'Sampled forward predictions | Gaps: invalid output | Hollow SAS points: baseline fallback')
            if writer is None:
                writer = VideoWriter(output / 'video_progress.mp4', (composed.shape[1], composed.shape[0]), fps)
            writer.write(composed)
            if i == len(frames) // 2:
                Image.fromarray(composed).save(output / 'preview.png')
    finally:
        chart.close()
        if writer is not None:
            writer.close()
    source_manifest = source / ('progress_video_manifest.json' if manifest.get('attention_captured') is False
                                else 'attention_video_manifest.json')
    result = {'videos': {'original': 'original.mp4', 'progress': 'video_progress.mp4'},
              'artifacts': {'curve': 'progress_curve.png', 'curve_csv': 'progress_curve.csv'},
              'progress_only': True, 'num_samples': len(frames), 'fallback_frames': sum(f['baseline_fallback'] for f in frames),
              'fps': fps, 'duration_seconds': len(frames) / fps,
              'original_duration_seconds': trajectory['source_duration_seconds'], 'original_sha256': original['sha256'],
              'options': rendering_options(args), 'source_manifest': str(source_manifest.resolve()),
              'curve_note': 'Raw forward progress, no smoothing/accumulation; invalid predictions are gaps.'}
    create_json(output / 'render.json', result)
    return result
