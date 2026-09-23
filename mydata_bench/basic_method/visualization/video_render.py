"""Smooth, legible video overlays inspired by visualize_stage3_head_attention.

Spatial interpolation is for display only. Numeric metrics and saved attention
always use the original token grid, without spatial or temporal filtering.
"""
from __future__ import annotations

import html
from pathlib import Path
import shutil
import subprocess
import textwrap

import cv2
import numpy as np
from PIL import Image


def display_heat(grid, size, *, scale='reference', maximum=None, blur_sigma=3.):
    values = np.asarray(grid, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected a finite, nonnegative spatial attention grid')
    if scale == 'reference':
        # Same per-frame min/max normalization as the reference probe.
        values = values - values.min()
        denominator = float(values.max())
    elif scale == 'shared' and maximum is not None and np.isfinite(maximum) and maximum > 0:
        denominator = maximum
    else:
        raise ValueError('Shared display needs a finite positive clip-wide maximum')
    values = values / denominator if denominator > 0 else np.zeros_like(values)
    # Keep floating point precision and interpolate once at the display size.
    heat = np.asarray(Image.fromarray(values).resize(size, Image.Resampling.BICUBIC)).copy()
    if blur_sigma:
        heat = cv2.GaussianBlur(heat, (0, 0), blur_sigma)
    return np.clip(heat, 0, 1)


def overlay(image, grid, box=None, *, scale='reference', maximum=None, alpha=.5, blur_sigma=3.):
    pixels = np.asarray(image, dtype=np.uint8)
    height, width = pixels.shape[:2]
    heat = display_heat(grid, (width, height), scale=scale, maximum=maximum, blur_sigma=blur_sigma)
    color = cv2.cvtColor(cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    # Fade low-intensity areas so the camera view stays clear; no uniform blue wash.
    opacity = (alpha * np.sqrt(heat))[..., None]
    result = np.uint8(np.clip(pixels * (1 - opacity) + color * opacity, 0, 255))
    if box is not None:
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (70, 235, 145), 1, cv2.LINE_AA)
    return result


def panel(image, title, detail, *, accent=(87, 165, 250)):
    height, width = image.shape[:2]
    canvas = np.full((height + 82, width, 3), (18, 24, 33), dtype=np.uint8)
    canvas[58:58 + height] = image
    cv2.rectangle(canvas, (0, 0), (width - 1, 3), accent, -1)
    cv2.putText(canvas, title, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, .72,
                (243, 246, 252), 1, cv2.LINE_AA)
    cv2.putText(canvas, detail, (12, height + 74), cv2.FONT_HERSHEY_SIMPLEX, .42,
                (192, 202, 216), 1, cv2.LINE_AA)
    return canvas


def comparison_frame(panels, task, subtitle, position, scale_label):
    width = sum(p.shape[1] for p in panels) + 12 * (len(panels) - 1)
    height = max(p.shape[0] for p in panels)
    # Even dimensions are required by H.264 yuv420p.
    total_height = height + 142
    canvas = np.full((total_height + total_height % 2, width + width % 2, 3), (11, 16, 24), dtype=np.uint8)
    task_lines = textwrap.wrap(task, max(30, width // 11))[:2]
    for i, line in enumerate(task_lines):
        cv2.putText(canvas, line, (16, 28 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, .67,
                    (240, 244, 250), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (16, 80), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (161, 178, 202), 1, cv2.LINE_AA)
    x = 0
    for p in panels:
        canvas[98:98 + p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + 12
    bar_y = 98 + height + 10
    cv2.rectangle(canvas, (16, bar_y), (width - 16, bar_y + 3), (49, 60, 78), -1)
    cv2.rectangle(canvas, (16, bar_y), (16 + int((width - 32) * position), bar_y + 3), (90, 173, 250), -1)
    cv2.putText(canvas, scale_label, (16, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, .42,
                (169, 187, 209), 1, cv2.LINE_AA)
    return canvas


class VideoWriter:
    """Stream RGB frames to a browser-compatible H.264 MP4, without buffering a clip."""

    def __init__(self, path, size, fps):
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('ffmpeg with libx264 is required for playable MP4 output')
        self.path = Path(path)
        self.size = size
        self.frames = 0
        self.process = subprocess.Popen([
            executable, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-',
            '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '19',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        if frame.shape != (self.size[1], self.size[0], 3) or frame.dtype != np.uint8:
            raise ValueError('Video frame size/dtype changed')
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(self.process.stderr.read().decode(errors='replace')) from exc
        self.frames += 1

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        error = self.process.stderr.read().decode(errors='replace')
        code = self.process.wait()
        self.process.stderr.close()
        if code or not self.frames:
            raise RuntimeError(f'Video encoding failed ({code}): {error}')


def write_video_gallery(output, items):
    escape = html.escape
    cards = []
    for item in items:
        links = ' · '.join(f'<a href="{escape(item[name], quote=True)}">{name}</a>'
                           for name in ('original', 'baseline', 'sas', 'curve', 'curve_csv', 'metadata') if name in item)
        video = item.get('comparison', item.get('progress'))
        cards.append(f'<article><h2>{escape(item["example_id"])}</h2><p>{escape(item["task"])}</p>'
                     f'<video controls preload="metadata" poster="{escape(item["preview"], quote=True)}" '
                     f'src="{escape(video, quote=True)}"></video><p>{links}</p>'
                     f'<p>{item["num_samples"]} sampled frames · {item["fallback_frames"]} baseline fallback frames</p></article>')
    progress_only = all('progress' in item for item in items)
    title = 'GRM / SAS progress curves' if progress_only else 'GRM / SAS attention videos'
    description = ('Original front video and sampled forward progress. Gaps indicate invalid predictions; '
                   'hollow SAS markers indicate baseline fallback.' if progress_only else
                   'Whole-trajectory sampled forward inference. The video footer identifies the display normalization; '
                   'raw masses are shown separately.')
    (Path(output) / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><title>{title}</title>'
        '<style>body{background:#0b1018;color:#eaf0f9;font:16px system-ui;margin:32px}'
        'article{padding:20px;background:#141c29;border-radius:14px;margin:24px 0}'
        'video{width:100%;max-height:80vh}a{color:#84b9ff}h2{font-size:19px}</style>'
        f'<h1>{title}</h1><p>{description}</p>'
        + ''.join(cards) + '</html>')
