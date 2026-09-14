"""Paired attention figures with a shared scale and an explicit difference map."""
from __future__ import annotations

import html
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from PIL import Image

# Reuse the reference probe's token-to-spatial-grid convention. Its overlay
# helpers independently min/max normalize each image, so do not use those for
# baseline/SAS comparisons.
from scan_localization_heads_best import vector_to_grid


def paired_maps(baseline, sas, normalization):
    arrays = [np.asarray(a, dtype=np.float64).copy() for a in (baseline, sas)]
    if arrays[0].shape != arrays[1].shape or any(not np.isfinite(a).all() or (a < 0).any() for a in arrays):
        raise ValueError('Expected matching finite nonnegative attention grids')
    if normalization == 'image_fraction':
        arrays = [a / a.sum() if a.sum() > 0 else np.zeros_like(a) for a in arrays]
    elif normalization != 'raw':
        raise ValueError(normalization)
    maximum = max(float(a.max()) for a in arrays)
    return *arrays, max(maximum, 1e-12)


def grid_metrics(grid, target):
    mass = float(grid.sum())
    target_mass = float(grid[target].sum()) if target is not None else None
    return {'image_mass': mass, 'target_mass': target_mass,
            'target_fraction': target_mass / mass if target_mass is not None and mass > 0 else None}


def save_comparison(path, image_path, baseline, sas, box, target_mask, title, subtitles,
                    normalization='raw', alpha=.5):
    with Image.open(image_path) as source:
        pixels = np.asarray(source.convert('RGB'))
    height, width = pixels.shape[:2]
    a, b, maximum = paired_maps(baseline, sas, normalization)
    delta = b - a
    difference_max = max(float(np.abs(delta).max()), 1e-12)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.9), layout='constrained')
    fig.suptitle(textwrap.fill(title, 140), fontsize=11)
    extent = (-.5, width - .5, height - .5, -.5)
    normalizer = Normalize(0, maximum)
    delta_normalizer = TwoSlopeNorm(vmin=-difference_max, vcenter=0, vmax=difference_max)
    for ax, label in zip(axes, ['Front camera', *subtitles, 'SAS − baseline']):
        ax.imshow(pixels)
        if box is not None:
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      fill=False, color='#37ed78', linewidth=1.5))
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-.5, width - .5)
        ax.set_ylim(height - .5, -.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, grid in zip(axes[1:3], (a, b)):
        heat = ax.imshow(grid, extent=extent, cmap='jet', norm=normalizer,
                         interpolation='nearest', alpha=alpha)
    diff = axes[3].imshow(delta, extent=extent, cmap='RdBu_r', norm=delta_normalizer,
                         interpolation='nearest', alpha=alpha)
    # Opaque colorbars label the heatmap values, independently of the overlay alpha.
    unit = 'attention probability / token' if normalization == 'raw' else 'probability / token, conditioned on this image'
    fig.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=heat.cmap), ax=axes[1:3],
                 orientation='horizontal', shrink=.85, pad=.06, label=unit)
    fig.colorbar(plt.cm.ScalarMappable(norm=delta_normalizer, cmap=diff.cmap), ax=axes[3],
                 orientation='horizontal', shrink=.85, pad=.06, label='change in displayed probability')
    metrics = [grid_metrics(g, target_mask) for g in (baseline, sas)]
    for ax, metric in zip(axes[1:3], metrics):
        fraction = metric['target_fraction']
        suffix = f'   target/image={fraction:.1%}' if fraction is not None else '   target unavailable'
        ax.set_xlabel(f'raw image mass={metric["image_mass"]:.4f}' + suffix, fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {'baseline': metrics[0], 'sas': metrics[1], 'shared_vmin': 0., 'shared_vmax': maximum,
            'difference_vmax': difference_max, 'normalization': normalization}


def write_gallery(output, items, title):
    escape = html.escape
    cards = []
    for item in items:
        figures = ''.join(f'<a href="{escape(p, quote=True)}"><img loading="lazy" src="{escape(p, quote=True)}"></a>'
                          for p in item['figures'])
        cards.append(f'<article><h2>{escape(item["example_id"])}</h2>'
                     f'<p>{escape(item["task"])}</p><p>{escape(item["status"])}</p>{figures}'
                     f'<p><a href="{escape(item["metadata"], quote=True)}">Metadata and per-head metrics</a></p></article>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)}</title><style>'
            'body{font:16px system-ui;margin:32px;background:#f4f6f8;color:#152638}'
            'article{background:white;margin:24px 0;padding:20px;border-radius:12px}'
            'img{width:100%;height:auto}a{color:#165bba}h2{font-size:18px}</style>'
            f'<h1>{escape(title)}</h1><p>Last prompt query. Each baseline/SAS pair uses the same heads and color scale. '
            'Green: exact-frame target box. Red/blue: increased/decreased displayed attention. '
            'Separate samples and heads may use different scales.</p>' + ''.join(cards) + '</html>')
    (Path(output) / 'index.html').write_text(page)
