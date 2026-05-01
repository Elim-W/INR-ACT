"""
Per-image strip plots for fitting / SR / inpaint.

For each task, produce one figure with 3 panels (low / mid / high). In each
panel, every method gets 5 dots (one per image in the group), so you can see:
  - within-group variance: do all 5 images give similar PSNR?
  - method ranking stability: does method A always beat B, or just on average?
  - outlier images: any single image where a strong method tanks?

A thin horizontal black tick marks the per-method mean.

Output:
    donghua/strip_fitting.png
    donghua/strip_sr.png
    donghua/strip_inpaint.png

Usage:
    python donghua/plot_strip.py
    python donghua/plot_strip.py --tasks fitting sr
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


GROUPS  = ['low_5', 'mid_5', 'high_5']
GROUP_LABELS = ['low', 'mid', 'high']
METHODS = ['siren', 'wire', 'gauss', 'finer', 'gf', 'wf', 'staf', 'relu', 'sl2a']

TASK_CONFIGS = {
    'fitting': {
        'root':  Path('results/image_fitting_groupss'),
        'title': 'Image Fitting — best PSNR per image',
        'out':   Path('donghua/strip_fitting.png'),
    },
    'sr': {
        'root':  Path('results/image_super_resolution_groupss'),
        'title': 'Super-Resolution (×8) — best HR PSNR per image',
        'out':   Path('donghua/strip_sr.png'),
    },
    'inpaint': {
        'root':  Path('results/image_inpaint_groups'),
        'title': 'Inpainting (sampling_ratio=0.2) — best PSNR per image',
        'out':   Path('donghua/strip_inpaint.png'),
    },
}


def _to_py(v):
    if v is None:
        return None
    return v.item() if hasattr(v, 'item') else float(v)


def collect(root):
    """Return values[group][method] = list of best_psnr (one per image)."""
    out = {g: {m: [] for m in METHODS} for g in GROUPS}
    for g in GROUPS:
        for m in METHODS:
            mdir = root / g / m
            if not mdir.is_dir():
                continue
            for pt in sorted(mdir.glob('*_results.pt')):
                d = torch.load(pt, map_location='cpu', weights_only=False)
                v = _to_py(d.get('best_psnr'))
                if v is not None:
                    out[g][m].append(v)
    return out


def plot_task(values, title, out_path):
    n_methods = len(METHODS)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_methods)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False, dpi=160)
    rng = np.random.default_rng(0)

    # Per-panel y-range (each group has its own scale)
    for ax, g, glab in zip(axes, GROUPS, GROUP_LABELS):
        for mi, m in enumerate(METHODS):
            ys = values[g][m]
            if not ys:
                continue
            xs = mi + (rng.random(len(ys)) - 0.5) * 0.32     # jitter
            ax.scatter(xs, ys, color=colors[mi], s=42,
                       edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3)
            mean = float(np.mean(ys))
            ax.hlines(mean, mi - 0.28, mi + 0.28,
                      color='black', linewidth=1.6, zorder=4)
            ax.text(mi, mean, f' {mean:.1f}', va='center', ha='left',
                    fontsize=7, color='black', alpha=0.7, zorder=5)

        ax.set_xticks(range(n_methods), labels=METHODS, rotation=45, ha='right')
        ax.set_title(f'{glab} (n=5)', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=1)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel('best PSNR (dB)', fontsize=10)
        ax.set_xlim(-0.6, n_methods - 0.4)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='+',
                   default=list(TASK_CONFIGS.keys()),
                   choices=list(TASK_CONFIGS.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    Path('donghua').mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        cfg = TASK_CONFIGS[task]
        if not cfg['root'].exists():
            print(f"[skip] {task}: {cfg['root']} does not exist")
            continue
        print(f"\n=== {task}: scanning {cfg['root']} ===")
        values = collect(cfg['root'])
        # quick sanity print
        for g in GROUPS:
            counts = {m: len(values[g][m]) for m in METHODS}
            print(f'  {g}: ' + '  '.join(f'{m}={n}' for m, n in counts.items()))
        plot_task(values, cfg['title'], cfg['out'])


if __name__ == '__main__':
    main()
