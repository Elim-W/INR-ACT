"""
Per-task avg-best-PSNR heatmaps over the high/mid/low 5-image groups.

For each task in {fitting, sr, inpaint}, walk the per-image *_results.pt
files and produce a heatmap:
    rows = 9 methods (cosmo + incode excluded)
    cols = low / mid / high
    cell = mean of best_psnr over images in that group

Methods are ordered top-down by their overall avg PSNR (strongest at top),
re-sorted independently per task.

Cells with no .pt files render as 'n/a' (e.g. SR low_5 if the resume
sbatch hasn't finished yet).

Usage:
    python donghua/plot_avg_psnr_heatmaps.py
    python donghua/plot_avg_psnr_heatmaps.py --tasks fitting sr
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


GROUPS  = ['low_5', 'mid_5', 'high_5']
METHODS = ['siren', 'wire', 'gauss', 'finer', 'gf', 'wf', 'staf', 'relu', 'sl2a']
COL_LABELS = ['low', 'mid', 'high']

TASK_CONFIGS = {
    'fitting': {
        'root':  Path('results/image_fitting_groupss'),
        'title': 'Image Fitting — avg best PSNR',
        'out':   Path('donghua/heatmap_fitting_avg_psnr.png'),
    },
    'sr': {
        'root':  Path('results/image_super_resolution_groupss'),
        'title': 'Super-Resolution (×8) — avg best HR PSNR',
        'out':   Path('donghua/heatmap_sr_avg_psnr.png'),
    },
    'inpaint': {
        'root':  Path('results/image_inpaint_groups'),
        'title': 'Inpainting (sampling_ratio=0.2) — avg best PSNR',
        'out':   Path('donghua/heatmap_inpaint_avg_psnr.png'),
    },
}


def _to_py(v):
    if v is None:
        return None
    return v.item() if hasattr(v, 'item') else float(v)


def collect_avg_psnr(root):
    """Return matrix[method, group] = mean best_psnr (NaN if no data)."""
    M, G = len(METHODS), len(GROUPS)
    mat = np.full((M, G), np.nan)
    n_imgs = np.zeros((M, G), dtype=int)
    for gi, g in enumerate(GROUPS):
        for mi, m in enumerate(METHODS):
            mdir = root / g / m
            if not mdir.is_dir():
                continue
            psnrs = []
            for pt in sorted(mdir.glob('*_results.pt')):
                d = torch.load(pt, map_location='cpu', weights_only=False)
                v = _to_py(d.get('best_psnr'))
                if v is not None:
                    psnrs.append(v)
            if psnrs:
                mat[mi, gi] = float(np.mean(psnrs))
                n_imgs[mi, gi] = len(psnrs)
    return mat, n_imgs


def heatmap(matrix, n_imgs, row_labels, col_labels, title, out_path):
    M, G = matrix.shape
    fig, ax = plt.subplots(figsize=(1.5 * G + 2.4, 0.55 * M + 1.6), dpi=160)

    valid = np.isfinite(matrix)
    if not valid.any():
        plt.close(fig)
        print(f"[skip] {title}: no data")
        return

    vmin = float(np.nanmin(matrix[valid]))
    vmax = float(np.nanmax(matrix[valid]))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(G), labels=col_labels)
    ax.set_yticks(range(M), labels=row_labels)
    ax.set_title(title, fontsize=11, pad=10)

    # Cell annotation
    span = max(vmax - vmin, 1e-9)
    for i in range(M):
        for j in range(G):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center',
                        color='gray', fontsize=9)
                continue
            t = (v - vmin) / span
            color = 'white' if t < 0.55 else 'black'
            label = f'{v:.2f}'
            if n_imgs[i, j] != 5:
                label += f'\n(n={n_imgs[i, j]})'
            ax.text(j, i, label, ha='center', va='center',
                    color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('avg best PSNR (dB)', fontsize=9)
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
    for task in args.tasks:
        cfg = TASK_CONFIGS[task]
        if not cfg['root'].exists():
            print(f"[skip] {task}: {cfg['root']} does not exist")
            continue
        print(f"\n=== {task}: scanning {cfg['root']} ===")
        mat, n_imgs = collect_avg_psnr(cfg['root'])

        # No sorting — use METHODS order as-is.
        mat_s    = mat
        nimg_s   = n_imgs
        names    = METHODS

        # Compact stdout table
        print(f"{'method':<8}  " + '  '.join(f'{c:>6}' for c in COL_LABELS))
        for mi, m in enumerate(names):
            row = '  '.join(
                f'{mat_s[mi, gi]:>6.2f}' if np.isfinite(mat_s[mi, gi]) else '   n/a'
                for gi in range(len(GROUPS))
            )
            print(f'{m:<8}  {row}')

        cfg['out'].parent.mkdir(parents=True, exist_ok=True)
        heatmap(mat_s, nimg_s, names, COL_LABELS, cfg['title'], cfg['out'])


if __name__ == '__main__':
    main()
