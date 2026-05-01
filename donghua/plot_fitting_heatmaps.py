"""
Two fitting heatmaps for the high/mid/low INR benchmark:

  1. method_group_avg_psnr_heatmap.png
     Rows = 11 methods, Cols = low/mid/high, Cells = avg best PSNR.
     Lets you see absolute method strength per frequency band.

  2. method_group_delta_psnr_heatmap.png
     Rows = 11 methods, Cols = low/mid/high, Cells = avg ΔPSNR vs best
     method on the same image (≤ 0).  Removes per-image difficulty so
     you can compare method ranks fairly across frequency bands.

Reads from:
    results/image_fitting_groupss/<group>/<method>/<name>_results.pt

Output:
    donghua/method_group_avg_psnr_heatmap.png
    donghua/method_group_delta_psnr_heatmap.png
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


GROUPS  = ['low_5', 'mid_5', 'high_5']                 # column order: low → high
METHODS = ['siren', 'wire', 'gauss', 'finer', 'gf', 'wf',
           'staf', 'relu', 'incode', 'sl2a', 'cosmo']
ROOT    = Path('results/image_fitting_groupss')
OUT_DIR = Path('donghua')


def _to_py(v):
    if v is None:
        return None
    return v.item() if hasattr(v, 'item') else float(v)


def collect():
    """Return data[group][method][image_name] = best_psnr (float)."""
    data = {g: {m: {} for m in METHODS} for g in GROUPS}
    for g in GROUPS:
        for m in METHODS:
            mdir = ROOT / g / m
            if not mdir.is_dir():
                continue
            for pt in sorted(mdir.glob('*_results.pt')):
                name = pt.stem.replace('_results', '')
                d = torch.load(pt, map_location='cpu', weights_only=False)
                data[g][m][name] = _to_py(d.get('best_psnr'))
    return data


def build_matrices(data):
    """Compute the two M×G matrices."""
    M, G = len(METHODS), len(GROUPS)
    avg_psnr = np.full((M, G), np.nan)
    avg_dpsnr = np.full((M, G), np.nan)

    for gi, g in enumerate(GROUPS):
        # union of image names across methods (in case some method failed)
        names = sorted({n for m in METHODS for n in data[g][m]})
        if not names:
            continue
        # Per-image max across methods → reference for ΔPSNR
        per_image_max = {}
        for n in names:
            vals = [data[g][m][n] for m in METHODS
                    if n in data[g][m] and data[g][m][n] is not None]
            per_image_max[n] = max(vals) if vals else None

        for mi, m in enumerate(METHODS):
            psnrs = [v for v in data[g][m].values() if v is not None]
            if psnrs:
                avg_psnr[mi, gi] = np.mean(psnrs)

            deltas = [data[g][m][n] - per_image_max[n]
                      for n in names
                      if n in data[g][m]
                      and data[g][m][n] is not None
                      and per_image_max[n] is not None]
            if deltas:
                avg_dpsnr[mi, gi] = np.mean(deltas)

    return avg_psnr, avg_dpsnr


def heatmap(matrix, row_labels, col_labels, title, value_fmt,
            cmap, out_path, vmin=None, vmax=None, cbar_label=''):
    M, G = matrix.shape
    fig, ax = plt.subplots(figsize=(1.2 * G + 2.2, 0.5 * M + 1.5), dpi=160)

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(G), labels=col_labels)
    ax.set_yticks(range(M), labels=row_labels)
    ax.set_title(title, fontsize=11, pad=10)

    # Cell annotations
    if vmin is None or vmax is None:
        norm_min, norm_max = np.nanmin(matrix), np.nanmax(matrix)
    else:
        norm_min, norm_max = vmin, vmax
    span = max(norm_max - norm_min, 1e-9)
    for i in range(M):
        for j in range(G):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center',
                        color='gray', fontsize=9)
                continue
            # White text on dark cells, black on light cells
            t = (v - norm_min) / span
            if 'Reds' in cmap or 'Blues' in cmap or 'YlOrRd' in cmap or 'magma' in cmap:
                color = 'white' if t > 0.55 else 'black'
            else:
                color = 'white' if t < 0.4 else 'black'
            ax.text(j, i, value_fmt.format(v),
                    ha='center', va='center', color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('[load] reading all *_results.pt files...')
    data = collect()
    avg_psnr, avg_dpsnr = build_matrices(data)

    # Sort rows by overall average PSNR (descending) so strong methods sit at top.
    overall = np.nanmean(avg_psnr, axis=1)
    order = np.argsort(-overall)
    methods_sorted = [METHODS[i] for i in order]
    avg_psnr_s  = avg_psnr[order]
    avg_dpsnr_s = avg_dpsnr[order]

    col_labels = ['low', 'mid', 'high']

    # --- Heatmap 1: absolute PSNR ---
    heatmap(
        avg_psnr_s,
        methods_sorted, col_labels,
        title='Image Fitting — avg best PSNR per (method, frequency band)',
        value_fmt='{:.2f}',
        cmap='viridis',
        out_path=OUT_DIR / 'method_group_avg_psnr_heatmap.png',
        cbar_label='avg best PSNR (dB)',
    )

    # --- Heatmap 2: ΔPSNR vs best method on same image ---
    # vmax pinned to 0 (best method), vmin = floor of observed
    floor = np.nanmin(avg_dpsnr_s)
    heatmap(
        avg_dpsnr_s,
        methods_sorted, col_labels,
        title='Image Fitting — avg ΔPSNR vs best method on same image',
        value_fmt='{:+.2f}',
        cmap='Reds_r',                     # dark red = far below best
        out_path=OUT_DIR / 'method_group_delta_psnr_heatmap.png',
        vmin=floor, vmax=0,
        cbar_label='ΔPSNR vs best (dB, ≤0)',
    )

    # Print compact tables to stdout for quick inspection
    print('\n=== avg best PSNR ===')
    print(f"{'method':<10}  " + '  '.join(f'{c:>6}' for c in col_labels))
    for mi, m in enumerate(methods_sorted):
        row = '  '.join(f'{avg_psnr_s[mi, gi]:>6.2f}' for gi in range(3))
        print(f'{m:<10}  {row}')

    print('\n=== avg ΔPSNR vs best ===')
    print(f"{'method':<10}  " + '  '.join(f'{c:>7}' for c in col_labels))
    for mi, m in enumerate(methods_sorted):
        row = '  '.join(f'{avg_dpsnr_s[mi, gi]:>+7.2f}' for gi in range(3))
        print(f'{m:<10}  {row}')


if __name__ == '__main__':
    main()
