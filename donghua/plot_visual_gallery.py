"""
Visual gallery for the fitting benchmark: one representative image per
frequency band, ground truth + 9 methods' reconstructions side by side,
each cell labelled with its best PSNR.

Output:
    donghua/visual_gallery_fitting.png

Usage:
    python donghua/plot_visual_gallery.py
    python donghua/plot_visual_gallery.py \
        --low 0040 --mid 0209 --high 0182    # pick specific images
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


GROUPS  = ['low_5', 'mid_5', 'high_5']
GROUP_LABELS = ['low freq', 'mid freq', 'high freq']
METHODS = ['siren', 'wire', 'gauss', 'finer', 'gf', 'wf', 'staf', 'relu', 'sl2a']

ROOT     = Path('results/image_fitting_groupss')
DATA_DIR = Path('data')
OUT_PATH = Path('donghua/visual_gallery_fitting.png')

# Default representative image per group
DEFAULT_PICKS = {'low_5': '0040', 'mid_5': '0209', 'high_5': '0182'}


def _to_py(v):
    if v is None:
        return None
    return v.item() if hasattr(v, 'item') else float(v)


def load_pred(group, method, name):
    """Return (PIL image, best_psnr) for one (group, method, image)."""
    img_path = ROOT / group / method / name / f'{name}_final.png'
    pt_path  = ROOT / group / method / f'{name}_results.pt'
    if not img_path.exists() or not pt_path.exists():
        return None, None
    img = Image.open(img_path).convert('RGB')
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    psnr = _to_py(d.get('best_psnr'))
    return img, psnr


def load_gt(group, name, target_size):
    """Load DIV2K original and downscale to match reconstruction (1/4 res)."""
    full = Image.open(DATA_DIR / group / f'{name}.png').convert('RGB')
    return full.resize(target_size, Image.BICUBIC)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--low',  default=DEFAULT_PICKS['low_5'])
    p.add_argument('--mid',  default=DEFAULT_PICKS['mid_5'])
    p.add_argument('--high', default=DEFAULT_PICKS['high_5'])
    p.add_argument('--out',  default=str(OUT_PATH))
    return p.parse_args()


def main():
    args = parse_args()
    picks = {'low_5': args.low, 'mid_5': args.mid, 'high_5': args.high}

    # 3 rows (groups) × (1 GT + 9 methods) = 10 cols
    n_cols = 1 + len(METHODS)
    n_rows = len(GROUPS)

    # Figure out the actual reconstruction resolution from the FIRST available
    # (group, method, image) cell.  Each cell is then sized in inches so that
    # at the chosen dpi, 1 cell pixel = 1 image pixel — no downsampling.
    cell_w_px = cell_h_px = None
    for g in GROUPS:
        ref_img, _ = load_pred(g, METHODS[0], picks[g])
        if ref_img is not None:
            cell_w_px, cell_h_px = ref_img.size       # PIL: (W, H)
            break
    if cell_w_px is None:
        raise SystemExit('[error] no reconstructions found — nothing to plot')

    DPI = 100
    cell_w_in = cell_w_px / DPI
    cell_h_in = cell_h_px / DPI
    title_pad_in = 0.5     # extra room above each cell for the title text

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w_in,
                 n_rows * (cell_h_in + title_pad_in)),
        dpi=DPI,
    )
    if n_rows == 1:
        axes = axes[None, :]

    for ri, (g, glab) in enumerate(zip(GROUPS, GROUP_LABELS)):
        name = picks[g]
        # Find target size from any one method's reconstruction
        ref_img, _ = load_pred(g, METHODS[0], name)
        if ref_img is None:
            print(f"[warn] no reconstruction found for {g}/{METHODS[0]}/{name}")
            continue
        target_size = ref_img.size  # (W, H) for PIL

        # Column 0: GT
        gt = load_gt(g, name, target_size)
        ax = axes[ri, 0]
        ax.imshow(np.asarray(gt), interpolation='none')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'GT  ({glab})\n{name}.png', fontsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#222')

        # Row label on the very left (outside the GT cell)
        if ri == 0:
            pass
        ax.set_ylabel(f'{glab}\n{name}', fontsize=10, rotation=0,
                      ha='right', va='center', labelpad=20)

        # Cols 1..: methods
        for ci, method in enumerate(METHODS, start=1):
            img, psnr = load_pred(g, method, name)
            ax = axes[ri, ci]
            if img is None:
                ax.text(0.5, 0.5, 'missing',
                        ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='red')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.imshow(np.asarray(img), interpolation='none')
            ax.set_xticks([]); ax.set_yticks([])
            psnr_label = 'n/a' if psnr is None else f'{psnr:.2f} dB'
            ax.set_title(f'{method}\n{psnr_label}', fontsize=9)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    fig.suptitle('Image Fitting — visual gallery (one image per frequency band)',
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == '__main__':
    main()
