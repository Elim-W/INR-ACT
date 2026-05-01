"""
Visual gallery for the Super-Resolution (×8) benchmark.

For each frequency band (low / mid / high) we pick one representative image
and produce a row of cells:
    LR (×8 input, nearest-upscaled to HR canvas)  →  HR GT  →  9 methods

Each cell shows the image at its NATIVE resolution (no resampling); the LR
is the only one upscaled (with nearest-neighbour) so the viewer sees the
8× pixelation directly. Output is a single big PNG.

⚠️ The HR images are ~2040×1300; with 11 cells × 3 rows at native dpi the
   resulting PNG is in the hundreds of MB.  Set --dpi 50 to halve all
   dimensions if your viewer struggles.

Output:
    donghua/visual_gallery_sr.png

Usage:
    python donghua/plot_visual_gallery_sr.py
    python donghua/plot_visual_gallery_sr.py --low 0040 --mid 0209 --high 0182
    python donghua/plot_visual_gallery_sr.py --dpi 50    # half-res output
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

ROOT     = Path('results/image_super_resolution_groupss')
OUT_PATH = Path('donghua/visual_gallery_sr.png')

DEFAULT_PICKS = {'low_5': '0040', 'mid_5': '0209', 'high_5': '0182'}


def _to_py(v):
    if v is None:
        return None
    return v.item() if hasattr(v, 'item') else float(v)


def load_pred(group, method, name):
    """Return (PIL HR reconstruction, best_psnr)."""
    img_path = ROOT / group / method / name / f'{name}_final_HR.png'
    pt_path  = ROOT / group / method / f'{name}_results.pt'
    if not img_path.exists() or not pt_path.exists():
        return None, None
    img = Image.open(img_path).convert('RGB')
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    return img, _to_py(d.get('best_psnr'))


def load_gt_and_lr(group, name):
    """Pick GT and LR from the first method dir that has them."""
    for m in METHODS:
        gt_path = ROOT / group / m / name / f'{name}_gt_HR.png'
        lr_path = ROOT / group / m / name / f'{name}_input_LR.png'
        if gt_path.exists() and lr_path.exists():
            return Image.open(gt_path).convert('RGB'), \
                   Image.open(lr_path).convert('RGB')
    return None, None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--low',  default=DEFAULT_PICKS['low_5'])
    p.add_argument('--mid',  default=DEFAULT_PICKS['mid_5'])
    p.add_argument('--high', default=DEFAULT_PICKS['high_5'])
    p.add_argument('--out',  default=str(OUT_PATH))
    p.add_argument('--dpi',  type=int, default=100,
                   help='Output dpi.  100 = native pixel ratio (1 image px = '
                        '1 file px).  50 halves output dimensions.')
    return p.parse_args()


def plot_one_group(group, group_label, name, dpi, out_path):
    """Render a single 1×11 row for one (group, image)."""
    gt, lr = load_gt_and_lr(group, name)
    if gt is None:
        print(f'[warn] {group}/{name}: no GT/LR found, skipping')
        return

    canvas_w, canvas_h = gt.size
    n_cols = 2 + len(METHODS)
    cell_w_in = canvas_w / dpi
    cell_h_in = canvas_h / dpi
    title_pad_in = 0.7

    print(f'[{group}] canvas {canvas_w}×{canvas_h} px  →  cell {cell_w_in:.2f}'
          f'×{cell_h_in:.2f} in  @ dpi={dpi}')

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(n_cols * cell_w_in, cell_h_in + title_pad_in),
        dpi=dpi,
    )

    # LR upsampled with NEAREST so the 8× pixelation is visible
    lr_canvas = lr.resize((canvas_w, canvas_h), Image.NEAREST)

    ax = axes[0]
    ax.imshow(np.asarray(lr_canvas), interpolation='none')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'LR (×8 input)\n{lr.size[0]}×{lr.size[1]}  '
                 f'({group_label}, {name})', fontsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2); spine.set_color('#0a4')

    ax = axes[1]
    ax.imshow(np.asarray(gt), interpolation='none')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'GT (HR)\n{gt.size[0]}×{gt.size[1]}', fontsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2); spine.set_color('#222')

    for ci, method in enumerate(METHODS, start=2):
        img, psnr = load_pred(group, method, name)
        ax = axes[ci]
        if img is None:
            ax.text(0.5, 0.5, 'missing', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xticks([]); ax.set_yticks([])
            continue
        if img.size != (canvas_w, canvas_h):
            img = img.resize((canvas_w, canvas_h), Image.BICUBIC)
        ax.imshow(np.asarray(img), interpolation='none')
        ax.set_xticks([]); ax.set_yticks([])
        psnr_label = 'n/a' if psnr is None else f'{psnr:.2f} dB'
        ax.set_title(f'{method}\n{psnr_label}', fontsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    fig.suptitle(f'Super-Resolution (×8) — {group_label} ({name})',
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    args = parse_args()
    picks = {'low_5': args.low, 'mid_5': args.mid, 'high_5': args.high}

    # One file per group
    out_base = Path(args.out)              # e.g. donghua/visual_gallery_sr.png
    stem = out_base.stem                   # 'visual_gallery_sr'
    parent = out_base.parent

    for g, glab in zip(GROUPS, GROUP_LABELS):
        suffix = g.split('_')[0]           # 'low' / 'mid' / 'high'
        out_path = parent / f'{stem}_{suffix}.png'
        plot_one_group(g, glab, picks[g], args.dpi, out_path)


if __name__ == '__main__':
    main()
