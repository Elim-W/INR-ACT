"""
Visual gallery for the Inpainting (sampling_ratio=0.2) benchmark.

For each frequency band we pick one representative image and produce a row:
    GT  →  Masked input (80% pixels dropped)  →  9 methods' reconstructions

Each cell is rendered at the native reconstruction resolution (1/4 of the
DIV2K original, since inpaint runs with downscale=4) — no resampling.

Output:
    donghua/visual_gallery_inpaint.png

Usage:
    python donghua/plot_visual_gallery_inpaint.py
    python donghua/plot_visual_gallery_inpaint.py --low 0040 --mid 0209 --high 0182
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

ROOT     = Path('results/image_inpaint_groups')
DATA_DIR = Path('data')
OUT_PATH = Path('donghua/visual_gallery_inpaint.png')

DEFAULT_PICKS = {'low_5': '0040', 'mid_5': '0209', 'high_5': '0182'}


def _to_py(v):
    if v is None:
        return None
    return v.item() if hasattr(v, 'item') else float(v)


def load_pred(group, method, name):
    """Return (PIL reconstruction, best_psnr)."""
    img_path = ROOT / group / method / name / f'{name}_final.png'
    pt_path  = ROOT / group / method / f'{name}_results.pt'
    if not img_path.exists() or not pt_path.exists():
        return None, None
    img = Image.open(img_path).convert('RGB')
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    return img, _to_py(d.get('best_psnr'))


def load_masked(group, name):
    """Pick masked input image from any method dir that has it."""
    for m in METHODS:
        p = ROOT / group / m / name / f'{name}_masked.png'
        if p.exists():
            return Image.open(p).convert('RGB')
    return None


def load_gt(group, name, target_size):
    """Original DIV2K image, downscaled to match reconstruction resolution."""
    full = Image.open(DATA_DIR / group / f'{name}.png').convert('RGB')
    return full.resize(target_size, Image.BICUBIC)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--low',  default=DEFAULT_PICKS['low_5'])
    p.add_argument('--mid',  default=DEFAULT_PICKS['mid_5'])
    p.add_argument('--high', default=DEFAULT_PICKS['high_5'])
    p.add_argument('--out',  default=str(OUT_PATH))
    p.add_argument('--dpi',  type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    picks = {'low_5': args.low, 'mid_5': args.mid, 'high_5': args.high}

    col_labels = ['GT', 'Masked (20% kept)'] + METHODS
    n_cols = len(col_labels)
    n_rows = len(GROUPS)

    # Use the first available reconstruction's size as canvas
    canvas_w = canvas_h = None
    for g in GROUPS:
        ref, _ = load_pred(g, METHODS[0], picks[g])
        if ref is not None:
            canvas_w, canvas_h = ref.size
            break
    if canvas_w is None:
        raise SystemExit('[error] no reconstructions found — nothing to plot')

    cell_w_in = canvas_w / args.dpi
    cell_h_in = canvas_h / args.dpi
    title_pad_in = 0.5

    print(f'[layout] canvas {canvas_w}×{canvas_h} px  →  cell {cell_w_in:.2f}'
          f'×{cell_h_in:.2f} in  @ dpi={args.dpi}')
    print(f'[layout] figure: {n_cols * cell_w_in:.1f} × '
          f'{n_rows * (cell_h_in + title_pad_in):.1f} inches')

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w_in,
                 n_rows * (cell_h_in + title_pad_in)),
        dpi=args.dpi,
    )
    if n_rows == 1:
        axes = axes[None, :]

    for ri, (g, glab) in enumerate(zip(GROUPS, GROUP_LABELS)):
        name = picks[g]
        ref, _ = load_pred(g, METHODS[0], name)
        if ref is None:
            print(f'[warn] {g}/{name}: no reconstruction found, skipping row')
            for ax in axes[ri]:
                ax.axis('off')
            continue
        target_size = (canvas_w, canvas_h)

        # Col 0: GT
        gt = load_gt(g, name, target_size)
        ax = axes[ri, 0]
        ax.imshow(np.asarray(gt), interpolation='none')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'GT  ({glab}, {name})', fontsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#222')

        # Col 1: Masked input
        masked = load_masked(g, name)
        ax = axes[ri, 1]
        if masked is None:
            ax.text(0.5, 0.5, 'no masked',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='red')
            ax.set_xticks([]); ax.set_yticks([])
        else:
            if masked.size != target_size:
                masked = masked.resize(target_size, Image.NEAREST)
            ax.imshow(np.asarray(masked), interpolation='none')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title('Masked input\n(20% pixels kept)', fontsize=10)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#0a4')

        # Cols 2..: methods
        for ci, method in enumerate(METHODS, start=2):
            img, psnr = load_pred(g, method, name)
            ax = axes[ri, ci]
            if img is None:
                ax.text(0.5, 0.5, 'missing',
                        ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='red')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            if img.size != target_size:
                img = img.resize(target_size, Image.BICUBIC)
            ax.imshow(np.asarray(img), interpolation='none')
            ax.set_xticks([]); ax.set_yticks([])
            psnr_label = 'n/a' if psnr is None else f'{psnr:.2f} dB'
            ax.set_title(f'{method}\n{psnr_label}', fontsize=10)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    fig.suptitle('Inpainting (sampling_ratio=0.2) — visual gallery '
                 '(GT + masked input + 9 methods, one image per band)',
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == '__main__':
    main()
