"""
Generate two summary figures for group meeting:
  1. results/figures/summary_psnr.png  — bar chart of PSNR across all tasks
  2. results/figures/startarget_recon.png — GT vs all-method reconstructions
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# 2d_startarget from results.csv (10k iter, searched hyperparams)
import csv
def load_results_csv(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row['method']] = float(row['psnr'])
    return out

startarget_psnr = load_results_csv(
    os.path.join(ROOT, 'results', '2d_startarget', 'results.csv'))

# Image tasks from search JSONs
def load_search_json(path):
    with open(path) as f:
        d = json.load(f)
    return {m: v['best_psnr'] for m, v in d.items()}

search_base = os.path.join(ROOT, 'benchmark', 'search_hyper')
img_fit   = load_search_json(os.path.join(search_base, 'div2k_img2_best_params.json'))
denoising = load_search_json(os.path.join(search_base, 'den_div2k_img2_g0p10_best_params.json'))
inpainting= load_search_json(os.path.join(search_base, 'inp_div2k_img127_r0p20_best_params.json'))
sr_x4     = load_search_json(os.path.join(search_base, 'sr_div2k_img2_x4_best_params.json'))

# Method display order (best to worst on startarget)
METHOD_ORDER = ['wf', 'gf', 'finer', 'siren', 'staf', 'wire', 'gauss',
                'sl2a', 'incode', 'cosmo', 'pemlp', 'relu']

COLORS = {
    'siren': '#4C72B0', 'wire': '#DD8452', 'gauss': '#55A868',
    'finer': '#C44E52', 'gf':   '#8172B3', 'wf':    '#937860',
    'staf':  '#DA8BC3', 'pemlp':'#8C8C8C', 'incode':'#CCB974',
    'sl2a':  '#64B5CD', 'cosmo':'#2ecc71', 'relu':  '#bdc3c7',
}

# ---------------------------------------------------------------------------
# Figure 1: Summary PSNR bar chart (4 tasks + synthetic)
# ---------------------------------------------------------------------------

tasks = [
    ('2D Star Target\n(synthetic, 10k iter)', startarget_psnr),
    ('Image Fitting\n(div2k img2)',           img_fit),
    ('Denoising σ=0.1\n(div2k img2)',         denoising),
    ('Inpainting 20%\n(div2k img127)',         inpainting),
    ('Super-Res ×4\n(div2k img2)',             sr_x4),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('INR Benchmark — PSNR by Task', fontsize=14, fontweight='bold', y=1.01)

for ax, (title, data) in zip(axes, tasks):
    methods = [m for m in METHOD_ORDER if m in data]
    psnrs   = [data[m] for m in methods]
    colors  = [COLORS.get(m, '#999') for m in methods]

    bars = ax.barh(methods[::-1], psnrs[::-1], color=colors[::-1], edgecolor='white', height=0.7)
    for bar, val in zip(bars, psnrs[::-1]):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', ha='left', fontsize=7.5)

    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('PSNR (dB)', fontsize=8)
    xmax = max(psnrs) * 1.12
    xmin = max(0, min(psnrs) - 3)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'summary_psnr.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'[saved] {out1}')

# ---------------------------------------------------------------------------
# Figure 2: GT vs reconstructions for 2d_startarget
# ---------------------------------------------------------------------------

recon_base = os.path.join(ROOT, 'results', '2d_startarget')
methods_with_recon = [m for m in METHOD_ORDER
                      if os.path.exists(os.path.join(recon_base, m, 'seed1234', 'best_recon.png'))]

gt_path = os.path.join(recon_base, methods_with_recon[0], 'seed1234', 'gt.png')
gt_img  = np.array(Image.open(gt_path).convert('L'))

n = len(methods_with_recon)
ncols = 4
nrows = (n + 1 + ncols - 1) // ncols  # +1 for GT

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.2))
fig2.suptitle('2D Star Target — Ground Truth vs Reconstructions (10k iter)',
              fontsize=13, fontweight='bold', y=1.01)
axes2 = axes2.flatten()

# GT panel
axes2[0].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
axes2[0].set_title('Ground Truth', fontsize=10, fontweight='bold')
axes2[0].axis('off')

# Reconstruction panels
for i, method in enumerate(methods_with_recon):
    ax = axes2[i + 1]
    recon_path = os.path.join(recon_base, method, 'seed1234', 'best_recon.png')
    recon = np.array(Image.open(recon_path).convert('L'))

    # Difference map (amplified)
    diff = np.abs(gt_img.astype(float) - recon.astype(float))

    psnr = startarget_psnr.get(method, float('nan'))
    ax.imshow(recon, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'{method.upper()}\n{psnr:.2f} dB', fontsize=9, fontweight='bold',
                 color=COLORS.get(method, 'black'))
    ax.axis('off')

# Hide unused panels
for j in range(len(methods_with_recon) + 1, len(axes2)):
    axes2[j].axis('off')

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'startarget_recon.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'[saved] {out2}')

# ---------------------------------------------------------------------------
# Figure 3: GT | Recon | Difference side-by-side for selected methods
# ---------------------------------------------------------------------------

selected = [m for m in METHOD_ORDER if os.path.exists(
    os.path.join(recon_base, m, 'seed1234', 'best_recon.png'))]

fig3, axes3 = plt.subplots(len(selected), 3,
                            figsize=(10, len(selected) * 2.8))
fig3.suptitle('2D Star Target — GT / Reconstruction / |Difference|×5',
              fontsize=12, fontweight='bold', y=1.01)

col_labels = ['Ground Truth', 'Best Recon', '|Difference| ×5']
for col, label in enumerate(col_labels):
    axes3[0, col].set_title(label, fontsize=10, fontweight='bold')

for row, method in enumerate(selected):
    recon_path = os.path.join(recon_base, method, 'seed1234', 'best_recon.png')
    recon = np.array(Image.open(recon_path).convert('L'))
    diff  = np.clip(np.abs(gt_img.astype(float) - recon.astype(float)) * 5, 0, 255).astype(np.uint8)
    psnr  = startarget_psnr.get(method, float('nan'))

    for col, img in enumerate([gt_img, recon, diff]):
        ax = axes3[row, col]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        if col == 1:
            ax.text(0.5, -0.04, f'{method.upper()}  {psnr:.2f} dB',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=9, fontweight='bold',
                    color=COLORS.get(method, 'black'))

plt.tight_layout()
out3 = os.path.join(OUT_DIR, 'startarget_diff.png')
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f'[saved] {out3}')
