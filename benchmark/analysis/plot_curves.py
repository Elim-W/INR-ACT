"""
Plot PSNR / SSIM training curves for all methods.
"""

import os
import glob
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt


def load_curves(results_dir, task='image_fitting'):
    """
    Returns { method: { image: {'epochs', 'psnr', 'ssim'} } }
    """
    task_dir = os.path.join(results_dir, task)
    data = {}
    for method_dir in sorted(os.listdir(task_dir)):
        method_path = os.path.join(task_dir, method_dir)
        if not os.path.isdir(method_path):
            continue
        data[method_dir] = {}
        for pt in sorted(glob.glob(os.path.join(method_path, '*.pt'))):
            r = torch.load(pt, map_location='cpu', weights_only=False)
            name = r.get('name', os.path.splitext(os.path.basename(pt))[0])
            data[method_dir][name] = {
                'epochs': r.get('epochs_curve', []),
                'psnr':   r.get('psnr_curve', []),
                'ssim':   r.get('ssim_curve', []),
            }
    return data


def plot_mean_curve(data, metric='psnr', out_path=None, title=None):
    """
    One line per method, showing mean across all images.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, images in sorted(data.items()):
        all_epochs = None
        curves = []
        for img_data in images.values():
            ep = img_data['epochs']
            vals = img_data[metric]
            if len(ep) == 0:
                continue
            if all_epochs is None:
                all_epochs = ep
            curves.append(vals)
        if not curves or all_epochs is None:
            continue
        mean_curve = np.mean(curves, axis=0)
        ax.plot(all_epochs, mean_curve, label=method, linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f'{metric.upper()} vs Epoch (mean over images)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close()


def plot_per_image(data, image_name, metric='psnr', out_path=None):
    """One line per method for a single image."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, images in sorted(data.items()):
        if image_name not in images:
            continue
        img_data = images[image_name]
        ax.plot(img_data['epochs'], img_data[metric], label=method, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{image_name} — {metric.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')
    p.add_argument('--task', default='image_fitting')
    p.add_argument('--metric', default='psnr')
    p.add_argument('--out_dir', default='results/plots')
    args = p.parse_args()

    data = load_curves(args.results_dir, args.task)
    plot_mean_curve(data, args.metric,
                    os.path.join(args.out_dir, f'{args.metric}_mean_curve.png'))
