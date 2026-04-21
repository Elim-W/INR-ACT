"""
Collect all result .pt files from a results directory and aggregate metrics.
"""

import os
import glob
import torch
import json
import numpy as np


def collect(results_dir, task='image_fitting'):
    """
    Walk results_dir/<task>/<method>/ and aggregate per-image metrics.

    Expected file layout:
        results/image_fitting/siren/kodak01_results.pt
        results/image_fitting/wire/kodak01_results.pt
        ...

    Each .pt file is a dict as returned by tasks/image_fitting.run().

    Returns:
        dict: { method_name: { image_name: {psnr, ssim, time} } }
    """
    task_dir = os.path.join(results_dir, task)
    if not os.path.isdir(task_dir):
        print(f"[collect] No results directory found at '{task_dir}'")
        return {}

    summary = {}
    for method_dir in sorted(os.listdir(task_dir)):
        method_path = os.path.join(task_dir, method_dir)
        if not os.path.isdir(method_path):
            continue
        summary[method_dir] = {}
        for pt_file in sorted(glob.glob(os.path.join(method_path, '*.pt'))):
            result = torch.load(pt_file, map_location='cpu',
                                weights_only=False)
            name = result.get('name', os.path.splitext(
                os.path.basename(pt_file))[0])
            summary[method_dir][name] = {
                'psnr':   result.get('final_psnr', float('nan')),
                'ssim':   result.get('final_ssim', float('nan')),
                'time_s': result.get('total_time_s', float('nan')),
            }

    return summary


def print_summary(summary):
    for method, images in summary.items():
        psnrs = [v['psnr'] for v in images.values()]
        ssims = [v['ssim'] for v in images.values()]
        print(f"  {method:12s}  PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f}"
              f"  SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}"
              f"  ({len(images)} images)")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')
    p.add_argument('--task', default='image_fitting')
    p.add_argument('--out_json', default=None)
    args = p.parse_args()

    summary = collect(args.results_dir, args.task)
    print_summary(summary)

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {args.out_json}")
