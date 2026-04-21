"""
2D INR benchmark entry point.

Handles 2D tasks that share the dataset interface
    dataset.iter_images() → (coords, pixels, meta)
    coords: (N, 2) in [-1, 1]        pixels: (N, C) in [0, 1]

Supported tasks:
    image_fitting, image_denoising, image_inpainting, image_super_resolution

For 3D shape tasks see run_experiment_3d.py.

Usage:
    python benchmark/run_experiment.py --config configs/experiments/image_fitting/siren.yaml
    python benchmark/run_experiment.py --config configs/experiments/image_fitting/wire.yaml \\
        --override training.lr=5e-3 training.num_epochs=3000
"""

import os
import sys
import time
import numpy as np
import torch

# Project root on sys.path so `python benchmark/run_experiment.py` works from any cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark._runner_common import (
    parse_cli,
    load_config, apply_overrides,
    get_device, build_model, make_save_dir,
)
from benchmark.datasets import get_dataset
from benchmark.tasks import get_task


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

_2D_TASKS = {
    'image_fitting',
    'image_denoising',
    'image_inpainting',
    'image_super_resolution',
}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(method, dataset_name, all_results, elapsed):
    print(f"\n{'='*60}")
    print(f"  Method : {method}")
    print(f"  Dataset: {dataset_name}  ({len(all_results)} images)")
    if all_results:
        psnrs = [r['final_psnr'] for r in all_results]
        ssims = [r['final_ssim'] for r in all_results]
        print(f"  PSNR   : {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
        print(f"  SSIM   : {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"  Total time: {elapsed:.1f}s")
    print('='*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_cli(description='2D INR Benchmark')
    cfg = apply_overrides(load_config(args.config), args.override)

    task_name = cfg['task']
    if task_name not in _2D_TASKS:
        raise ValueError(
            f"'{task_name}' is not a 2D task.  Supported: {sorted(_2D_TASKS)}.\n"
            f"For 3D shape tasks use: python benchmark/run_experiment_3d.py")

    device = get_device(cfg)
    task_mod = get_task(task_name)
    save_dir_base, out_cfg = make_save_dir(cfg, task_name)

    print(f"[run_experiment] task={task_name}  method={cfg['method']}  device={device}")

    dataset_name = cfg.get('dataset', 'kodak')
    data_root = cfg.get('data_root', os.path.join('data', dataset_name))
    indices = cfg.get('image_indices', None)
    dataset = get_dataset(dataset_name, data_root,
                          indices=indices, normalize=True)
    print(f"[run_experiment] dataset={dataset_name}  n_images={len(dataset)}")

    all_results = []
    total_t0 = time.time()

    for coords, pixels, meta in dataset.iter_images():
        print(f"\n[run_experiment] === {meta['name']} "
              f"({meta['H']}×{meta['W']}) ===")

        model = build_model(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        img_save_dir = os.path.join(save_dir_base, meta['name']) \
            if out_cfg.get('save_images', True) else None

        result = task_mod.run(
            model=model,
            coords=coords,
            pixels=pixels,
            meta=meta,
            cfg=cfg,
            device=device,
            save_dir=img_save_dir,
        )
        all_results.append(result)

        if out_cfg.get('save_model', True):
            pt_path = os.path.join(save_dir_base, f"{meta['name']}_results.pt")
            torch.save(dict(result), pt_path)
            print(f"  Saved → {pt_path}")

    _print_summary(cfg['method'], dataset_name, all_results,
                   time.time() - total_t0)


if __name__ == '__main__':
    main()
