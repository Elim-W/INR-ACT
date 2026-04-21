"""
3D INR benchmark entry point.

Handles 3D shape tasks that share the dataset interface
    dataset.iter_shapes() → (coords, targets, meta)
    coords:  (N, 3) in [-1, 1]^3
    targets: (N, 1) — e.g. binary occupancy

Supported tasks:
    shape_occupancy
Not yet implemented (placeholders):
    nerf, sdf

For 2D image tasks see run_experiment.py.

Usage:
    python benchmark/run_experiment_3d.py --config configs/experiments/shape_occupancy/siren.yaml
    python benchmark/run_experiment_3d.py --config configs/experiments/shape_occupancy/siren.yaml \\
        --override mesh_ids=[bunny] dataset_kwargs.grid_res=32 training.num_epochs=100
"""

import os
import sys
import time
import numpy as np
import torch

# Project root on sys.path so `python benchmark/run_experiment_3d.py` works from any cwd.
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

_3D_SHAPE_TASKS = {'shape_occupancy'}
_3D_TASKS_TODO  = {'nerf', 'sdf'}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(method, dataset_name, all_results, elapsed):
    print(f"\n{'='*60}")
    print(f"  Method : {method}")
    print(f"  Dataset: {dataset_name}  ({len(all_results)} shapes)")
    if all_results:
        ious = [r['final_iou'] for r in all_results]
        accs = [r['final_acc'] for r in all_results]
        print(f"  IoU    : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
        print(f"  Acc    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Total time: {elapsed:.1f}s")
    print('='*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_cli(description='3D INR Benchmark')
    cfg = apply_overrides(load_config(args.config), args.override)

    task_name = cfg['task']
    if task_name in _3D_TASKS_TODO:
        raise NotImplementedError(
            f"Task '{task_name}' is not yet implemented. "
            "When you add it, register its dataset interface in this file.")
    if task_name not in _3D_SHAPE_TASKS:
        raise ValueError(
            f"'{task_name}' is not a 3D shape task.  Supported: "
            f"{sorted(_3D_SHAPE_TASKS)}.  Not yet implemented: "
            f"{sorted(_3D_TASKS_TODO)}.\n"
            f"For 2D image tasks use: python benchmark/run_experiment.py")

    device = get_device(cfg)
    task_mod = get_task(task_name)
    save_dir_base, out_cfg = make_save_dir(cfg, task_name)

    print(f"[run_experiment_3d] task={task_name}  "
          f"method={cfg['method']}  device={device}")

    dataset_name = cfg.get('dataset', 'stanford_3d')
    data_root = cfg.get('data_root', os.path.join('data', dataset_name))
    ds_kwargs = dict(cfg.get('dataset_kwargs', {}))
    if 'mesh_ids' in cfg:
        ds_kwargs['mesh_ids'] = cfg['mesh_ids']
    dataset = get_dataset(dataset_name, data_root, **ds_kwargs)
    print(f"[run_experiment_3d] dataset={dataset_name}  n_shapes={len(dataset)}")

    all_results = []
    total_t0 = time.time()

    for coords, occupancy, meta in dataset.iter_shapes():
        print(f"\n[run_experiment_3d] === {meta['name']} "
              f"(grid {meta['H']}³, {meta['n_points']:,} points) ===")

        model = build_model(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        shape_save_dir = os.path.join(save_dir_base, meta['name']) \
            if out_cfg.get('save_images', True) else None

        result = task_mod.run(
            model=model,
            coords=coords,
            targets=occupancy,
            meta=meta,
            cfg=cfg,
            device=device,
            save_dir=shape_save_dir,
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
