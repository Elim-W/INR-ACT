"""
3D INR benchmark entry point.

Two data interfaces are dispatched from this script:

    shape_occupancy
        dataset.iter_shapes() → (coords[N,3], occupancy[N,1], meta)
        metric: IoU / Acc

    nerf
        blender_nerf.build_nerf_dataloaders(cfg, device) →
            (scene_name, opt, train_loader, val_loader) per scene
        metric: val PSNR
        Requires torch-ngp's 4 CUDA extensions to be compiled on a GPU
        node (see scripts/nerf/compile_extensions.sh).

For 2D image tasks see run_experiment.py.

Usage:
    python benchmark/run_experiment_3d.py --config configs/experiments/shape_occupancy/siren.yaml
    python benchmark/run_experiment_3d.py --config configs/experiments/nerf/siren.yaml
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
_NERF_TASKS     = {'nerf'}
_3D_TASKS_TODO  = {'sdf'}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _print_shape_summary(method, dataset_name, all_results, elapsed):
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


def _print_nerf_summary(method, dataset_name, all_results, elapsed):
    print(f"\n{'='*60}")
    print(f"  Method : {method}")
    print(f"  Dataset: {dataset_name}  ({len(all_results)} scenes)")
    if all_results:
        psnrs = [r['final_psnr'] for r in all_results
                 if r['final_psnr'] is not None and np.isfinite(r['final_psnr'])]
        if psnrs:
            print(f"  PSNR   : {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    print(f"  Total time: {elapsed:.1f}s")
    print('='*60)


# ---------------------------------------------------------------------------
# Dispatch: shape_occupancy
# ---------------------------------------------------------------------------

def _run_shape_task(cfg, task_mod, device, save_dir_base, out_cfg):
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
            model=model, coords=coords, targets=occupancy, meta=meta,
            cfg=cfg, device=device, save_dir=shape_save_dir,
        )
        all_results.append(result)

        if out_cfg.get('save_model', True):
            pt_path = os.path.join(save_dir_base, f"{meta['name']}_results.pt")
            torch.save(dict(result), pt_path)
            print(f"  Saved → {pt_path}")

    _print_shape_summary(cfg['method'], dataset_name, all_results,
                         time.time() - total_t0)


# ---------------------------------------------------------------------------
# Dispatch: nerf
# ---------------------------------------------------------------------------

def _run_nerf_task(cfg, task_mod, device, save_dir_base, out_cfg):
    from benchmark.datasets.blender_nerf import build_nerf_dataloaders
    from benchmark.methods.nerf_networks import build_nerf_network

    dataset_name = cfg.get('dataset', 'blender_nerf')
    print(f"[run_experiment_3d] dataset={dataset_name}")

    all_results = []
    total_t0 = time.time()

    for scene, opt, train_loader, val_loader in build_nerf_dataloaders(cfg, device):
        print(f"\n[run_experiment_3d] === scene={scene} ===")

        # Build NeRF network via the parameterised factory (activation
        # chosen by cfg['method'], other knobs from cfg['model']).
        mcfg = dict(cfg.get('model', {}))
        model = build_nerf_network(
            activation=cfg['method'],
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            **mcfg,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        scene_save_dir = os.path.join(save_dir_base, scene)
        result = task_mod.run(
            model=model, opt=opt,
            train_loader=train_loader, val_loader=val_loader,
            scene_name=scene, cfg=cfg, device=device,
            save_dir=scene_save_dir,
        )
        all_results.append(result)

        if out_cfg.get('save_model', True):
            # torch-ngp already saves checkpoints in workspace/;
            # we persist just the metrics summary here.
            pt_path = os.path.join(save_dir_base, f"{scene}_results.pt")
            torch.save({k: v for k, v in result.items() if k != 'model_state'},
                       pt_path)
            print(f"  Saved → {pt_path}")

    _print_nerf_summary(cfg['method'], dataset_name, all_results,
                        time.time() - total_t0)


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
    if task_name not in _3D_SHAPE_TASKS and task_name not in _NERF_TASKS:
        raise ValueError(
            f"'{task_name}' is not a 3D task.  Supported: "
            f"{sorted(_3D_SHAPE_TASKS | _NERF_TASKS)}.  "
            f"Not yet implemented: {sorted(_3D_TASKS_TODO)}.\n"
            f"For 2D image tasks use: python benchmark/run_experiment.py")

    device = get_device(cfg)
    task_mod = get_task(task_name)
    save_dir_base, out_cfg = make_save_dir(cfg, task_name)

    print(f"[run_experiment_3d] task={task_name}  "
          f"method={cfg['method']}  device={device}")

    if task_name in _3D_SHAPE_TASKS:
        _run_shape_task(cfg, task_mod, device, save_dir_base, out_cfg)
    elif task_name in _NERF_TASKS:
        _run_nerf_task(cfg, task_mod, device, save_dir_base, out_cfg)


if __name__ == '__main__':
    main()
