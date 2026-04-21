"""
Unified experiment entry point.

Usage:
    python benchmark/run_experiment.py --config configs/experiments/image_fitting/siren.yaml
    python benchmark/run_experiment.py --config configs/experiments/image_fitting/wire.yaml \\
        --override training.lr=5e-3 training.num_epochs=3000
"""

import os
import sys
import argparse
import time
import torch
import yaml

# Make sure the project root is importable regardless of cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark.methods.models import get_INR
from benchmark.datasets import get_dataset
from benchmark.tasks import get_task


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_overrides(cfg, overrides):
    """
    Apply dot-notation overrides, e.g. 'training.lr=1e-3'.
    Supports nested keys and numeric / bool / string coercion.
    """
    for ov in overrides:
        key, _, val_str = ov.partition('=')
        keys = key.split('.')
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # coerce type
        try:
            val = yaml.safe_load(val_str)
        except Exception:
            val = val_str
        node[keys[-1]] = val
    return cfg


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(cfg):
    req = cfg.get('device', 'auto')
    if req == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(req)


# ---------------------------------------------------------------------------
# Build model from config
# ---------------------------------------------------------------------------

def build_model(cfg):
    method = cfg['method']
    mcfg = cfg.get('model', {})
    return get_INR(
        method=method,
        in_features=mcfg.get('in_features', 2),
        hidden_features=mcfg.get('hidden_features', 256),
        hidden_layers=mcfg.get('hidden_layers', 3),
        out_features=mcfg.get('out_features', 3),
        **{k: v for k, v in mcfg.items()
           if k not in ('in_features', 'hidden_features', 'hidden_layers', 'out_features')},
    )


# ---------------------------------------------------------------------------
# Task families
# 2D tasks share the same data interface: dataset.iter_images() →
#   (coords, pixels, meta).  Add new 2D tasks here as they are implemented.
# 3D tasks (nerf, sdf, …) will need their own data interface — refactor
#   run_experiment.py when those are added.
# ---------------------------------------------------------------------------

_2D_TASKS = {
    'image_fitting',
    'image_denoising',
    'image_inpainting',
    'image_super_resolution',
}
_3D_SHAPE_TASKS = {'shape_occupancy'}
_3D_TASKS = {'nerf', 'sdf'}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='INR Benchmark')
    parser.add_argument('--config', required=True,
                        help='Path to YAML config file')
    parser.add_argument('--override', nargs='*', default=[],
                        metavar='KEY=VAL',
                        help='Override any config value, e.g. training.lr=1e-3 '
                             'image_indices=[1,2,3]')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    task_name = cfg['task']
    device = get_device(cfg)
    print(f"[run_experiment] task={task_name}  method={cfg['method']}  device={device}")

    # ---- Task module ----
    task_mod = get_task(task_name)

    # ---- Output directory ----
    out_cfg = cfg.get('output', {})
    save_dir_base = out_cfg.get(
        'save_dir',
        os.path.join('results', task_name, cfg['method']))
    os.makedirs(save_dir_base, exist_ok=True)

    # ================================================================
    # 2D tasks: image_fitting, image_denoising, image_inpainting,
    #           image_super_resolution
    #   Data interface: dataset.iter_images() → (coords, pixels, meta)
    #   The task module is responsible for any per-task transform
    #   (adding noise, applying a pixel mask, building an LR version, ...).
    # ================================================================
    if task_name in _2D_TASKS:
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
                save_dict = {k: v for k, v in result.items()}
                torch.save(save_dict, pt_path)
                print(f"  Saved → {pt_path}")

        _print_2d_summary(cfg['method'], dataset_name, all_results,
                          time.time() - total_t0)

    # ================================================================
    # 3D shape tasks: shape_occupancy
    #   Data interface: dataset.iter_shapes() → (coords, occupancy, meta)
    #   coords are (N, 3) in [-1, 1]^3; occupancy is (N, 1) in {0, 1}.
    # ================================================================
    elif task_name in _3D_SHAPE_TASKS:
        dataset_name = cfg.get('dataset', 'stanford_3d')
        data_root = cfg.get('data_root', os.path.join('data', dataset_name))
        ds_kwargs = dict(cfg.get('dataset_kwargs', {}))
        # Allow either `mesh_ids` (preferred) or `image_indices` (legacy alias)
        if 'mesh_ids' in cfg:
            ds_kwargs['mesh_ids'] = cfg['mesh_ids']
        dataset = get_dataset(dataset_name, data_root, **ds_kwargs)
        print(f"[run_experiment] dataset={dataset_name}  n_shapes={len(dataset)}")

        all_results = []
        total_t0 = time.time()

        for coords, occupancy, meta in dataset.iter_shapes():
            print(f"\n[run_experiment] === {meta['name']} "
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

        _print_shape_summary(cfg['method'], dataset_name, all_results,
                             time.time() - total_t0)

    # ================================================================
    # Other 3D tasks: nerf, sdf  (not yet implemented)
    # ================================================================
    elif task_name in _3D_TASKS:
        raise NotImplementedError(
            f"Task '{task_name}' is not yet implemented. "
            "When adding 3D tasks, implement the data interface here.")

    else:
        raise ValueError(f"Unknown task '{task_name}'. "
                         f"2D: {_2D_TASKS}  3D: {_3D_TASKS}")


def _print_2d_summary(method, dataset_name, all_results, elapsed):
    import numpy as np
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


def _print_shape_summary(method, dataset_name, all_results, elapsed):
    import numpy as np
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


if __name__ == '__main__':
    main()
