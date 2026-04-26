"""
CT reconstruction runner.

Standalone entry for image_ct_reconstruction. Single image per run, supervises
the INR through `proj` parallel-beam Radon projections of the GT image.

Usage:
    python benchmark/run_ct.py --config configs/experiments/image_ct_reconstruction/siren.yaml
    python benchmark/run_ct.py --config <yaml> --override training.proj=100 training.lr=2e-4
"""

import os
import sys
import time
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark._runner_common import (
    parse_cli, load_config, apply_overrides,
    get_device, build_model, make_save_dir,
)
from benchmark.datasets import get_dataset
from benchmark.tasks import image_ct_reconstruction


def _print_summary(method, dataset_name, all_results, elapsed):
    print(f"\n{'='*60}")
    print(f"  Method : {method}")
    print(f"  Task   : image_ct_reconstruction")
    print(f"  Dataset: {dataset_name}  ({len(all_results)} images)")
    if all_results:
        psnrs = [r['final_psnr'] for r in all_results]
        ssims = [r['final_ssim'] for r in all_results]
        bpsnrs = [r['best_psnr']  for r in all_results]
        print(f"  final  PSNR : {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
        print(f"  best   PSNR : {np.mean(bpsnrs):.2f} ± {np.std(bpsnrs):.2f} dB")
        print(f"  final  SSIM : {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"  Total time: {elapsed:.1f}s")
    print('=' * 60)


def main():
    args = parse_cli(description='INR CT Reconstruction Runner')
    cfg = apply_overrides(load_config(args.config), args.override)

    if cfg.get('task') and cfg['task'] != 'image_ct_reconstruction':
        raise ValueError(
            f"This runner is for image_ct_reconstruction only; "
            f"got task='{cfg['task']}' in the config")
    cfg['task'] = 'image_ct_reconstruction'

    device = get_device(cfg)
    save_dir_base, out_cfg = make_save_dir(cfg, 'image_ct_reconstruction')

    print(f"[run_ct] method={cfg['method']}  device={device}  proj={cfg['training'].get('proj', 150)}")

    dataset_name = cfg.get('dataset', 'kodak')
    data_root = cfg.get('data_root', os.path.join('data', dataset_name))
    indices   = cfg.get('image_indices', None)

    ds_kw = {'indices': indices, 'normalize': True}
    if dataset_name == 'div2k':
        if cfg.get('downscale') and cfg['downscale'] > 1:
            ds_kw['downscale'] = cfg['downscale']
        elif cfg.get('max_size'):
            ds_kw['max_size'] = cfg['max_size']
    dataset = get_dataset(dataset_name, data_root, **ds_kw)
    print(f"[run_ct] dataset={dataset_name}  n_images={len(dataset)}")

    all_results = []
    total_t0 = time.time()

    for coords, pixels, meta in dataset.iter_images():
        print(f"\n[run_ct] === {meta['name']} ({meta['H']}×{meta['W']}) ===")

        model = build_model(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        img_save_dir = (os.path.join(save_dir_base, meta['name'])
                        if out_cfg.get('save_images', True) else None)

        result = image_ct_reconstruction.run(
            model=model, coords=coords, pixels=pixels, meta=meta,
            cfg=cfg, device=device, save_dir=img_save_dir,
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
