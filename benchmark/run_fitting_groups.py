"""
Run image_fitting on all images in data/{high_5, mid_5, low_5} for all 11 INR
methods, then compute per-group averages of best/final PSNR & SSIM.

Usage:
    python benchmark/run_fitting_groups.py
    python benchmark/run_fitting_groups.py --groups high_5 mid_5
    python benchmark/run_fitting_groups.py --methods siren wire --max_size 512

Outputs:
    results/image_fitting_groups/
        summary.json          # full per-image + averaged numbers
        summary.md            # readable markdown table
        <group>/<method>/
            <name>_results.pt
            <name>/<name>_ep*.png, <name>_final.png
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark._runner_common import load_config, build_model
from benchmark.datasets import get_dataset
from benchmark.tasks import get_task


GROUPS_DEFAULT = ['high_5', 'mid_5', 'low_5']
METHODS_DEFAULT = ['siren', 'wire', 'gauss', 'finer', 'gf', 'wf',
                   'staf', 'pemlp', 'incode', 'sl2a', 'cosmo']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--groups',  nargs='+', default=GROUPS_DEFAULT)
    p.add_argument('--methods', nargs='+', default=METHODS_DEFAULT)
    p.add_argument('--config_dir', default='configs/experiments/image_fitting')
    p.add_argument('--out_root',   default='results/image_fitting_groups')
    p.add_argument('--max_size', type=int, default=None,
                   help='Resize longer side to N pixels (ignored if --downscale set)')
    p.add_argument('--downscale', type=int, default=4,
                   help='Divide both W and H by this factor (BICUBIC). '
                        'Standard DIV2K-1/N convention; default 4 matches '
                        'common INR paper setting. Pass 1 to disable (full res).')
    p.add_argument('--skip_existing', action='store_true',
                   help='Skip (group, method, image) triples that already have a .pt result')
    p.add_argument('--seed', type=int, default=1234,
                   help='Per-run seed: reset to this before each model init so reruns are reproducible')
    return p.parse_args()


def run_one(task_mod, cfg, coords, pixels, meta, device, save_dir):
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    n_params={n_params:,}')
    return task_mod.run(model, coords, pixels, meta, cfg, device, save_dir=str(save_dir))


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[fitting-groups] device={device}')

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    task_mod = get_task('image_fitting')
    summary = {g: {} for g in args.groups}

    # Preload each group's images once (shared across methods to avoid repeat disk IO)
    group_images = {}
    for g in args.groups:
        ds_kw = {'normalize': True}
        if args.downscale and args.downscale > 1:
            ds_kw['downscale'] = args.downscale
        elif args.max_size:
            ds_kw['max_size'] = args.max_size
        ds = get_dataset('div2k', f'data/{g}', **ds_kw)
        group_images[g] = list(ds.iter_images())
        sizes = set((m['H'], m['W']) for _, _, m in group_images[g])
        print(f'[{g}] {len(group_images[g])} images loaded, sizes={sizes}')

    grand_t0 = time.time()
    total_runs = sum(len(group_images[g]) for g in args.groups) * len(args.methods)
    runs_done = 0

    for g in args.groups:
        for method in args.methods:
            cfg_path = Path(args.config_dir) / f'{method}.yaml'
            if not cfg_path.exists():
                print(f'[skip] {method}: {cfg_path} not found')
                continue
            cfg = load_config(str(cfg_path))

            save_dir_base = out_root / g / method
            save_dir_base.mkdir(parents=True, exist_ok=True)

            print(f'\n=== {g} / {method} ===')
            per_image = []

            for coords, pixels, meta in group_images[g]:
                runs_done += 1
                progress = f'[{runs_done}/{total_runs}]'
                pt_path = save_dir_base / f'{meta["name"]}_results.pt'
                if args.skip_existing and pt_path.exists():
                    print(f'  {progress} [{meta["name"]}] skip — {pt_path.name} exists')
                    loaded = torch.load(pt_path, weights_only=False)
                    per_image.append({
                        'name':       meta['name'],
                        'best_psnr':  loaded.get('best_psnr'),
                        'best_ssim':  loaded.get('best_ssim'),
                        'final_psnr': loaded.get('final_psnr'),
                        'final_ssim': loaded.get('final_ssim'),
                    })
                    continue

                print(f'  {progress} --- {meta["name"]} ({meta["H"]}x{meta["W"]}) ---')
                img_save_dir = save_dir_base / meta['name']
                try:
                    torch.manual_seed(args.seed)
                    if device.type == 'cuda':
                        torch.cuda.manual_seed_all(args.seed)
                    t0 = time.time()
                    result = run_one(task_mod, cfg, coords, pixels, meta,
                                     device, img_save_dir)
                    dt = time.time() - t0
                    torch.save(dict(result), pt_path)
                    per_image.append({
                        'name':       meta['name'],
                        'best_psnr':  result['best_psnr'],
                        'best_ssim':  result['best_ssim'],
                        'final_psnr': result['final_psnr'],
                        'final_ssim': result['final_ssim'],
                    })
                    print(f'    → best_PSNR={result["best_psnr"]:.2f}  '
                          f'best_SSIM={result["best_ssim"]:.4f}  '
                          f'({dt:.1f}s)')
                except Exception as e:
                    print(f'    ERROR: {type(e).__name__}: {e}')
                    per_image.append({
                        'name': meta['name'],
                        'best_psnr': None, 'best_ssim': None,
                        'final_psnr': None, 'final_ssim': None,
                        'error': str(e),
                    })
                finally:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

            summary[g][method] = per_image

    # ---------------- aggregate ----------------
    print('\n' + '=' * 80)
    print('SUMMARY (averaged over valid images per group)')
    print('=' * 80)

    def _mean(vals):
        ok = [v for v in vals if v is not None]
        return sum(ok) / len(ok) if ok else None

    agg_rows = []
    for g in args.groups:
        for method in args.methods:
            data = summary[g].get(method, [])
            if not data:
                continue
            row = {
                'group': g,
                'method': method,
                'n_images': sum(1 for d in data if d['best_psnr'] is not None),
                'avg_best_psnr':  _mean([d['best_psnr']  for d in data]),
                'avg_best_ssim':  _mean([d['best_ssim']  for d in data]),
                'avg_final_psnr': _mean([d['final_psnr'] for d in data]),
                'avg_final_ssim': _mean([d['final_ssim'] for d in data]),
                'per_image': data,
            }
            agg_rows.append(row)
            def _fmt(v, nd=3):
                return 'n/a' if v is None else f'{v:.{nd}f}'
            print(f"{g:8s}  {method:8s}  "
                  f"best_PSNR={_fmt(row['avg_best_psnr'])}  "
                  f"best_SSIM={_fmt(row['avg_best_ssim'], 4)}  "
                  f"final_PSNR={_fmt(row['avg_final_psnr'])}  "
                  f"final_SSIM={_fmt(row['avg_final_ssim'], 4)}  "
                  f"(n={row['n_images']})")

    # Save JSON
    (out_root / 'summary.json').write_text(json.dumps(agg_rows, indent=2))
    print(f"\n[saved] {out_root / 'summary.json'}")

    # Save Markdown table
    md_lines = [
        '# Image Fitting on High / Mid / Low 5-image groups',
        '',
        '| group | method | n | avg best PSNR | avg best SSIM | avg final PSNR | avg final SSIM |',
        '|-------|--------|---|---------------|---------------|----------------|----------------|',
    ]
    def _md(v, nd=2):
        return 'n/a' if v is None else f'{v:.{nd}f}'
    for r in agg_rows:
        md_lines.append(
            f"| {r['group']} | {r['method']} | {r['n_images']} | "
            f"{_md(r['avg_best_psnr'])} | {_md(r['avg_best_ssim'], 4)} | "
            f"{_md(r['avg_final_psnr'])} | {_md(r['avg_final_ssim'], 4)} |"
        )
    (out_root / 'summary.md').write_text('\n'.join(md_lines) + '\n')
    print(f"[saved] {out_root / 'summary.md'}")

    print(f"\n[total time] {time.time() - grand_t0:.1f}s")


if __name__ == '__main__':
    main()
