"""
Regenerate summary.md / summary.json for the image-fitting groups benchmark
from the per-image *_results.pt files that run_fitting_groups.py already saved.

Use this when the original run finished but the summary write step crashed
(e.g. numpy float32 not JSON serializable).

Usage:
    python donghua/regenerate_fitting_summary.py
    python donghua/regenerate_fitting_summary.py \
        --root results/image_fitting_groupss \
        --groups high_5 mid_5 low_5 \
        --methods siren wire ...
"""

import argparse
import json
from pathlib import Path

import torch


GROUPS_DEFAULT  = ['high_5', 'mid_5', 'low_5']
METHODS_DEFAULT = ['siren', 'wire', 'gauss', 'finer', 'gf', 'wf',
                   'staf', 'relu', 'incode', 'sl2a', 'cosmo']


def _to_py(v):
    if v is None:
        return None
    if hasattr(v, 'item'):
        return v.item()
    return float(v)


def _mean(vals):
    ok = [v for v in vals if v is not None]
    return sum(ok) / len(ok) if ok else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='results/image_fitting_groupss',
                   help='Directory containing <group>/<method>/<name>_results.pt')
    p.add_argument('--out_dir', default='donghua',
                   help='Where to write summary.json / summary.md')
    p.add_argument('--groups',  nargs='+', default=GROUPS_DEFAULT)
    p.add_argument('--methods', nargs='+', default=METHODS_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f'[error] {root} does not exist')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_rows = []
    for g in args.groups:
        for method in args.methods:
            mdir = root / g / method
            if not mdir.is_dir():
                continue
            pt_files = sorted(mdir.glob('*_results.pt'))
            if not pt_files:
                continue

            per_image = []
            for pt in pt_files:
                d = torch.load(pt, map_location='cpu', weights_only=False)
                per_image.append({
                    'name':       d.get('name', pt.stem.replace('_results', '')),
                    'best_psnr':  _to_py(d.get('best_psnr')),
                    'best_ssim':  _to_py(d.get('best_ssim')),
                    'final_psnr': _to_py(d.get('final_psnr')),
                    'final_ssim': _to_py(d.get('final_ssim')),
                })

            row = {
                'group': g,
                'method': method,
                'n_images':       sum(1 for d in per_image if d['best_psnr'] is not None),
                'avg_best_psnr':  _mean([d['best_psnr']  for d in per_image]),
                'avg_best_ssim':  _mean([d['best_ssim']  for d in per_image]),
                'avg_final_psnr': _mean([d['final_psnr'] for d in per_image]),
                'avg_final_ssim': _mean([d['final_ssim'] for d in per_image]),
                'per_image': per_image,
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

    (out_dir / 'summary.json').write_text(json.dumps(agg_rows, indent=2))
    print(f"\n[saved] {out_dir / 'summary.json'}")

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
    (out_dir / 'summary.md').write_text('\n'.join(md_lines) + '\n')
    print(f"[saved] {out_dir / 'summary.md'}")


if __name__ == '__main__':
    main()
