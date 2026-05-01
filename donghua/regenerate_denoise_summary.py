"""
Regenerate summary.md / summary.json for the image-denoising groups benchmark
from the per-image *_results.pt files that run_denoise_groups.py already saved.

Mirrors regenerate_fitting_summary.py / regenerate_sr_summary.py /
regenerate_inpaint_summary.py — but with denoise-specific extras:
    - per_image carries noise_info, noisy_input_psnr, best_epoch
    - aggregate adds avg_noisy_psnr and gain (avg_best_psnr - avg_noisy_psnr),
      which is the actual denoise metric of interest.

Default groups are high_5 + mid_5 only (not low_5).

Usage:
    python donghua/regenerate_denoise_summary.py
    python donghua/regenerate_denoise_summary.py --groups high_5 mid_5 low_5
"""

import argparse
import json
from pathlib import Path

import torch


GROUPS_DEFAULT  = ['high_5', 'mid_5']                # ← only high & mid
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
    p.add_argument('--root', default='results/image_denoise_groups',
                   help='Directory containing <group>/<method>/<name>_results.pt')
    p.add_argument('--out_dir', default='donghua',
                   help='Where to write summary.json / summary.md')
    p.add_argument('--groups',  nargs='+', default=GROUPS_DEFAULT)
    p.add_argument('--methods', nargs='+', default=METHODS_DEFAULT)
    p.add_argument('--out_prefix', default='denoise_summary',
                   help='Output filename prefix (avoids clobbering other '
                        'task summaries already in --out_dir)')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f'[error] {root} does not exist')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_rows = []
    used_noise = None
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
                if used_noise is None:
                    used_noise = d.get('noise_info')
                per_image.append({
                    'name':             d.get('name', pt.stem.replace('_results', '')),
                    'best_psnr':        _to_py(d.get('best_psnr')),
                    'best_ssim':        _to_py(d.get('best_ssim')),
                    'final_psnr':       _to_py(d.get('final_psnr')),
                    'final_ssim':       _to_py(d.get('final_ssim')),
                    'noisy_input_psnr': _to_py(d.get('noisy_input_psnr')),
                    'best_epoch':       d.get('best_epoch'),
                    'noise_info':       d.get('noise_info'),
                })

            row = {
                'group': g,
                'method': method,
                'n_images':       sum(1 for d in per_image if d['best_psnr'] is not None),
                'avg_best_psnr':  _mean([d['best_psnr']  for d in per_image]),
                'avg_best_ssim':  _mean([d['best_ssim']  for d in per_image]),
                'avg_final_psnr': _mean([d['final_psnr'] for d in per_image]),
                'avg_final_ssim': _mean([d['final_ssim'] for d in per_image]),
                'avg_noisy_psnr': _mean([d.get('noisy_input_psnr') for d in per_image]),
                'per_image': per_image,
            }
            row['gain'] = (row['avg_best_psnr'] - row['avg_noisy_psnr']
                           if row['avg_best_psnr'] is not None
                              and row['avg_noisy_psnr'] is not None
                           else None)
            agg_rows.append(row)

            def _fmt(v, nd=3):
                return 'n/a' if v is None else f'{v:.{nd}f}'
            gain = row['gain']
            gain_s = 'n/a' if gain is None else f'{gain:+.2f}dB'
            print(f"{g:8s}  {method:8s}  "
                  f"best_PSNR={_fmt(row['avg_best_psnr'])}  "
                  f"best_SSIM={_fmt(row['avg_best_ssim'], 4)}  "
                  f"noisy_PSNR={_fmt(row['avg_noisy_psnr'])}  "
                  f"gain={gain_s}  "
                  f"(n={row['n_images']})")

    json_path = out_dir / f'{args.out_prefix}.json'
    json_path.write_text(json.dumps(agg_rows, indent=2))
    print(f"\n[saved] {json_path}")

    # Build a human-readable noise label for the markdown title
    if used_noise is None:
        noise_label = '?'
    elif used_noise.get('noise_type') == 'poisson_gaussian':
        noise_label = (f"poisson_gaussian "
                       f"(τ={used_noise.get('tau')}, "
                       f"readout_snr={used_noise.get('readout_snr')})")
    elif used_noise.get('noise_type') == 'gaussian':
        noise_label = f"gaussian (σ={used_noise.get('sigma')})"
    else:
        noise_label = str(used_noise)

    grp_label = ' / '.join(args.groups)
    md_lines = [
        f'# Image Denoising ({noise_label}) on {grp_label} 5-image groups',
        '',
        '| group | method | n | avg best PSNR | avg best SSIM | avg final PSNR | avg final SSIM | avg noisy PSNR | gain |',
        '|-------|--------|---|---------------|---------------|----------------|----------------|----------------|------|',
    ]
    def _md(v, nd=2):
        return 'n/a' if v is None else f'{v:.{nd}f}'
    for r in agg_rows:
        gain = r['gain']
        gain_s = 'n/a' if gain is None else f'{gain:+.2f}'
        md_lines.append(
            f"| {r['group']} | {r['method']} | {r['n_images']} | "
            f"{_md(r['avg_best_psnr'])} | {_md(r['avg_best_ssim'], 4)} | "
            f"{_md(r['avg_final_psnr'])} | {_md(r['avg_final_ssim'], 4)} | "
            f"{_md(r['avg_noisy_psnr'])} | {gain_s} |"
        )
    md_path = out_dir / f'{args.out_prefix}.md'
    md_path.write_text('\n'.join(md_lines) + '\n')
    print(f"[saved] {md_path}")


if __name__ == '__main__':
    main()
