"""
Generate LaTeX / CSV comparison tables from collected results.
"""

import os
import json
import numpy as np
import argparse
from collections import defaultdict


def load_summary(json_path):
    with open(json_path) as f:
        return json.load(f)


def make_csv(summary, metric='psnr', out_path=None):
    """
    Rows = methods, Columns = images + mean.
    """
    methods = sorted(summary.keys())
    all_images = sorted({img for m in summary.values() for img in m})

    header = ['method'] + all_images + ['mean']
    rows = [header]

    for m in methods:
        row = [m]
        vals = []
        for img in all_images:
            v = summary[m].get(img, {}).get(metric, float('nan'))
            row.append(f'{v:.4f}')
            vals.append(v)
        row.append(f'{np.nanmean(vals):.4f}')
        rows.append(row)

    lines = [','.join(r) for r in rows]
    text = '\n'.join(lines)
    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
        print(f"CSV saved to {out_path}")
    else:
        print(text)
    return text


def make_latex(summary, metric='psnr', out_path=None, bold_best=True):
    """
    LaTeX table: methods as rows, mean±std as columns.
    """
    methods = sorted(summary.keys())

    means = {}
    stds = {}
    for m in methods:
        vals = [v[metric] for v in summary[m].values()
                if not np.isnan(v.get(metric, float('nan')))]
        means[m] = np.mean(vals) if vals else float('nan')
        stds[m]  = np.std(vals)  if vals else float('nan')

    best_m = max(methods, key=lambda m: means[m])

    lines = [
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'Method & ' + metric.upper() + r' (mean) & Std \\ \midrule',
    ]
    for m in methods:
        val_str = f'{means[m]:.2f}'
        std_str = f'{stds[m]:.2f}'
        if bold_best and m == best_m:
            val_str = r'\textbf{' + val_str + '}'
        lines.append(f'{m} & {val_str} & {std_str} \\\\')

    lines += [r'\bottomrule', r'\end{tabular}']
    text = '\n'.join(lines)

    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
        print(f"LaTeX table saved to {out_path}")
    else:
        print(text)
    return text


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True, help='Path to summary JSON')
    p.add_argument('--metric', default='psnr')
    p.add_argument('--format', choices=['csv', 'latex', 'both'], default='both')
    p.add_argument('--out_dir', default='results/tables')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = load_summary(args.json)

    if args.format in ('csv', 'both'):
        make_csv(summary, args.metric,
                 os.path.join(args.out_dir, f'{args.metric}.csv'))
    if args.format in ('latex', 'both'):
        make_latex(summary, args.metric,
                   os.path.join(args.out_dir, f'{args.metric}.tex'))
