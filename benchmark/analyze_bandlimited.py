"""
benchmark/analyze_bandlimited.py

Generates 5 summary figures for the 2d_bandlimited experiment.
Run after: python benchmark/run_synthetic.py --signal 2d_bandlimited

Output  → results/2d_bandlimited/figures/
  01_psnr_vs_bandwidth.png
  02_psnr_heatmap.png
  03_freq_band_error.png
  04_oob_leakage.png
  05_radial_spectrum_bwX.X.png   (one per --spectrum_bws)

Usage:
    python benchmark/analyze_bandlimited.py
    python benchmark/analyze_bandlimited.py --out_dir results/2d_bandlimited
    python benchmark/analyze_bandlimited.py --methods siren wire gauss finer gf relu
    python benchmark/analyze_bandlimited.py --spectrum_bws 0.3 0.5 0.7 --spectrum_seed 1234
"""

import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark.run_synthetic import _CUTOFFS

def bw_to_freq(bw):
    idx = int(round(bw * 10)) - 1
    return float(_CUTOFFS[idx] / np.sqrt(2))

# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    'relu':   'ReLU',   'pemlp':  'PE-MLP',
    'siren':  'SIREN',  'wire':   'WIRE',
    'gauss':  'Gaussian', 'finer': 'FINER',
    'gf':     'FINER+(G)', 'wf':  'FINER+(W)',
    'staf':   'STAF',   'incode': 'INCODE',
    'sl2a':   'SL2A',   'cosmo':  'COSMO',
}
COLORS = {
    'relu': '#bdc3c7', 'pemlp': '#8C8C8C',
    'siren': '#4C72B0', 'wire': '#DD8452',
    'gauss': '#55A868', 'finer': '#C44E52',
    'gf': '#8172B3', 'wf': '#937860',
    'staf': '#DA8BC3', 'incode': '#CCB974',
    'sl2a': '#64B5CD', 'cosmo': '#2ecc71',
}
MARKERS = {
    'relu': 'o', 'pemlp': 's', 'siren': '^', 'wire': 'v',
    'gauss': 'D', 'finer': 'P', 'gf': 'X', 'wf': '*',
    'staf': 'h', 'incode': 'p', 'sl2a': '<', 'cosmo': '>',
}
DEFAULT_ORDER = ['relu', 'siren', 'wire', 'gauss', 'finer', 'gf', 'wf',
                 'staf', 'pemlp', 'incode', 'sl2a', 'cosmo']

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_results(path):
    """Returns {method: {bw: [psnr, ...]}}"""
    data = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            m  = row['method']
            bw = float(row['bandwidth'])
            data.setdefault(m, {}).setdefault(bw, []).append(float(row['psnr']))
    return data


def load_freq(path):
    """Returns {method: {bw: {field: [values over seeds]}}}"""
    data = {}
    fields = ['gt_energy_low', 'gt_energy_mid', 'gt_energy_high',
              'err_low', 'err_mid', 'err_high',
              'rel_err_low', 'rel_err_mid', 'rel_err_high',
              'oob_leakage']
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            m  = row['method']
            bw = float(row['bandwidth'])
            data.setdefault(m, {}).setdefault(bw, {})
            for k in fields:
                if k in row and row[k] not in ('', 'nan'):
                    try:
                        data[m][bw].setdefault(k, []).append(float(row[k]))
                    except ValueError:
                        pass
    return data

# ---------------------------------------------------------------------------
# Plot 1 — PSNR vs Bandwidth
# ---------------------------------------------------------------------------

def plot_psnr_vs_bw(results, methods, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        if m not in results:
            continue
        bws   = sorted(results[m])
        freqs = [bw_to_freq(bw) for bw in bws]
        means = [np.mean(results[m][bw]) for bw in bws]
        stds  = [np.std(results[m][bw])  for bw in bws]
        ax.errorbar(freqs, means, yerr=stds,
                    label=METHOD_LABELS.get(m, m),
                    color=COLORS.get(m), marker=MARKERS.get(m, 'o'),
                    markersize=5, linewidth=1.5, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Cutoff frequency (log scale)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR vs Bandwidth')
    freqs_all = [bw_to_freq(bw) for bw in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
    ax.set_xticks(freqs_all)
    ax.set_xticklabels([f'{f:.4g}' for f in freqs_all], rotation=30, ha='right', fontsize=7)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    _save(fig, fig_dir, '01_psnr_vs_bandwidth.png')

# ---------------------------------------------------------------------------
# Plot 2 — Method × Bandwidth PSNR heatmap
# ---------------------------------------------------------------------------

def plot_psnr_heatmap(results, methods, fig_dir):
    methods_present = [m for m in methods if m in results]
    all_bws = sorted({bw for m in methods_present for bw in results[m]})
    if not methods_present or not all_bws:
        return

    matrix = np.full((len(methods_present), len(all_bws)), np.nan)
    for i, m in enumerate(methods_present):
        for j, bw in enumerate(all_bws):
            if bw in results[m]:
                matrix[i, j] = np.mean(results[m][bw])

    fig, ax = plt.subplots(figsize=(max(7, len(all_bws) * 0.8),
                                    max(4, len(methods_present) * 0.55)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='PSNR (dB)')

    ax.set_xticks(range(len(all_bws)))
    ax.set_xticklabels([f'{bw_to_freq(bw):.4g}' for bw in all_bws],
                       rotation=30, ha='right', fontsize=7)
    ax.set_yticks(range(len(methods_present)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods_present])
    ax.set_xlabel('Cutoff frequency')
    ax.set_title('PSNR Heatmap (Method × Bandwidth)')

    # Annotate cells
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                text_color = 'black' if (val - vmin) / (vmax - vmin + 1e-9) > 0.5 else 'white'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=7, color=text_color)
    plt.tight_layout()
    _save(fig, fig_dir, '02_psnr_heatmap.png')

# ---------------------------------------------------------------------------
# Plot 3 — Relative freq band error vs bandwidth
# ---------------------------------------------------------------------------

def plot_freq_band_error(freq, methods, fig_dir):
    bands  = ['low', 'mid', 'high']
    blabels = ['Low (0–20%)', 'Mid (20–50%)', 'High (50–100%)']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, band, blabel in zip(axes, bands, blabels):
        for m in methods:
            if m not in freq:
                continue
            bws, means, stds = [], [], []
            for bw in sorted(freq[m]):
                vals = np.array(freq[m][bw].get(f'rel_err_{band}', []))
                if len(vals) == 0:
                    # fallback: compute from raw columns
                    errs = np.array(freq[m][bw].get(f'err_{band}', []))
                    gts  = np.array(freq[m][bw].get(f'gt_energy_{band}', []))
                    vals = errs / (gts + 1e-12) if len(errs) else np.array([])
                if len(vals) == 0:
                    continue
                bws.append(bw)
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            if not bws:
                continue
            freqs = [bw_to_freq(bw) for bw in bws]
            ax.errorbar(freqs, means, yerr=stds,
                        label=METHOD_LABELS.get(m, m),
                        color=COLORS.get(m), marker=MARKERS.get(m, 'o'),
                        markersize=4, linewidth=1.5, capsize=3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Cutoff frequency (log scale)')
        ax.set_ylabel('Relative Error (err / GT energy)')
        ax.set_title(f'{blabel} Band Error')
        sel_freqs = [bw_to_freq(bw) for bw in [0.1, 0.3, 0.5, 0.7, 0.9]]
        ax.set_xticks(sel_freqs)
        ax.set_xticklabels([f'{f:.4g}' for f in sel_freqs], rotation=30, ha='right', fontsize=7)
        ax.grid(True, alpha=0.3, which='both')

    axes[0].legend(fontsize=7, ncol=1)
    plt.suptitle('Frequency Band Reconstruction Error', fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, fig_dir, '03_freq_band_error.png', bbox_inches='tight')

# ---------------------------------------------------------------------------
# Plot 4 — Out-of-band leakage vs bandwidth
# ---------------------------------------------------------------------------

def plot_oob_leakage(freq, methods, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        if m not in freq:
            continue
        bws, means, stds = [], [], []
        for bw in sorted(freq[m]):
            vals = [v for v in freq[m][bw].get('oob_leakage', [])
                    if not np.isnan(v)]
            if not vals:
                continue
            bws.append(bw)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        if not bws:
            continue
        freqs = [bw_to_freq(bw) for bw in bws]
        ax.errorbar(freqs, means, yerr=stds,
                    label=METHOD_LABELS.get(m, m),
                    color=COLORS.get(m), marker=MARKERS.get(m, 'o'),
                    markersize=5, linewidth=1.5, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Cutoff frequency (log scale)')
    ax.set_ylabel('Out-of-band energy fraction')
    ax.set_title('Out-of-band Leakage vs Bandwidth')
    freqs_all = [bw_to_freq(bw) for bw in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
    ax.set_xticks(freqs_all)
    ax.set_xticklabels([f'{f:.4g}' for f in freqs_all], rotation=30, ha='right', fontsize=7)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    _save(fig, fig_dir, '04_oob_leakage.png')

# ---------------------------------------------------------------------------
# Plot 5 — Radial spectrum comparison (one figure per bandwidth)
# ---------------------------------------------------------------------------

def plot_radial_spectrum(out_dir, methods, bandwidth, seed, fig_dir):
    from benchmark.run_synthetic import _CUTOFFS
    bw_str     = f'{bandwidth:.1f}'
    run_subdir = f'bw{bw_str}_seed{seed}'

    cutoff_r = None
    try:
        idx = int(round(bandwidth * 10)) - 1
        if 0 <= idx < len(_CUTOFFS):
            cutoff_r = _CUTOFFS[idx] / np.sqrt(2)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_pred, ax_res = axes
    gt_drawn = False

    for m in methods:
        run_dir  = os.path.join(out_dir, m, run_subdir)
        npz_path = os.path.join(run_dir, 'radial_spectrum.npz')

        if os.path.exists(npz_path):
            d = np.load(npz_path)
            centers, gt_p, pred_p, res_p = (
                d['centers'], d['gt_power'], d['pred_power'], d['residual_power'])
        else:
            # fallback: compute from saved numpy arrays
            gt_npy   = os.path.join(run_dir, 'gt_norm.npy')
            pred_npy = os.path.join(run_dir, 'best_recon.npy')
            if not (os.path.exists(gt_npy) and os.path.exists(pred_npy)):
                continue
            from benchmark.run_synthetic import _compute_radial_spectrum_nd
            gt_norm = np.load(gt_npy)
            pred    = np.load(pred_npy)
            centers, gt_p   = _compute_radial_spectrum_nd(gt_norm)
            _,       pred_p = _compute_radial_spectrum_nd(pred)
            _,       res_p  = _compute_radial_spectrum_nd(gt_norm - pred)

        label  = METHOD_LABELS.get(m, m)
        color  = COLORS.get(m)

        if not gt_drawn:
            ax_pred.semilogy(centers, gt_p + 1e-30, 'k-',
                             linewidth=2, label='GT', zorder=10)
            gt_drawn = True

        ax_pred.semilogy(centers, pred_p + 1e-30,
                         label=label, color=color, linewidth=1.2)
        ax_res.semilogy(centers, res_p + 1e-30,
                        label=label, color=color, linewidth=1.2)

    if not gt_drawn:
        print(f'  [skip] no spectrum data for bw={bw_str} seed={seed}')
        plt.close()
        return

    for ax in axes:
        ax.set_xscale('log')
        if cutoff_r is not None:
            ax.axvline(cutoff_r, color='gray', linestyle='--', linewidth=1,
                       label=f'GT cutoff ({cutoff_r:.4g})')
        ax.set_xlabel('Radial frequency (log scale)')
        ax.set_ylabel('Mean power (log)')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=7, ncol=2)

    ax_pred.set_title(f'GT vs Predicted  |  bw={bw_str}  seed={seed}')
    ax_res.set_title(f'Residual Spectrum  |  bw={bw_str}  seed={seed}')
    plt.tight_layout()
    _save(fig, fig_dir, f'05_radial_spectrum_bw{bw_str}.png')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, fig_dir, name, **kwargs):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150, **kwargs)
    plt.close(fig)
    print(f'  saved {path}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/2d_bandlimited')
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--spectrum_bws', nargs='+', type=float, default=[0.3, 0.5, 0.7])
    p.add_argument('--spectrum_seed', type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(_ROOT, args.out_dir)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    results_csv = os.path.join(out_dir, 'results.csv')
    freq_csv    = os.path.join(out_dir, 'freq_analysis.csv')

    if not os.path.exists(results_csv):
        sys.exit(f'[error] {results_csv} not found — run run_synthetic.py first')

    results = load_results(results_csv)
    freq    = load_freq(freq_csv) if os.path.exists(freq_csv) else {}

    methods = args.methods or [m for m in DEFAULT_ORDER if m in results]
    print(f'[analyze_bandlimited] methods: {methods}')

    plot_psnr_vs_bw(results, methods, fig_dir)
    plot_psnr_heatmap(results, methods, fig_dir)

    if freq:
        plot_freq_band_error(freq, methods, fig_dir)
        plot_oob_leakage(freq, methods, fig_dir)
    else:
        print('  [skip] freq_analysis.csv not found')

    for bw in args.spectrum_bws:
        plot_radial_spectrum(out_dir, methods, bw, args.spectrum_seed, fig_dir)

    print(f'\n[analyze_bandlimited] done → {fig_dir}')


if __name__ == '__main__':
    main()
