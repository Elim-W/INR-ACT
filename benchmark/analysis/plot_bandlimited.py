"""
Frequency-domain analysis plots for the 2d_bandlimited experiment.

Usage:
    python benchmark/analysis/plot_bandlimited.py
    python benchmark/analysis/plot_bandlimited.py --out_dir results/2d_bandlimited
    python benchmark/analysis/plot_bandlimited.py --methods siren wire gauss finer gf relu

Generates figures in <out_dir>/figures/:
    01_psnr_vs_bandwidth.png      - PSNR (y) vs bandwidth (x), one line per method
    02_freq_band_error.png        - relative error in low/mid/high Fourier bands
    03_radial_spectrum_bwX.X.png  - GT vs predicted radial power spectrum
    04_oob_leakage.png            - out-of-band energy leakage vs bandwidth
"""

import os
import sys
import csv
import argparse

import numpy as np
import numpy.fft as fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    'relu':   'ReLU',
    'pemlp':  'PE-MLP',
    'siren':  'SIREN',
    'wire':   'WIRE',
    'gauss':  'Gaussian',
    'finer':  'FINER',
    'gf':     'FINER+(G)',
    'wf':     'FINER+(W)',
    'staf':   'STAF',
    'incode': 'INCODE',
    'sl2a':   'SL2A',
    'cosmo':  'COSMO',
}

COLORS = {
    'relu':   '#bdc3c7',
    'pemlp':  '#8C8C8C',
    'siren':  '#4C72B0',
    'wire':   '#DD8452',
    'gauss':  '#55A868',
    'finer':  '#C44E52',
    'gf':     '#8172B3',
    'wf':     '#937860',
    'staf':   '#DA8BC3',
    'incode': '#CCB974',
    'sl2a':   '#64B5CD',
    'cosmo':  '#2ecc71',
}

MARKERS = {
    'relu': 'o', 'pemlp': 's', 'siren': '^', 'wire': 'v',
    'gauss': 'D', 'finer': 'P', 'gf': 'X', 'wf': '*',
    'staf': 'h', 'incode': 'p', 'sl2a': '<', 'cosmo': '>',
}

DEFAULT_METHOD_ORDER = [
    'relu', 'siren', 'wire', 'gauss', 'finer', 'gf', 'wf',
    'staf', 'pemlp', 'incode', 'sl2a', 'cosmo',
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results_csv(path):
    """
    Returns {method: {bandwidth: [psnr, ...]}} aggregated over seeds.
    """
    data = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            m = row['method']
            bw = float(row['bandwidth'])
            psnr = float(row['psnr'])
            data.setdefault(m, {}).setdefault(bw, []).append(psnr)
    return data


def load_freq_csv(path):
    """
    Returns {method: {bandwidth: {field: [values over seeds]}}}
    Fields: gt_energy_{low,mid,high}, err_{low,mid,high}, oob_leakage.
    """
    data = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            m = row['method']
            bw = float(row['bandwidth'])
            data.setdefault(m, {}).setdefault(bw, {})
            for k in ['gt_energy_low', 'gt_energy_mid', 'gt_energy_high',
                      'err_low', 'err_mid', 'err_high', 'oob_leakage']:
                if k in row:
                    data[m][bw].setdefault(k, []).append(float(row[k]))
    return data


def _mean_std(values_list):
    arr = np.array(values_list)
    return arr.mean(), arr.std()


# ---------------------------------------------------------------------------
# Radial power spectrum
# ---------------------------------------------------------------------------

def radial_spectrum(arr_2d, n_bins=100):
    """
    Compute radial mean power spectrum of a 2D array.
    Returns (bin_centers, mean_power) where bin_centers are in fftfreq units.
    """
    power = np.abs(fft.fft2(arr_2d)) ** 2
    H, W = arr_2d.shape
    fy = fft.fftfreq(H)[:, None]
    fx = fft.fftfreq(W)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)

    max_r = np.sqrt(0.5 ** 2 + 0.5 ** 2)
    bins = np.linspace(0, max_r, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    mean_power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (bins[i] <= radius) & (radius < bins[i + 1])
        if mask.any():
            mean_power[i] = power[mask].mean()

    return centers, mean_power


# ---------------------------------------------------------------------------
# Plot 1: PSNR vs Bandwidth
# ---------------------------------------------------------------------------

def plot_psnr_vs_bandwidth(results_data, methods, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        if m not in results_data:
            continue
        bws_data = results_data[m]
        bws = sorted(bws_data.keys())
        means = [np.mean(bws_data[bw]) for bw in bws]
        stds  = [np.std(bws_data[bw])  for bw in bws]
        label = METHOD_LABELS.get(m, m)
        color = COLORS.get(m, None)
        marker = MARKERS.get(m, 'o')
        ax.errorbar(bws, means, yerr=stds, label=label, color=color,
                    marker=marker, markersize=5, linewidth=1.5, capsize=3)

    ax.set_xlabel('Bandwidth', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR vs Bandwidth', fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.tight_layout()
    out = os.path.join(fig_dir, '01_psnr_vs_bandwidth.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Plot 2: Frequency Band Error
# ---------------------------------------------------------------------------

def plot_freq_band_error(freq_data, methods, fig_dir):
    """
    3 subplots (low / mid / high band), each showing relative error vs bandwidth.
    Relative error = err_band / gt_energy_band (ratio; lower is better).
    """
    bands = ['low', 'mid', 'high']
    band_labels = ['Low (0–20%)', 'Mid (20–50%)', 'High (50–100%)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, band, blabel in zip(axes, bands, band_labels):
        for m in methods:
            if m not in freq_data:
                continue
            bws = sorted(freq_data[m].keys())
            rel_means, rel_stds = [], []
            for bw in bws:
                d = freq_data[m][bw]
                # Prefer pre-computed rel_err from CSV; fall back to computing it
                rel = np.array(d.get(f'rel_err_{band}', []))
                if len(rel) == 0:
                    errs = np.array(d.get(f'err_{band}', []))
                    gts  = np.array(d.get(f'gt_energy_{band}', []))
                    rel  = errs / (gts + 1e-30) if len(errs) else np.array([np.nan])
                rel_means.append(np.nanmean(rel))
                rel_stds.append(np.nanstd(rel))

            label = METHOD_LABELS.get(m, m)
            color = COLORS.get(m, None)
            marker = MARKERS.get(m, 'o')
            ax.errorbar(bws, rel_means, yerr=rel_stds, label=label, color=color,
                        marker=marker, markersize=4, linewidth=1.5, capsize=3)

        ax.set_xlabel('Bandwidth', fontsize=11)
        ax.set_ylabel('Relative Error (err / GT energy)', fontsize=10)
        ax.set_title(f'{blabel} Freq Band Error', fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

    axes[0].legend(fontsize=7, ncol=1, loc='best')
    plt.suptitle('Frequency Band Reconstruction Error', fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(fig_dir, '02_freq_band_error.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Plot 3: Radial Fourier Spectrum
# ---------------------------------------------------------------------------

def plot_radial_spectrum(out_dir, methods, bandwidth, seed, fig_dir):
    """
    For a fixed (bandwidth, seed), plot GT / predicted / residual radial power spectra.
    Reads radial_spectrum.npz saved by run_synthetic.py (one file per method run).
    Falls back to computing from gt_norm.npy + best_recon.npy if npz is absent.
    """
    bw_str = f'{bandwidth:.1f}'
    run_subdir = f'bw{bw_str}_seed{seed}'

    # Cutoff radius line (bandlimited signals only)
    from benchmark.run_synthetic import _CUTOFFS
    cutoff_r = None
    try:
        idx = int(round(bandwidth * 10)) - 1
        if 0 <= idx < len(_CUTOFFS):
            cutoff_r = _CUTOFFS[idx] / np.sqrt(2)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_pred, ax_res = axes
    gt_plotted = False

    for m in methods:
        run_dir = os.path.join(out_dir, m, run_subdir)
        npz_path = os.path.join(run_dir, 'radial_spectrum.npz')

        if os.path.exists(npz_path):
            d = np.load(npz_path)
            centers      = d['centers']
            gt_power     = d['gt_power']
            pred_power   = d['pred_power']
            residual_power = d['residual_power']
        else:
            # Fallback: recompute from saved numpy arrays
            gt_npy   = os.path.join(run_dir, 'gt_norm.npy')
            pred_npy = os.path.join(run_dir, 'best_recon.npy')
            if not (os.path.exists(gt_npy) and os.path.exists(pred_npy)):
                continue
            gt_norm = np.load(gt_npy)
            pred    = np.load(pred_npy)
            centers, gt_power   = radial_spectrum(gt_norm)
            _,       pred_power = radial_spectrum(pred)
            _,       residual_power = radial_spectrum(gt_norm - pred)

        label = METHOD_LABELS.get(m, m)
        color = COLORS.get(m, None)

        if not gt_plotted:
            ax_pred.semilogy(centers, gt_power + 1e-30, 'k-',
                             linewidth=2, label='GT', zorder=10)
            gt_plotted = True

        ax_pred.semilogy(centers, pred_power + 1e-30,
                         label=label, color=color, linewidth=1.2)
        ax_res.semilogy(centers, residual_power + 1e-30,
                        label=label, color=color, linewidth=1.2)

    if not gt_plotted:
        print(f'  [skip] no spectrum data for bw={bw_str} seed={seed}')
        plt.close()
        return

    for ax in axes:
        if cutoff_r is not None:
            ax.axvline(cutoff_r, color='gray', linestyle='--', linewidth=1,
                       label=f'GT cutoff ({cutoff_r:.3f})')
        ax.set_xlabel('Radial frequency', fontsize=11)
        ax.set_ylabel('Mean power (log scale)', fontsize=11)
        ax.grid(True, alpha=0.3)

    ax_pred.set_title(f'GT vs Predicted Spectrum  |  bw={bw_str} seed={seed}', fontsize=11)
    ax_res.set_title(f'Residual Spectrum  |  bw={bw_str} seed={seed}', fontsize=11)
    ax_pred.legend(fontsize=7, ncol=2)
    ax_res.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = os.path.join(fig_dir, f'03_radial_spectrum_bw{bw_str}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Plot 4: Out-of-band Leakage
# ---------------------------------------------------------------------------

def plot_oob_leakage(freq_data, methods, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        if m not in freq_data:
            continue
        bws = sorted(freq_data[m].keys())
        means, stds = [], []
        for bw in bws:
            vals = np.array(freq_data[m][bw].get('oob_leakage', []))
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) else 0.0)

        label = METHOD_LABELS.get(m, m)
        color = COLORS.get(m, None)
        marker = MARKERS.get(m, 'o')
        ax.errorbar(bws, means, yerr=stds, label=label, color=color,
                    marker=marker, markersize=5, linewidth=1.5, capsize=3)

    ax.set_xlabel('Bandwidth', fontsize=12)
    ax.set_ylabel('Out-of-band energy fraction', fontsize=12)
    ax.set_title('Out-of-band Leakage vs Bandwidth', fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.tight_layout()
    out = os.path.join(fig_dir, '04_oob_leakage.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/2d_bandlimited')
    p.add_argument('--methods', nargs='+', default=None,
                   help='Methods to plot (default: all found in results.csv)')
    p.add_argument('--spectrum_bandwidths', nargs='+', type=float,
                   default=[0.3, 0.5, 0.7],
                   help='Bandwidth values for radial spectrum plots')
    p.add_argument('--spectrum_seed', type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(ROOT, args.out_dir)

    results_csv = os.path.join(out_dir, 'results.csv')
    freq_csv    = os.path.join(out_dir, 'freq_analysis.csv')
    fig_dir     = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.exists(results_csv):
        print(f'[ERROR] {results_csv} not found. Run run_synthetic.py first.')
        return

    results_data = load_results_csv(results_csv)

    if args.methods is None:
        methods = [m for m in DEFAULT_METHOD_ORDER if m in results_data]
    else:
        methods = args.methods

    print(f'[plot_bandlimited] methods: {methods}')

    # Plot 1: PSNR vs Bandwidth
    plot_psnr_vs_bandwidth(results_data, methods, fig_dir)

    if os.path.exists(freq_csv):
        freq_data = load_freq_csv(freq_csv)

        # Plot 2: Frequency Band Error
        plot_freq_band_error(freq_data, methods, fig_dir)

        # Plot 4: Out-of-band Leakage
        plot_oob_leakage(freq_data, methods, fig_dir)
    else:
        print(f'  [skip] {freq_csv} not found — run run_synthetic.py to generate freq data')

    # Plot 3: Radial Spectrum (one figure per bandwidth)
    for bw in args.spectrum_bandwidths:
        plot_radial_spectrum(out_dir, methods, bw, args.spectrum_seed, fig_dir)

    print(f'\n[plot_bandlimited] Done → {fig_dir}')


if __name__ == '__main__':
    main()
