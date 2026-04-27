"""
Unified synthetic signal fitting benchmark.

Supports all voilalab/INR-benchmark synthetic signal types:
  2d_bandlimited, 2d_sierpinski, 2d_sphere, 2d_startarget
  3d_bandlimited, 3d_sphere

Protocol:
  - 1000 gradient steps, eval every 100 steps
  - Results saved to results/<signal>/

Usage:
    python benchmark/run_synthetic.py --signal 2d_bandlimited
    python benchmark/run_synthetic.py --signal 2d_sierpinski --bandwidths 0.1 0.5 0.9
    python benchmark/run_synthetic.py --signal 2d_sphere --seeds 1234 2024
    python benchmark/run_synthetic.py --signal 3d_bandlimited --methods siren wire
    python benchmark/run_synthetic.py --signal 2d_startarget --methods siren finer
"""

import os
import sys
import csv
import time
import random
import argparse

import numpy as np
import numpy.fft as fft
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark.methods.models import get_INR, BENCHMARK_DEFAULTS, BENCHMARK_DEFAULTS_3D, TRAIN_KEYS
from benchmark.metrics.image_metrics import psnr, ssim


# ===========================================================================
# Signal generators
# ===========================================================================

# --------------- shared bandlimit helpers ----------------------------------

def _generate_bandlimits(start, stop, num_points, base):
    bl = (np.logspace(0, 1, num_points, base=base) - 1) / (base - 1)
    return bl * (stop - start) + start


def _bandlimit_filter(data, cutoff_low, cutoff_high):
    data_fft = fft.fftn(data)
    frequencies = [fft.fftfreq(n, d=1.0) for n in data.shape]
    grid = np.meshgrid(*frequencies, indexing='ij')
    radius = np.sqrt(np.sum(np.array(grid) ** 2, axis=0))
    mask = (cutoff_low / np.sqrt(2) <= radius) & (radius <= cutoff_high / np.sqrt(2))
    return np.real(fft.ifftn(data_fft * mask))


_CUTOFFS = _generate_bandlimits(0.0015, 0.7, 9, base=300)

# --------------- Fourier analysis helpers ----------------------------------

def _radial_freq_mask_nd(shape, r_low, r_high):
    grids = np.meshgrid(*[fft.fftfreq(n) for n in shape], indexing='ij')
    radius = np.sqrt(sum(g ** 2 for g in grids))
    return (r_low <= radius) & (radius < r_high)


def compute_freq_band_errors(gt_norm, pred):
    """
    MSE and relative error in low (0-20%), mid (20-50%), high (50-100%) Fourier bands.
    Works for N-D arrays (2D and 3D). max_r = sqrt(ndim)/2.
    rel_err = err / gt_energy normalises out the fact that high-freq GT energy is
    naturally smaller, so bands are comparable on the same scale.
    """
    max_r = np.sqrt(gt_norm.ndim) / 2
    gt_f  = fft.fftn(gt_norm)
    err_f = gt_f - fft.fftn(pred)
    out = {}
    for name, (r0, r1) in [('low',  (0.0,          0.20 * max_r)),
                            ('mid',  (0.20 * max_r, 0.50 * max_r)),
                            ('high', (0.50 * max_r, max_r + 1e-9))]:
        mask = _radial_freq_mask_nd(gt_norm.shape, r0, r1)
        n = mask.sum()
        gt_e  = float((np.abs(gt_f[mask]) ** 2).mean())  if n else 0.0
        err_e = float((np.abs(err_f[mask]) ** 2).mean()) if n else 0.0
        out[f'gt_energy_{name}'] = gt_e
        out[f'err_{name}']       = err_e
        out[f'rel_err_{name}']   = err_e / (gt_e + 1e-12)
    return out


def _compute_radial_spectrum_nd(arr, n_bins=200):
    """Radial mean power spectrum for N-D arrays. Returns (centers, mean_power)."""
    power = np.abs(fft.fftn(arr)) ** 2
    grids = np.meshgrid(*[fft.fftfreq(n) for n in arr.shape], indexing='ij')
    radius = np.sqrt(sum(g ** 2 for g in grids))
    max_r = np.sqrt(arr.ndim) / 2  # max radius for unit-sampled ND FFT
    bins = np.linspace(0, max_r, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    mean_power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (bins[i] <= radius) & (radius < bins[i + 1])
        if mask.any():
            mean_power[i] = power[mask].mean()
    return centers, mean_power


def compute_oob_leakage(pred, bandwidth_label):
    """
    Fraction of prediction's Fourier energy outside the GT bandwidth cutoff.
    Works for N-D arrays. GT was constructed with cutoff radius = _CUTOFFS[idx] / sqrt(2)
    (the sqrt(2) factor comes from _bandlimit_filter, which uses the same divisor for all dims).
    """
    idx = int(round(bandwidth_label * 10)) - 1
    cutoff_r = _CUTOFFS[idx] / np.sqrt(2)
    power = np.abs(fft.fftn(pred)) ** 2
    in_band = _radial_freq_mask_nd(pred.shape, 0.0, cutoff_r + 1e-10)
    return float(power[~in_band].sum() / (power.sum() + 1e-12))


# --------------- 2D bandlimited --------------------------------------------

def make_bandlimited_2d(length, bandwidth_label, seed):
    np.random.seed(seed)
    if length % 2 == 0:
        length += 1
    signal = np.random.uniform(size=(length, length))
    idx = int(round(bandwidth_label * 10)) - 1
    signal = _bandlimit_filter(signal, 0.0, _CUTOFFS[idx])
    return signal.astype(np.float32)


# --------------- 2D Sierpinski --------------------------------------------

def _sierpinski_tri(vertices, depth, ax):
    if depth == 0:
        ax.add_patch(plt.Polygon(vertices, edgecolor='black', facecolor='black'))
    else:
        mids = [(vertices[i] + vertices[(i + 1) % 3]) / 2 for i in range(3)]
        _sierpinski_tri([vertices[0], mids[0], mids[2]], depth - 1, ax)
        _sierpinski_tri([vertices[1], mids[0], mids[1]], depth - 1, ax)
        _sierpinski_tri([vertices[2], mids[1], mids[2]], depth - 1, ax)


def make_sierpinski_2d(bandwidth_label, **_):
    """bandwidth 0.1–0.9 → fractal depth 0–8.  Deterministic, seed ignored."""
    depth = int(round(bandwidth_label * 10)) - 1
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    ax.set_aspect('equal')
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    _sierpinski_tri(vertices, depth, ax)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.sqrt(3) / 2)
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(int(h), int(w), 4)
    plt.close(fig)
    img = img[..., :3].mean(axis=-1)
    img = img[100:1100, 115:1115]   # crop to 1000×1000
    img = (img > 150).astype(np.float32)
    return 1.0 - img                # triangles=1, bg=0


# --------------- 2D sphere ------------------------------------------------

def make_sphere_2d(length, bandwidth_label, seed, occupied_fraction=0.1):
    np.random.seed(seed)
    signal = np.zeros((length, length), dtype=np.float32)
    occupied_cells = occupied_fraction * length * length
    sphere_radius = length / (bandwidth_label * 100)
    num_spheres = max(1, int(occupied_cells / (np.pi * sphere_radius ** 2)))
    centers = np.random.uniform(0, length, size=(num_spheres, 2))
    ys, xs = np.mgrid[0:length, 0:length]
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    r2 = sphere_radius ** 2
    for c in centers:
        signal.ravel()[((coords - c) ** 2).sum(1) <= r2] = 1.0
    return signal


# --------------- 2D star target -------------------------------------------

def make_startarget_2d(num_triangles=40, img_size=1000, dst_edge=20, **_):
    """Deterministic — bandwidth/seed ignored."""
    n = num_triangles * 2
    center = img_size // 2
    radius = center - dst_edge
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ys, xs = np.mgrid[0:img_size, 0:img_size]
    dx = (xs - center).astype(np.float64)
    dy = (ys - center).astype(np.float64)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    palette = np.zeros((img_size, img_size), dtype=np.float32)
    for i in range(0, n, 2):
        x1 = radius * np.cos(angles[i]);   y1 = radius * np.sin(angles[i])
        x2 = radius * np.cos(angles[i+1]); y2 = radius * np.sin(angles[i+1])
        mask = (dist <= radius) & (np.sign(dx*y1 - dy*x1) >= 0) & (np.sign(dx*y2 - dy*x2) <= 0)
        palette[mask] = 1.0
    return palette


# --------------- 3D bandlimited -------------------------------------------

def make_bandlimited_3d(length, bandwidth_label, seed):
    np.random.seed(seed)
    if length % 2 == 0:
        length += 1
    signal = np.random.uniform(size=(length, length, length))
    idx = int(round(bandwidth_label * 10)) - 1
    signal = _bandlimit_filter(signal, 0.0, _CUTOFFS[idx])
    return signal.astype(np.float32)


# --------------- 3D sphere ------------------------------------------------

def make_sphere_3d(length, bandwidth_label, seed, occupied_fraction=0.1):
    np.random.seed(seed)
    signal = np.zeros((length, length, length), dtype=np.float32)
    occupied_cells = occupied_fraction * length ** 3
    sphere_radius = length / (bandwidth_label * 100)
    sphere_vol = (4.0 / 3.0) * np.pi * sphere_radius ** 3
    num_spheres = max(1, int(occupied_cells / sphere_vol))
    centers = np.random.uniform(0, length, size=(num_spheres, 3))
    zs, ys, xs = np.mgrid[0:length, 0:length, 0:length]
    coords = np.stack([zs.ravel(), ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    r2 = sphere_radius ** 2
    for c in centers:
        signal.ravel()[((coords - c) ** 2).sum(1) <= r2] = 1.0
    return signal


# ===========================================================================
# Signal registry
# ===========================================================================

# Each entry: (generator_fn, default_length, is_3d, has_bandwidth_sweep, description)
SIGNALS = {
    '2d_bandlimited': dict(
        fn=make_bandlimited_2d, length=1000, ndim=2,
        has_bw=True, fourier_bw=True, desc='2-D bandlimited noise (length=1000)',
    ),
    '2d_sierpinski': dict(
        fn=lambda length, bandwidth_label, seed: make_sierpinski_2d(bandwidth_label),
        length=1000, ndim=2, has_bw=True, fourier_bw=False,
        desc='2-D Sierpinski triangle (depth=0–8)',
    ),
    '2d_sphere': dict(
        fn=make_sphere_2d, length=1000, ndim=2,
        has_bw=True, fourier_bw=False, desc='2-D sparse circles (vectorized)',
    ),
    '2d_startarget': dict(
        fn=lambda length, bandwidth_label, seed: make_startarget_2d(img_size=length),
        length=1000, ndim=2, has_bw=False, fourier_bw=False,
        desc='2-D star resolution target (40 triangles)',
    ),
    '3d_bandlimited': dict(
        fn=make_bandlimited_3d, length=100, ndim=3,
        has_bw=True, fourier_bw=True, desc='3-D bandlimited noise (length=100)',
    ),
    '3d_sphere': dict(
        fn=make_sphere_3d, length=100, ndim=3,
        has_bw=True, fourier_bw=False, desc='3-D sparse spheres (vectorized)',
    ),
}


# Hyperparameters now live in models.py (BENCHMARK_DEFAULTS / BENCHMARK_DEFAULTS_3D / TRAIN_KEYS)


def _split_cfg(d):
    model_kw = {k: v for k, v in d.items() if k not in TRAIN_KEYS}
    train_kw = {k: v for k, v in d.items() if k in TRAIN_KEYS}
    return model_kw, train_kw


def _make_scheduler(optimizer, kind, num_iters):
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=0)
    elif kind == 'lambda':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 0.1 ** min(t / num_iters, 1.0))
    return None


# ===========================================================================
# Coordinate grids
# ===========================================================================

def make_coords_2d(H, W):
    """Returns (H*W, 2) tensor in [-1, 1]."""
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


def make_coords_3d(L):
    """Returns (L^3, 3) tensor in [-1, 1]."""
    t = torch.linspace(-1, 1, L)
    gz, gy, gx = torch.meshgrid(t, t, t, indexing='ij')
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


# ===========================================================================
# Training loop
# ===========================================================================

def _chunked_forward(model, coords, chunk_size):
    """Run model on coords in chunks to avoid OOM during eval."""
    outs = []
    for i in range(0, coords.shape[0], chunk_size):
        outs.append(model(coords[i:i + chunk_size]).squeeze(-1))
    return torch.cat(outs, dim=0)


def train_one(model, coords, signal_flat, signal_shape, train_cfg,
              num_iters, eval_every, device, save_dir, is_3d, batch_size=262144):
    """
    Train for num_iters steps.
    batch_size=0   → full-batch (only reasonable for very small signals)
    batch_size>0   → mini-batch training; eval uses 4× chunk (no grad → less memory)
    Returns (best_psnr, best_ssim_or_None, elapsed_s, iter_list, psnr_list, best_output).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    scheduler = _make_scheduler(optimizer, train_cfg.get('scheduler', 'none'), num_iters)
    loss_fn = torch.nn.MSELoss()
    best_psnr, best_state, best_output = -float('inf'), None, None
    iter_list, psnr_list = [], []
    t0 = time.time()
    N = coords.shape[0]
    use_minibatch = batch_size > 0 and batch_size < N
    # eval doesn't need gradients → can use a larger chunk without extra memory
    eval_chunk = min(N, batch_size * 4) if use_minibatch else N

    sig_min = signal_flat.min()
    sig_range = (signal_flat.max() - sig_min).clamp(min=1e-12)
    gt_norm = (signal_flat - sig_min) / sig_range

    for it in range(1, num_iters + 1):
        model.train()
        if use_minibatch:
            idx = torch.randint(0, N, (batch_size,), device=device)
            pred = model(coords[idx]).squeeze(-1)
            loss = loss_fn(pred, gt_norm[idx])
        else:
            pred = model(coords).squeeze(-1)
            loss = loss_fn(pred, gt_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if it % eval_every == 0 or it == num_iters:
            with torch.no_grad():
                model.eval()
                out = _chunked_forward(model, coords, eval_chunk)
                mse_val = loss_fn(out, gt_norm).item()
                psnr_val = -10.0 * np.log10(mse_val + 1e-12)
            iter_list.append(it)
            psnr_list.append(psnr_val)
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_output = out.detach().cpu().reshape(signal_shape).numpy()
                if save_dir:
                    _save_vis(best_output, os.path.join(save_dir, 'best_recon.png'), is_3d)

    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    if not is_3d:
        gt_img = gt_norm.cpu().reshape(signal_shape).numpy()
        ssim_val = ssim(
            torch.from_numpy(best_output).unsqueeze(-1),
            torch.from_numpy(gt_img).unsqueeze(-1),
        )
    else:
        ssim_val = None

    return best_psnr, ssim_val, elapsed, iter_list, psnr_list, best_output


# ===========================================================================
# Save helpers
# ===========================================================================

def _save_vis(arr, path, is_3d):
    """Save 2D image or middle slice of 3D volume, always min-max normalised."""
    img = arr[arr.shape[0] // 2] if is_3d else arr
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo + 1e-12)
    plt.imsave(path, img, cmap='gray', vmin=0, vmax=1)


def save_gt_image(signal_arr, path, is_3d):
    _save_vis(signal_arr, path, is_3d)


def save_psnr_curve(iter_list, psnr_list, best_psnr, path):
    plt.figure(figsize=(5, 3))
    plt.plot(iter_list, psnr_list, marker='o', markersize=3)
    lo = max(0, min(psnr_list) - 2)
    hi = best_psnr + 2
    plt.ylim([lo, hi])
    plt.xlabel('iteration')
    plt.ylabel('PSNR (dB)')
    plt.title(f'best PSNR: {best_psnr:.2f} dB')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ===========================================================================
# Main
# ===========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(
        description='Unified synthetic INR benchmark',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--signal', required=True, choices=list(SIGNALS.keys()),
                   help='Signal type:\n' + '\n'.join(
                       f'  {k}: {v["desc"]}' for k, v in SIGNALS.items()))
    p.add_argument('--methods', nargs='+', default=None,
                   help='Methods to run (default: all for the signal dimension)')
    p.add_argument('--bandwidths', nargs='+', type=float,
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument('--seeds', nargs='+', type=int, default=[1234])
    p.add_argument('--signal_length', type=int, default=None,
                   help='Override signal grid size per side')
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=65536,
                   help='Mini-batch size per training step (0 = full-batch, slow for large signals).')
    p.add_argument('--device', default='auto')
    p.add_argument('--out_dir', default=None,
                   help='Output directory (default: results/<signal>)')
    return p.parse_args()


def _load_signal_yaml(signal_name):
    """Load configs/experiments/synthetic/<signal>.yaml if it exists. Returns dict or {}."""
    yaml_path = os.path.join(
        _ROOT, 'configs', 'experiments', 'synthetic', f'{signal_name}.yaml')
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()

    cfg = SIGNALS[args.signal]
    is_3d = cfg['ndim'] == 3
    method_defaults = BENCHMARK_DEFAULTS_3D if is_3d else BENCHMARK_DEFAULTS

    # Load per-signal YAML config; CLI flags override YAML which overrides BENCHMARK_DEFAULTS
    yaml_cfg = _load_signal_yaml(args.signal)
    yaml_methods = yaml_cfg.get('methods', {})
    _ARG_DEFAULTS = {'iters': 1000, 'eval_every': 100, 'batch_size': 65536}
    if yaml_cfg:
        print(f'[config] loaded configs/experiments/synthetic/{args.signal}.yaml')
        # Apply top-level training overrides from YAML (only if not set via CLI)
        for key in ('iters', 'eval_every', 'batch_size'):
            if key in yaml_cfg and getattr(args, key) == _ARG_DEFAULTS[key]:
                setattr(args, key, yaml_cfg[key])

    # Build effective per-method defaults: BENCHMARK_DEFAULTS ← YAML override
    effective_defaults = {}
    for method, base in method_defaults.items():
        effective_defaults[method] = {**base, **(yaml_methods.get(method, {}))}

    if args.methods is None:
        args.methods = list(effective_defaults.keys())
    if args.out_dir is None:
        args.out_dir = f'results/{args.signal}'
    if args.signal_length is None:
        args.signal_length = cfg['length']

    # For signals without a bandwidth sweep, use a single placeholder
    bandwidths = args.bandwidths if cfg['has_bw'] else [None]

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    ndim = cfg['ndim']
    L = args.signal_length
    print(f'[run_synthetic] signal={args.signal}  device={device}  '
          f'size={L}{"³" if is_3d else "²"}  iters={args.iters}')
    os.makedirs(args.out_dir, exist_ok=True)

    # Resume: load already-completed runs
    summary_path = os.path.join(args.out_dir, 'results.csv')
    has_bw = cfg['has_bw']
    summary_fields = (['method', 'bandwidth', 'seed', 'psnr', 'time_s']
                      if has_bw else ['method', 'seed', 'psnr', 'time_s'])
    if not is_3d:
        summary_fields.insert(-1, 'ssim')   # insert ssim before time_s
    done = set()
    if os.path.exists(summary_path):
        with open(summary_path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['method'], float(row.get('bandwidth', 0)), int(row['seed']))
                done.add(key)

    iters_path = os.path.join(args.out_dir, 'iters_psnrs.csv')
    iters_fields = (['method', 'bandwidth', 'seed', 'iteration', 'psnr']
                    if has_bw else ['method', 'seed', 'iteration', 'psnr'])
    iters_exists = os.path.exists(iters_path)

    # Frequency analysis CSV (2D bandlimited only)
    freq_path = os.path.join(args.out_dir, 'freq_analysis.csv')
    freq_fields = ['method', 'bandwidth', 'seed',
                   'gt_energy_low', 'gt_energy_mid', 'gt_energy_high',
                   'err_low', 'err_mid', 'err_high',
                   'rel_err_low', 'rel_err_mid', 'rel_err_high',
                   'oob_leakage']
    freq_done = set()
    if os.path.exists(freq_path):
        with open(freq_path, newline='') as f:
            for row in csv.DictReader(f):
                freq_done.add((row['method'], float(row.get('bandwidth', 0)), int(row['seed'])))
    freq_exists = os.path.exists(freq_path)

    # Pre-generate deterministic signals (Sierpinski, StarTarget)
    sig_cache = {}
    if not cfg['has_bw']:
        print(f'  generating {args.signal} (deterministic) ...')
        sig_cache[None] = cfg['fn'](length=L, bandwidth_label=0.1, seed=args.seeds[0])

    with open(summary_path, 'a', newline='') as sum_f, \
         open(iters_path, 'a', newline='') as iter_f, \
         open(freq_path, 'a', newline='') as freq_f:

        sum_writer = csv.DictWriter(sum_f, fieldnames=summary_fields)
        iter_writer = csv.DictWriter(iter_f, fieldnames=iters_fields)
        freq_writer = csv.DictWriter(freq_f, fieldnames=freq_fields)
        if os.path.getsize(summary_path) == 0:
            sum_writer.writeheader()
        if not iters_exists:
            iter_writer.writeheader()
        if not freq_exists:
            freq_writer.writeheader()

        for method in args.methods:
            if method not in effective_defaults:
                print(f'  unknown method {method!r}, skipping')
                continue

            defaults = dict(effective_defaults[method])
            method_batch_size = defaults.pop('batch_size', args.batch_size)
            model_kw, train_cfg = _split_cfg(defaults)
            hf = model_kw.pop('hidden_features', 256)
            hl = model_kw.pop('hidden_layers', 3)

            for bw in bandwidths:
                for seed in args.seeds:
                    key = (method, bw if bw is not None else 0.0, seed)
                    if key in done:
                        print(f'  skip {method} bw={bw} seed={seed}')
                        continue

                    bw_str = f'bw{bw:.1f} ' if bw is not None else ''
                    print(f'\n[{method}]  {bw_str}seed={seed}')
                    set_seed(seed)

                    # Generate signal
                    if bw in sig_cache:
                        sig = sig_cache[bw]
                    else:
                        sig = cfg['fn'](length=L, bandwidth_label=bw, seed=seed)
                        if not cfg['has_bw']:
                            sig_cache[bw] = sig   # cache deterministic signals

                    signal_shape = sig.shape

                    # Coordinates
                    if is_3d:
                        coords = make_coords_3d(sig.shape[0]).to(device)
                    else:
                        H, W = sig.shape
                        coords = make_coords_2d(H, W).to(device)

                    signal_flat = torch.from_numpy(sig).reshape(-1).to(device)

                    # Output directory
                    bw_tag = f'bw{bw:.1f}_' if bw is not None else ''
                    run_dir = os.path.join(args.out_dir, method, f'{bw_tag}seed{seed}')
                    os.makedirs(run_dir, exist_ok=True)

                    # Save GT once
                    gt_img_path = os.path.join(run_dir, 'gt.png')
                    if not os.path.exists(gt_img_path):
                        save_gt_image(sig, gt_img_path, is_3d)

                    # Build model
                    model = get_INR(
                        method=method, in_features=ndim,
                        hidden_features=hf, hidden_layers=hl, out_features=1,
                        **model_kw,
                    ).to(device)
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f'  params={n_params:,}  lr={train_cfg["lr"]}')

                    # INCODE needs GT image for its Harmonizer (ResNet expects 3-ch 2D input)
                    if hasattr(model, 'set_gt') and not is_3d:
                        gt_t = torch.from_numpy(sig)
                        if gt_t.dim() == 2:
                            gt_t = gt_t[None, None].expand(1, 3, -1, -1)
                        model.set_gt(gt_t.to(device))

                    best_psnr, best_ssim, elapsed, iter_list, psnr_list, best_output = train_one(
                        model, coords, signal_flat, signal_shape,
                        train_cfg, args.iters, args.eval_every, device, run_dir, is_3d,
                        batch_size=method_batch_size,
                    )

                    save_psnr_curve(iter_list, psnr_list, best_psnr,
                                    os.path.join(run_dir, 'psnr_curve.png'))

                    # Save best reconstruction as numpy for later analysis
                    np.save(os.path.join(run_dir, 'best_recon.npy'), best_output)

                    # Save normalized GT alongside prediction (same value range, 2D and 3D)
                    gt_norm_np = ((sig - sig.min())
                                  / (sig.max() - sig.min() + 1e-12))
                    gt_npy = os.path.join(run_dir, 'gt_norm.npy')
                    if not os.path.exists(gt_npy):
                        np.save(gt_npy, gt_norm_np)

                    ssim_str = f'  SSIM={best_ssim:.4f}' if best_ssim is not None else ''
                    print(f'  → PSNR={best_psnr:.2f}{ssim_str}  time={elapsed:.1f}s')

                    # Write summary row
                    row = {'method': method, 'seed': seed,
                           'psnr': f'{best_psnr:.4f}', 'time_s': f'{elapsed:.1f}'}
                    if has_bw:
                        row['bandwidth'] = bw
                    if not is_3d:
                        row['ssim'] = f'{best_ssim:.4f}'
                    sum_writer.writerow(row)
                    sum_f.flush()

                    for it, pv in zip(iter_list, psnr_list):
                        irow = {'method': method, 'seed': seed,
                                'iteration': it, 'psnr': f'{pv:.4f}'}
                        if has_bw:
                            irow['bandwidth'] = bw
                        iter_writer.writerow(irow)
                    iter_f.flush()

                    # Frequency analysis (signals with bandwidth only)
                    if bw is not None and key not in freq_done:
                        fm = compute_freq_band_errors(gt_norm_np, best_output)
                        # OOB leakage is only meaningful when bandwidth_label maps
                        # to a Fourier cutoff (bandlimited signals); for sphere /
                        # sierpinski the label is not a frequency parameter.
                        if cfg['fourier_bw']:
                            fm['oob_leakage'] = compute_oob_leakage(best_output, bw)
                        else:
                            fm['oob_leakage'] = float('nan')
                        frow = {'method': method, 'bandwidth': bw, 'seed': seed}
                        for k, v in fm.items():
                            frow[k] = f'{v:.6e}'
                        freq_writer.writerow(frow)
                        freq_f.flush()

                        # Radial power spectrum: GT / pred / residual (ND)
                        centers, gt_power   = _compute_radial_spectrum_nd(gt_norm_np)
                        _,       pred_power = _compute_radial_spectrum_nd(best_output)
                        residual = gt_norm_np - best_output
                        _,       res_power  = _compute_radial_spectrum_nd(residual)
                        np.savez(os.path.join(run_dir, 'radial_spectrum.npz'),
                                 centers=centers,
                                 gt_power=gt_power,
                                 pred_power=pred_power,
                                 residual_power=res_power)

    print(f'\n[run_synthetic] Done → {args.out_dir}')


if __name__ == '__main__':
    main()
