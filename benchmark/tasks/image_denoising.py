"""
Image denoising task.

Given a clean image I, form a noisy version I_n = I + noise, train an INR
on the noisy pixels, and evaluate PSNR/SSIM against the CLEAN ground truth.

Two noise models are supported (config: training.noise_type):
    'gaussian' (default)
        I_n = I + N(0, sigma^2)
        Controlled by training.noise_sigma (default 0.1 in [0,1] range).
    'poisson_gaussian'   (the realistic sensor model used by INCODE)
        Poisson(x * tau) / tau + N(0, sigma_readout^2)
        Controlled by training.noise_tau, training.noise_readout_snr.

The INR never sees the clean image — supervision is the noisy image.
This mirrors INCODE/WIRE denoising experiments.
"""

import os
import time
import torch
import numpy as np
from PIL import Image

from benchmark.metrics.image_metrics import psnr, ssim


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

def _add_gaussian_noise(clean, sigma, generator=None):
    """clean: (N, C) in [0,1].  Returns noisy tensor (same shape, unclamped)."""
    noise = torch.randn(clean.shape, generator=generator, device=clean.device) * sigma
    return clean + noise


def _add_poisson_gaussian_noise(clean, tau, readout_snr, generator=None):
    """
    Realistic sensor noise (INCODE-style).
        x_meas = Poisson(clean * tau) / tau + N(0, readout_snr^2) / tau
    Poisson sampling is done via numpy for correctness; the Gaussian readout
    is added in torch so the RNG remains controllable via `generator`.
    """
    clean_np = clean.detach().cpu().numpy()
    scaled = clean_np * tau
    pos = scaled > 0
    sampled = np.zeros_like(scaled)
    sampled[pos] = np.random.poisson(scaled[pos])
    sampled[~pos] = -np.random.poisson(-scaled[~pos])
    readout = torch.randn(clean.shape, generator=generator) * readout_snr
    noisy_np = (sampled + readout.numpy()) / tau
    return torch.from_numpy(noisy_np.astype(np.float32)).to(clean.device)


def _make_noisy(clean, cfg_train):
    ntype = cfg_train.get('noise_type', 'gaussian')
    seed = cfg_train.get('noise_seed', None)
    g = None
    if seed is not None:
        g = torch.Generator(device='cpu').manual_seed(int(seed))
    if ntype == 'gaussian':
        sigma = cfg_train.get('noise_sigma', 0.1)
        return _add_gaussian_noise(clean, sigma, generator=g), \
               {'noise_type': 'gaussian', 'sigma': sigma}
    elif ntype == 'poisson_gaussian':
        tau = cfg_train.get('noise_tau', 40.0)
        readout = cfg_train.get('noise_readout_snr', 2.0)
        return _add_poisson_gaussian_noise(clean, tau, readout, generator=g), \
               {'noise_type': 'poisson_gaussian',
                'tau': tau, 'readout_snr': readout}
    else:
        raise ValueError(f"Unknown noise_type '{ntype}'")


# ---------------------------------------------------------------------------
# Scheduler (copy from image_fitting for consistency)
# ---------------------------------------------------------------------------

def _make_scheduler(optimizer, cfg_train):
    sched_type = cfg_train.get('scheduler', 'cosine')
    n = cfg_train['num_epochs']
    if sched_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n, eta_min=0)
    elif sched_type == 'lambda':
        decay = cfg_train.get('lambda_decay', 0.1)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: decay ** min(ep / n, 1.0))
    elif sched_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler '{sched_type}'")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run(model, coords, pixels, meta, cfg, device, save_dir=None):
    """
    Args:
        coords:  (N, 2) in [-1, 1]
        pixels:  (N, C) in [0, 1]      -- clean ground truth
    Returns same dict shape as image_fitting.run().
    """
    cfg_train = cfg['training']
    H, W, C = meta['H'], meta['W'], meta['C']
    N = H * W

    coords = coords.to(device)
    clean = pixels.to(device)

    # Build noisy observation (used for training supervision)
    noisy, noise_info = _make_noisy(clean, cfg_train)
    noisy = noisy.to(device)

    noisy_psnr = psnr(noisy.clamp(0, 1).reshape(H, W, C),
                      clean.reshape(H, W, C)).item()
    print(f"  noise: {noise_info}  noisy-vs-clean PSNR={noisy_psnr:.2f} dB")

    # INCODE: feed NOISY image to the harmonizer (what it sees at inference)
    if hasattr(model, 'set_gt'):
        gt_img = noisy.clamp(0, 1).reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)
        model.set_gt(gt_img)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['lr'])
    scheduler = _make_scheduler(optimizer, cfg_train)

    num_epochs = cfg_train['num_epochs']
    batch_size = cfg_train.get('batch_size', -1)
    log_every = cfg_train.get('log_every', 100)
    save_every = cfg_train.get('save_every', 500)

    psnr_curve, ssim_curve, epochs_curve = [], [], []
    total_time = 0.0
    best_psnr = -float('inf')
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if batch_size == -1 or batch_size >= N:
            pred = model(coords)
            loss = torch.mean((pred - noisy) ** 2)
        else:
            idx = torch.randperm(N, device=device)[:batch_size]
            pred = model(coords[idx])
            loss = torch.mean((pred - noisy[idx]) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_time += time.time() - t0

        if epoch % log_every == 0 or epoch == num_epochs:
            with torch.no_grad():
                pred_full = model(coords)
            pred_img = pred_full.clamp(0, 1).reshape(H, W, C)
            clean_hw = clean.reshape(H, W, C)

            # Evaluate against CLEAN ground truth (that's the point of denoising)
            psnr_val = psnr(pred_img, clean_hw).item()
            ssim_val = ssim(pred_img.cpu(), clean_hw.cpu())

            psnr_curve.append(psnr_val)
            ssim_curve.append(ssim_val)
            epochs_curve.append(epoch)

            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            print(f"  [{meta['name']}] epoch {epoch:5d}/{num_epochs}"
                  f"  loss={loss.item():.6f}"
                  f"  PSNR(vs clean)={psnr_val:.2f}  SSIM={ssim_val:.4f}")

        if save_dir is not None and epoch % save_every == 0:
            with torch.no_grad():
                pred_full = model(coords).clamp(0, 1)
            _save_image(pred_full, H, W, C,
                        os.path.join(save_dir, f"{meta['name']}_ep{epoch:05d}.png"))

    # Final eval from best checkpoint
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred_full = model(coords).clamp(0, 1)
    pred_img = pred_full.reshape(H, W, C)
    clean_hw = clean.reshape(H, W, C)
    final_psnr = psnr(pred_img, clean_hw).item()
    final_ssim = ssim(pred_img.cpu(), clean_hw.cpu())

    if save_dir is not None:
        _save_image(pred_full, H, W, C,
                    os.path.join(save_dir, f"{meta['name']}_best.png"))
        _save_image(noisy.clamp(0, 1), H, W, C,
                    os.path.join(save_dir, f"{meta['name']}_noisy.png"))
        _save_image(clean, H, W, C,
                    os.path.join(save_dir, f"{meta['name']}_clean.png"))

    return {
        'name':          meta['name'],
        'psnr_curve':    psnr_curve,
        'ssim_curve':    ssim_curve,
        'epochs_curve':  epochs_curve,
        'final_psnr':    final_psnr,
        'final_ssim':    final_ssim,
        'total_time_s':  total_time,
        'model_state':   best_state,
        'noise_info':    noise_info,
        'noisy_input_psnr': noisy_psnr,
    }


def _save_image(pred_flat, H, W, C, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = pred_flat.reshape(H, W, C).detach().cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
