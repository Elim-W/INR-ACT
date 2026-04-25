"""
Image inpainting task.

Randomly drop a fraction of pixels.  Train the INR only on KEPT pixels and
evaluate PSNR/SSIM against the full ground-truth image (i.e. the model must
hallucinate the missing pixels).

Config:
    training.sampling_ratio: fraction of pixels kept for training (default 0.2).
    training.mask_seed: optional int for reproducible mask.

This mirrors INCODE's inpainting experiment (random pixel dropout).
"""

import os
import time
import torch
import numpy as np
from PIL import Image

from benchmark.metrics.image_metrics import psnr, ssim


def _make_scheduler(optimizer, cfg_train):
    sched_type = cfg_train.get('scheduler', 'cosine')
    n = cfg_train['num_epochs']
    if sched_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n, eta_min=0)
    elif sched_type == 'lambda':
        decay = cfg_train.get('lambda_decay', 0.25)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: decay ** min(ep / n, 1.0))
    elif sched_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler '{sched_type}'")


def _make_mask(N, sampling_ratio, seed=None, device='cpu'):
    """Return (keep_idx, mask_bool) where keep_idx are the training indices."""
    n_keep = int(round(N * sampling_ratio))
    n_keep = max(1, min(n_keep, N))
    if seed is not None:
        g = torch.Generator(device='cpu').manual_seed(int(seed))
        perm = torch.randperm(N, generator=g)
    else:
        perm = torch.randperm(N)
    keep_idx = perm[:n_keep].to(device)
    mask_bool = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bool[keep_idx] = True
    return keep_idx, mask_bool


def run(model, coords, pixels, meta, cfg, device, save_dir=None):
    """
    Args:
        coords:  (N, 2) in [-1, 1]
        pixels:  (N, C) in [0, 1]      -- full ground truth
    """
    cfg_train = cfg['training']
    H, W, C = meta['H'], meta['W'], meta['C']
    N = H * W

    coords = coords.to(device)
    pixels = pixels.to(device)

    sampling_ratio = cfg_train.get('sampling_ratio', 0.2)
    mask_seed = cfg_train.get('mask_seed', None)
    keep_idx, mask_bool = _make_mask(N, sampling_ratio,
                                     seed=mask_seed, device=device)

    train_coords = coords[keep_idx]          # (M, 2)
    train_pixels = pixels[keep_idx]          # (M, C)
    M = keep_idx.numel()
    print(f"  mask: keeping {M}/{N} pixels ({100*M/N:.1f}%)")

    # INCODE: harmonizer gets the PARTIAL observation as a 2D image (missing=0).
    # We reshape the masked pixels back into a full (1,C,H,W) tensor, zeroing
    # out the dropped ones — that's what the resnet harmonizer can consume.
    if hasattr(model, 'set_gt'):
        masked_img = torch.zeros_like(pixels)
        masked_img[keep_idx] = train_pixels
        gt_img = masked_img.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)
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
    best_ssim = -float('inf')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if batch_size == -1 or batch_size >= M:
            pred = model(train_coords)
            loss = torch.mean((pred - train_pixels) ** 2)
        else:
            sub = torch.randperm(M, device=device)[:batch_size]
            pred = model(train_coords[sub])
            loss = torch.mean((pred - train_pixels[sub]) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_time += time.time() - t0

        if epoch % log_every == 0 or epoch == num_epochs:
            with torch.no_grad():
                # Evaluate on ALL pixels (the inpainting metric)
                pred_full = _forward_all(model, coords, batch_size, device)
            pred_img = pred_full.clamp(0, 1).reshape(H, W, C)
            gt_img_hw = pixels.reshape(H, W, C)

            psnr_val = psnr(pred_img, gt_img_hw).item()
            ssim_val = ssim(pred_img.cpu(), gt_img_hw.cpu())

            psnr_curve.append(psnr_val)
            ssim_curve.append(ssim_val)
            epochs_curve.append(epoch)

            if psnr_val > best_psnr:
                best_psnr = psnr_val
            if ssim_val > best_ssim:
                best_ssim = ssim_val

            print(f"  [{meta['name']}] epoch {epoch:5d}/{num_epochs}"
                  f"  loss={loss.item():.6f}"
                  f"  PSNR(full)={psnr_val:.2f}  SSIM={ssim_val:.4f}")

        if save_dir is not None and epoch % save_every == 0:
            with torch.no_grad():
                pred_full = _forward_all(model, coords, batch_size, device).clamp(0, 1)
            _save_image(pred_full, H, W, C,
                        os.path.join(save_dir, f"{meta['name']}_ep{epoch:05d}.png"))

    # Use the LAST-iter weights so a saved model can be resumed / fine-tuned
    # from exactly where training stopped.
    final_state = {k: v.cpu() for k, v in model.state_dict().items()}
    with torch.no_grad():
        pred_full = _forward_all(model, coords, batch_size, device).clamp(0, 1)
    pred_img = pred_full.reshape(H, W, C)
    gt_img_hw = pixels.reshape(H, W, C)
    final_psnr = psnr(pred_img, gt_img_hw).item()
    final_ssim = ssim(pred_img.cpu(), gt_img_hw.cpu())

    if save_dir is not None:
        _save_image(pred_full, H, W, C,
                    os.path.join(save_dir, f"{meta['name']}_final.png"))
        # Also save the masked input for visual sanity
        masked_vis = torch.zeros_like(pixels)
        masked_vis[keep_idx] = train_pixels
        _save_image(masked_vis, H, W, C,
                    os.path.join(save_dir, f"{meta['name']}_masked.png"))

    return {
        'name':          meta['name'],
        'psnr_curve':    psnr_curve,
        'ssim_curve':    ssim_curve,
        'epochs_curve':  epochs_curve,
        'final_psnr':    final_psnr,
        'final_ssim':    final_ssim,
        'best_psnr':     best_psnr,
        'best_ssim':     best_ssim,
        'total_time_s':  total_time,
        'model_state':   final_state,
        'sampling_ratio': sampling_ratio,
        'n_train_pixels': M,
    }


def _forward_all(model, coords, batch_size, device):
    """Forward over all pixels, chunked if batch_size is a finite positive int."""
    N = coords.shape[0]
    if batch_size is None or batch_size == -1 or batch_size >= N:
        return model(coords)
    outs = []
    for i in range(0, N, batch_size):
        outs.append(model(coords[i:i + batch_size]))
    return torch.cat(outs, dim=0)


def _save_image(pred_flat, H, W, C, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = pred_flat.reshape(H, W, C).detach().cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
