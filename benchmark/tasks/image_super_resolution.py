"""
Image super-resolution task.

Take the full-resolution dataset image as the HR ground truth, build an LR
observation by average-pool downsampling, train the INR on the LR grid only,
then evaluate on the HR grid.

Config:
    training.scale_factor: integer downsample factor (default 4)
    training.eval_epoch:   start computing HR metrics after this many epochs
                           (default 0 = always).  INCODE uses 400 by default.

This mirrors INCODE's SR experiment, but generalises cleanly to all 9 methods
— they all accept (N,2) coords and return (N,C) RGB.
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from benchmark.metrics.image_metrics import psnr, ssim


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


def _build_lr(hr_img_nchw, scale):
    """hr_img_nchw: (1, C, H, W).  Return (lr_img_nchw, H_lr, W_lr)."""
    H_lr = hr_img_nchw.shape[-2] // scale
    W_lr = hr_img_nchw.shape[-1] // scale
    # Average-pool is a well-defined anti-aliasing downsampler; behaves like
    # INTER_AREA for integer scales.
    lr = F.avg_pool2d(hr_img_nchw, kernel_size=scale, stride=scale)
    # Trim HR to a multiple of `scale` so HR/LR are consistent
    return lr, H_lr, W_lr


def _grid_coords(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (H*W, 2)


def run(model, coords, pixels, meta, cfg, device, save_dir=None):
    """
    Args:
        coords:  (N_hr, 2) in [-1, 1] — HR grid (unused directly; rebuilt)
        pixels:  (N_hr, C) in [0, 1]  — HR ground truth
    """
    cfg_train = cfg['training']
    H, W, C = meta['H'], meta['W'], meta['C']

    scale = int(cfg_train.get('scale_factor', 4))
    assert scale >= 1, "scale_factor must be >= 1"

    # Build HR image tensor
    hr_img = pixels.to(device).reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    # Crop HR so H and W are divisible by `scale` (keeps LR/HR perfectly aligned)
    H_crop = (H // scale) * scale
    W_crop = (W // scale) * scale
    if (H_crop, W_crop) != (H, W):
        hr_img = hr_img[..., :H_crop, :W_crop]
    H, W = H_crop, W_crop

    lr_img, H_lr, W_lr = _build_lr(hr_img, scale)
    print(f"  SR: HR={H}x{W} → LR={H_lr}x{W_lr}  (scale={scale})")

    # Build flat tensors + coordinate grids
    hr_pixels = hr_img.squeeze(0).permute(1, 2, 0).reshape(-1, C)   # (H*W, C)
    lr_pixels = lr_img.squeeze(0).permute(1, 2, 0).reshape(-1, C)   # (Hl*Wl, C)
    hr_coords = _grid_coords(H, W, device)
    lr_coords = _grid_coords(H_lr, W_lr, device)

    # INCODE: harmonizer gets the LR image (what the model "sees")
    if hasattr(model, 'set_gt'):
        model.set_gt(lr_img)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['lr'])
    scheduler = _make_scheduler(optimizer, cfg_train)

    num_epochs = cfg_train['num_epochs']
    batch_size = cfg_train.get('batch_size', -1)
    log_every = cfg_train.get('log_every', 100)
    save_every = cfg_train.get('save_every', 500)
    eval_epoch = cfg_train.get('eval_epoch', 0)  # start HR eval after this

    N_lr = lr_coords.shape[0]
    psnr_curve, ssim_curve, epochs_curve = [], [], []
    total_time = 0.0
    best_psnr = -float('inf')
    best_ssim = -float('inf')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Training is always on the LR grid
        if batch_size == -1 or batch_size >= N_lr:
            pred = model(lr_coords)
            loss = torch.mean((pred - lr_pixels) ** 2)
        else:
            idx = torch.randperm(N_lr, device=device)[:batch_size]
            pred = model(lr_coords[idx])
            loss = torch.mean((pred - lr_pixels[idx]) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_time += time.time() - t0

        if epoch % log_every == 0 or epoch == num_epochs:
            # HR evaluation (may be deferred by eval_epoch)
            if epoch >= eval_epoch:
                with torch.no_grad():
                    pred_hr = _forward_all(model, hr_coords, batch_size, device)
                pred_img = pred_hr.clamp(0, 1).reshape(H, W, C)
                gt_img_hw = hr_pixels.reshape(H, W, C)
                psnr_val = psnr(pred_img, gt_img_hw).item()
                ssim_val = ssim(pred_img.cpu(), gt_img_hw.cpu())
            else:
                # Fall back to LR PSNR while HR eval is deferred
                with torch.no_grad():
                    pred_lr = model(lr_coords)
                pred_img_lr = pred_lr.clamp(0, 1).reshape(H_lr, W_lr, C)
                gt_lr_hw = lr_pixels.reshape(H_lr, W_lr, C)
                psnr_val = psnr(pred_img_lr, gt_lr_hw).item()
                ssim_val = ssim(pred_img_lr.cpu(), gt_lr_hw.cpu())

            psnr_curve.append(psnr_val)
            ssim_curve.append(ssim_val)
            epochs_curve.append(epoch)

            if epoch >= eval_epoch and psnr_val > best_psnr:
                best_psnr = psnr_val
            if epoch >= eval_epoch and ssim_val > best_ssim:
                best_ssim = ssim_val

            tag = 'HR' if epoch >= eval_epoch else 'LR'
            print(f"  [{meta['name']}] epoch {epoch:5d}/{num_epochs}"
                  f"  loss={loss.item():.6f}"
                  f"  PSNR({tag})={psnr_val:.2f}  SSIM={ssim_val:.4f}")

        if save_dir is not None and epoch % save_every == 0:
            with torch.no_grad():
                pred_hr = _forward_all(model, hr_coords, batch_size, device).clamp(0, 1)
            _save_image(pred_hr, H, W, C,
                        os.path.join(save_dir, f"{meta['name']}_ep{epoch:05d}.png"))

    # Use the LAST-iter weights so the saved model can be resumed / fine-tuned.
    final_state = {k: v.cpu() for k, v in model.state_dict().items()}
    with torch.no_grad():
        pred_hr = _forward_all(model, hr_coords, batch_size, device).clamp(0, 1)
    pred_img = pred_hr.reshape(H, W, C)
    gt_img_hw = hr_pixels.reshape(H, W, C)
    final_psnr = psnr(pred_img, gt_img_hw).item()
    final_ssim = ssim(pred_img.cpu(), gt_img_hw.cpu())

    if save_dir is not None:
        _save_image(pred_hr, H, W, C,
                    os.path.join(save_dir, f"{meta['name']}_final_HR.png"))
        _save_image(lr_pixels, H_lr, W_lr, C,
                    os.path.join(save_dir, f"{meta['name']}_input_LR.png"))
        _save_image(hr_pixels, H, W, C,
                    os.path.join(save_dir, f"{meta['name']}_gt_HR.png"))

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
        'scale_factor':  scale,
        'H_hr': H, 'W_hr': W,
        'H_lr': H_lr, 'W_lr': W_lr,
    }


def _forward_all(model, coords, batch_size, device):
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
