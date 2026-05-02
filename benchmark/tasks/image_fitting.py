"""
Image fitting task.

Per-image training loop: fit one INR to one image by minimising MSE over
all pixels.  Supports full-batch and random mini-batch modes.
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
        # decay to 0.1× at the end
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: 0.1 ** min(ep / n, 1.0))
    elif sched_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler '{sched_type}'")


def run(model, coords, pixels, meta, cfg, device, save_dir=None):
    """
    Train one INR on a single image.

    Args:
        model:   nn.Module (already on `device`)
        coords:  (N, 2) float tensor — pixel coordinates in [-1, 1]
        pixels:  (N, C) float tensor — ground truth pixel values in [0, 1]
        meta:    dict with 'H', 'W', 'C', 'name', 'path'
        cfg:     full config dict (from YAML)
        device:  torch.device
        save_dir: if given, images/curves are saved here

    Returns:
        dict with 'psnr_curve', 'ssim_curve', 'epochs_curve',
                  'final_psnr', 'final_ssim', 'total_time_s',
                  'model_state'
    """
    cfg_train = cfg['training']
    H, W, C = meta['H'], meta['W'], meta['C']
    N = H * W

    coords = coords.to(device)
    pixels = pixels.to(device)

    # ---- INCODE needs GT image ----
    if hasattr(model, 'set_gt'):
        # reshape pixels → (1, C, H, W) for the ResNet harmonizer
        gt_img = pixels.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)
        model.set_gt(gt_img)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg_train['lr'])
    scheduler = _make_scheduler(optimizer, cfg_train)

    num_epochs = cfg_train['num_epochs']
    batch_size = cfg_train.get('batch_size', -1)  # -1 = full image
    log_every = cfg_train.get('log_every', 100)
    save_every = cfg_train.get('save_every', 500)

    psnr_curve, ssim_curve, epochs_curve = [], [], []
    total_time = 0.0
    # tracked for logging only; weights saved are final, not best
    best_psnr = -float('inf')
    best_ssim = -float('inf')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if batch_size == -1 or batch_size >= N:
            # Full-batch
            pred = model(coords)
            loss = torch.mean((pred - pixels) ** 2)
        else:
            # Random mini-batch
            idx = torch.randperm(N, device=device)[:batch_size]
            pred = model(coords[idx])
            loss = torch.mean((pred - pixels[idx]) ** 2)

        if hasattr(model, 'aux_loss'):
            loss = loss + model.aux_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_time += time.time() - t0

        if epoch % log_every == 0 or epoch == num_epochs:
            with torch.no_grad():
                pred_full = model(coords)  # (N, C)

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
                  f"  PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}")

        # Optionally save intermediate images
        if save_dir is not None and epoch % save_every == 0:
            with torch.no_grad():
                pred_full = model(coords).clamp(0, 1)
            _save_image(pred_full, H, W, C,
                        os.path.join(save_dir, f"{meta['name']}_ep{epoch:05d}.png"))

    # Final evaluation — use the LAST-iter weights (not the best seen), so a
    # saved model can be resumed / fine-tuned from exactly where training stopped.
    final_state = {k: v.cpu() for k, v in model.state_dict().items()}
    with torch.no_grad():
        pred_full = model(coords).clamp(0, 1)
    pred_img = pred_full.reshape(H, W, C)
    gt_img_hw = pixels.reshape(H, W, C)
    final_psnr = psnr(pred_img, gt_img_hw).item()
    final_ssim = ssim(pred_img.cpu(), gt_img_hw.cpu())

    if save_dir is not None:
        _save_image(pred_full, H, W, C,
                    os.path.join(save_dir, f"{meta['name']}_final.png"))

    return {
        'name':          meta['name'],
        'psnr_curve':    psnr_curve,
        'ssim_curve':    ssim_curve,
        'epochs_curve':  epochs_curve,
        'final_psnr':    final_psnr,      # PSNR of LAST-iter weights
        'final_ssim':    final_ssim,      # SSIM of LAST-iter weights
        'best_psnr':     best_psnr,       # max PSNR seen during training
        'best_ssim':     best_ssim,       # max SSIM seen during training
        'total_time_s':  total_time,
        'model_state':   final_state,     # FINAL iter weights (for resuming training)
    }


def _save_image(pred_flat, H, W, C, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = pred_flat.reshape(H, W, C).detach().cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
