"""
Image CT reconstruction task.

Adapted from INCODE's train_ct_reconstruction.ipynb. The forward operator
is parallel-beam Radon at `proj` evenly-spaced angles in [0, 180]:
    sinogram = sum over y of rotate(image, theta).
Train an INR on coords -> grayscale, supervise by sinogram MSE; evaluate
PSNR/SSIM in the IMAGE domain against the ground-truth image.

Config:
    training.proj          number of projection angles   (default 150)
    training.scheduler     'cosine' / 'lambda' / 'none'  (default 'cosine')

Notes:
    - The INR must be configured with out_features=1.
    - If the dataset returns 3-channel pixels we convert to grayscale by
      taking the R channel (matches INCODE's `im[..., 0]`).
    - Radon transform is implemented in pure PyTorch (affine_grid +
      grid_sample) — no kornia dependency.
"""

import os
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from benchmark.metrics.image_metrics import psnr, ssim


# ---------------------------------------------------------------------------
# Forward operator: parallel-beam Radon transform
# ---------------------------------------------------------------------------

def _radon(img, angles_deg):
    """
    img: (1, 1, H, W) tensor in [0, 1].
    angles_deg: (n_angles,) in degrees.
    Returns sinogram (n_angles, W).

    Method: for each angle, rotate the image and sum along the height axis.
    Equivalent to kornia.geometry.rotate + .sum(2) used by INCODE.
    """
    n = angles_deg.numel()
    img_rep = img.expand(n, *img.shape[1:]).contiguous()    # (n, 1, H, W)

    theta = -angles_deg * (math.pi / 180.0)                 # negate = same as kornia
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    affine = torch.zeros(n, 2, 3, device=img.device, dtype=img.dtype)
    affine[:, 0, 0] = cos_t;  affine[:, 0, 1] = -sin_t
    affine[:, 1, 0] = sin_t;  affine[:, 1, 1] = cos_t

    grid = F.affine_grid(affine, img_rep.shape, align_corners=False)
    rotated = F.grid_sample(img_rep, grid, align_corners=False, mode='bilinear',
                            padding_mode='zeros')
    sinogram = rotated.sum(dim=2).squeeze(1)                # (n, W)
    return sinogram


# ---------------------------------------------------------------------------
# Scheduler
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
        coords: (N, 2) float32 in [-1, 1]
        pixels: (N, C) float32 in [0, 1]   — clean ground truth image
    """
    cfg_train = cfg['training']
    H, W, C = meta['H'], meta['W'], meta['C']

    coords = coords.to(device)
    pixels = pixels.to(device)

    # Take R channel as grayscale (matches INCODE)
    gt_image = pixels.reshape(H, W, C)[:, :, 0:1].permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)

    proj = int(cfg_train.get('proj', 150))
    thetas = torch.linspace(0.0, 180.0, proj, device=device)

    with torch.no_grad():
        sinogram_gt = _radon(gt_image, thetas)              # (proj, W)

    # INCODE harmonizer wants the SINOGRAM as its "GT" (it's what the model "sees")
    if hasattr(model, 'set_gt'):
        # Reshape sinogram to (1, 1, proj, W) image-like for the ResNet
        sg = sinogram_gt.unsqueeze(0).unsqueeze(0)
        # Normalize to [0,1] for the harmonizer (sum-of-rotations can exceed 1)
        sg_norm = (sg - sg.min()) / (sg.max() - sg.min() + 1e-8)
        model.set_gt(sg_norm.expand(1, 3, *sg_norm.shape[-2:]))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['lr'])
    scheduler = _make_scheduler(optimizer, cfg_train)

    num_epochs = cfg_train['num_epochs']
    log_every  = cfg_train.get('log_every', 100)
    save_every = cfg_train.get('save_every', 500)

    psnr_curve, ssim_curve, epochs_curve = [], [], []
    total_time = 0.0
    best_psnr = -float('inf')
    best_ssim = -float('inf')

    print(f"  CT: {meta['name']}  H×W={H}×{W}  proj={proj}  iters={num_epochs}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        pred = model(coords)                                # (N, 1)
        pred_img = pred.reshape(1, 1, H, W)
        sinogram_pred = _radon(pred_img, thetas)
        loss = torch.mean((sinogram_pred - sinogram_gt) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_time += time.time() - t0

        if epoch % log_every == 0 or epoch == num_epochs:
            with torch.no_grad():
                pred_img_clamped = pred_img.clamp(0, 1)
                # Image-domain PSNR (INCODE-style: mse on the IMAGE, not sinogram)
                psnr_val = psnr(pred_img_clamped.squeeze(0).permute(1, 2, 0),
                                gt_image.squeeze(0).permute(1, 2, 0)).item()
                ssim_val = ssim(pred_img_clamped.squeeze(0).permute(1, 2, 0).cpu(),
                                gt_image.squeeze(0).permute(1, 2, 0).cpu())

            psnr_curve.append(psnr_val)
            ssim_curve.append(ssim_val)
            epochs_curve.append(epoch)

            if psnr_val > best_psnr: best_psnr = psnr_val
            if ssim_val > best_ssim: best_ssim = ssim_val

            print(f"  [{meta['name']}] epoch {epoch:5d}/{num_epochs}"
                  f"  loss={loss.item():.6f}"
                  f"  PSNR(img)={psnr_val:.2f}  SSIM={ssim_val:.4f}")

        if save_dir is not None and epoch % save_every == 0:
            with torch.no_grad():
                _save_image(pred_img.clamp(0, 1).squeeze().cpu(),
                            os.path.join(save_dir, f"{meta['name']}_ep{epoch:05d}.png"))

    # ---- final eval (LAST-iter weights, for resumability) ----
    final_state = {k: v.cpu() for k, v in model.state_dict().items()}
    with torch.no_grad():
        pred_img_clamped = pred_img.clamp(0, 1)
        gt_perm = gt_image.squeeze(0).permute(1, 2, 0)       # (H, W, 1)
        pred_perm = pred_img_clamped.squeeze(0).permute(1, 2, 0)
        final_psnr = psnr(pred_perm, gt_perm).item()
        final_ssim = ssim(pred_perm.cpu(), gt_perm.cpu())

    if save_dir is not None:
        _save_image(pred_img_clamped.squeeze().cpu(),
                    os.path.join(save_dir, f"{meta['name']}_final.png"))
        _save_image(gt_image.squeeze().cpu(),
                    os.path.join(save_dir, f"{meta['name']}_gt.png"))
        # Save sinograms too (visual sanity check)
        sg_gt_norm = (sinogram_gt - sinogram_gt.min()) / (sinogram_gt.max() - sinogram_gt.min() + 1e-8)
        _save_image(sg_gt_norm.cpu(),
                    os.path.join(save_dir, f"{meta['name']}_sinogram_gt.png"))

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
        'proj':          proj,
        'H': H, 'W': W,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_image(img_tensor, path):
    """img_tensor: 2D (H, W) or 3D (H, W, C) tensor in [0, 1]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = img_tensor.detach().cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, mode='L').save(path)
    else:
        Image.fromarray(arr).save(path)
