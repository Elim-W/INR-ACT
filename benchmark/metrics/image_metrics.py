import torch
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


def mse(pred, target):
    """Mean squared error between two tensors."""
    return torch.mean((pred - target) ** 2)


def psnr(pred, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio (dB).
    pred, target: float tensors in [0, max_val].
    """
    mse_val = mse(pred, target)
    if mse_val == 0:
        return torch.tensor(float('inf'))
    return 10.0 * torch.log10(max_val ** 2 / mse_val)


def ssim(pred, target):
    """
    Structural Similarity Index (SSIM).
    pred, target: (H, W, C) or (H, W) numpy arrays in [0, 1],
                  or (N, C)/(H, W, C) torch tensors → converted internally.
    Returns scalar float.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.clip(0, 1)
    target = target.clip(0, 1)

    if pred.ndim == 2:
        return skimage_ssim(pred, target, data_range=1.0)
    # (H, W, C) → channel_axis=-1
    return skimage_ssim(pred, target, data_range=1.0, channel_axis=-1)


def compute_all(pred, target, max_val=1.0):
    """
    Compute PSNR and SSIM together.

    Args:
        pred, target: (H, W, C) tensors/arrays in [0, max_val]
    Returns:
        dict with keys 'psnr' (float), 'ssim' (float), 'mse' (float)
    """
    if isinstance(pred, np.ndarray):
        pred_t = torch.from_numpy(pred).float()
        target_t = torch.from_numpy(target).float()
    else:
        pred_t = pred.float()
        target_t = target.float()

    psnr_val = psnr(pred_t / max_val, target_t / max_val).item()

    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
    else:
        pred_np, target_np = pred, target

    ssim_val = ssim(pred_np / max_val, target_np / max_val)
    mse_val = mse(pred_t / max_val, target_t / max_val).item()

    return {'psnr': psnr_val, 'ssim': ssim_val, 'mse': mse_val}
