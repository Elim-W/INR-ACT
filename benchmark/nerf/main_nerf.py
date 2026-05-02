"""
NeRF training entry point — picks a network by --method and trains on
one Blender synthetic scene via torch-ngp's Trainer / NeRFDataset.

Usage:
    python -m benchmark.nerf.main_nerf data/blender_nerf/lego \\
        --method finer --iters 30000 -O

Outputs are written to:
    outputs/nerf/<scene>/<method>/
        config.json         # all CLI args
        metrics.json        # best_psnr + per-eval curve
        checkpoints/        # auto from torch-ngp Trainer
        log_<method>.txt    # auto from torch-ngp Trainer
        run/                # tensorboard, auto
        validation/         # val renders, auto
        results/            # test renders + .mp4 (if --test)
"""

from . import ensure_torch_ngp_importable
ensure_torch_ngp_importable()

import argparse
import json
import os
import sys

import numpy as np
import torch

from nerf.provider import NeRFDataset
from nerf.utils import Trainer, PSNRMeter, SSIMMeter, LPIPSMeter, seed_everything


METHODS = ['finer', 'siren', 'gauss', 'relu_pe']


def build_model(method, opt):
    """Import the right NeRFNetwork lazily and instantiate it."""
    common = dict(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        num_layers=opt.num_layers,
        hidden_dim=opt.hidden_dim,
        geo_feat_dim=opt.geo_feat_dim,
        num_layers_color=opt.num_layers_color,
        hidden_dim_color=opt.hidden_dim_color,
    )
    if method == 'finer':
        from .network_finer import NeRFNetwork
        return NeRFNetwork(fw0=opt.fw0, hw0=opt.hw0, fbs=opt.fbs, **common)
    if method == 'siren':
        from .network_siren import NeRFNetwork
        return NeRFNetwork(fw0=opt.fw0, hw0=opt.hw0, **common)
    if method == 'gauss':
        from .network_gauss import NeRFNetwork
        return NeRFNetwork(gauss_scale=opt.gauss_scale, **common)
    if method == 'relu_pe':
        from .network_relu_pe import NeRFNetwork
        return NeRFNetwork(
            multires_x=opt.multires_x, multires_dir=opt.multires_dir, **common)
    raise ValueError(f"unknown method: {method}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('path', type=str,
                   help="path to scene directory (e.g. data/blender_nerf/lego)")
    p.add_argument('--method', required=True, choices=METHODS)
    p.add_argument('--workspace', type=str, default=None,
                   help="output dir (default: outputs/nerf/<scene>/<method>/)")
    p.add_argument('--seed', type=int, default=0)

    # Mode
    p.add_argument('--test', action='store_true',
                   help="skip training, evaluate latest ckpt on test split")
    p.add_argument('-O', action='store_true',
                   help="shortcut: --fp16 --cuda_ray --preload")

    # Architecture
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--geo_feat_dim', type=int, default=128)
    p.add_argument('--num_layers_color', type=int, default=4)
    p.add_argument('--hidden_dim_color', type=int, default=128)

    # Method-specific
    p.add_argument('--fw0', type=float, default=30,
                   help="first-layer omega_0 (siren/finer)")
    p.add_argument('--hw0', type=float, default=30,
                   help="hidden-layer omega_0 (siren/finer)")
    p.add_argument('--fbs', type=float, default=None,
                   help="first-layer bias scale (finer)")
    p.add_argument('--gauss_scale', type=float, default=10.0,
                   help="Gaussian envelope scale (gauss)")
    p.add_argument('--multires_x', type=int, default=10,
                   help="positional encoding frequencies for x (relu_pe)")
    p.add_argument('--multires_dir', type=int, default=4,
                   help="positional encoding frequencies for dir (relu_pe)")

    # Training
    p.add_argument('--iters', type=int, default=30000)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--ckpt', type=str, default='latest')
    p.add_argument('--num_rays', type=int, default=4096)
    p.add_argument('--max_ray_batch', type=int, default=4096)
    p.add_argument('--patch_size', type=int, default=1)
    p.add_argument('--eval_interval', type=int, default=10,
                   help="run eval + save 'best' ckpt every N epochs")
    p.add_argument('--eval_lpips_ssim', action='store_true',
                   help="also compute SSIM+LPIPS at eval time (slower)")

    # Raymarching
    p.add_argument('--cuda_ray', action='store_true')
    p.add_argument('--max_steps', type=int, default=1024)
    p.add_argument('--num_steps', type=int, default=512)
    p.add_argument('--upsample_steps', type=int, default=0)
    p.add_argument('--update_extra_interval', type=int, default=16)

    # Dataset / scene
    p.add_argument('--downscale', type=int, default=1)
    p.add_argument('--trainskip', type=int, default=1)
    p.add_argument('--color_space', type=str, default='srgb',
                   choices=['linear', 'srgb'])
    p.add_argument('--preload', action='store_true')
    p.add_argument('--bound', type=float, default=1.0)
    p.add_argument('--scale', type=float, default=0.8)
    p.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0])
    p.add_argument('--dt_gamma', type=float, default=0.0)
    p.add_argument('--min_near', type=float, default=0.2)
    p.add_argument('--density_thresh', type=float, default=10.0)
    p.add_argument('--bg_radius', type=float, default=-1)

    # Misc (forwarded to NeRFDataset / Trainer for compat)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--rand_pose', type=int, default=-1)
    p.add_argument('--error_map', action='store_true')

    args = p.parse_args()
    if args.O:
        args.fp16 = True
        args.cuda_ray = True
        args.preload = True
    return args


def derive_workspace(opt):
    if opt.workspace:
        return opt.workspace
    scene = os.path.basename(os.path.normpath(opt.path))
    return os.path.join('outputs', 'nerf', scene, opt.method)


def write_config(workspace, opt):
    cfg_path = os.path.join(workspace, 'config.json')
    cfg = {k: v for k, v in vars(opt).items()}
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2, default=str)


def write_metrics(workspace, trainer, total_time, n_params):
    """
    Pull PSNR curve from trainer.stats. With use_loss_as_metric=False and
    best_mode='max', torch-ngp stores -PSNR in stats['results'] (utils.py L1000).
    """
    results = [r for r in trainer.stats.get('results', [])
               if isinstance(r, (int, float))]
    psnr_curve = [-r for r in results]
    best_psnr = float(max(psnr_curve)) if psnr_curve else float('nan')

    metrics = {
        'method':         os.path.basename(workspace),
        'scene':          os.path.basename(os.path.dirname(workspace)),
        'best_psnr':      best_psnr,
        'final_psnr':     float(psnr_curve[-1]) if psnr_curve else float('nan'),
        'psnr_curve':     psnr_curve,
        'valid_loss':     trainer.stats.get('valid_loss', []),
        'train_loss':     trainer.stats.get('loss', []),
        'num_params':     n_params,
        'epochs_trained': trainer.epoch,
        'total_time_s':   total_time,
    }
    out = os.path.join(workspace, 'metrics.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    import time
    opt = parse_args()
    seed_everything(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    workspace = derive_workspace(opt)
    os.makedirs(workspace, exist_ok=True)
    write_config(workspace, opt)

    model = build_model(opt.method, opt)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main_nerf] method={opt.method}  scene={os.path.basename(opt.path)}  "
          f"params={n_params:,}  workspace={workspace}")

    criterion = torch.nn.MSELoss(reduction='none')
    metrics = [PSNRMeter()]
    if opt.eval_lpips_ssim:
        metrics += [SSIMMeter(device=device), LPIPSMeter(device=device)]

    if opt.test:
        trainer = Trainer(
            opt.method, opt, model, device=device, workspace=workspace,
            criterion=criterion, fp16=opt.fp16, metrics=metrics,
            use_checkpoint=opt.ckpt,
            use_loss_as_metric=False, best_mode='max',
        )
        test_loader = NeRFDataset(opt, device=device, type='test',
                                  downscale=opt.downscale).dataloader()
        if test_loader.has_gt:
            trainer.evaluate(test_loader)
        trainer.test(test_loader, write_video=True)
        return

    optimizer = lambda m: torch.optim.Adam(
        m.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda o: torch.optim.lr_scheduler.LambdaLR(
        o, lambda it: 0.1 ** min(it / opt.iters, 1))

    trainer = Trainer(
        opt.method, opt, model, device=device, workspace=workspace,
        optimizer=optimizer, criterion=criterion, ema_decay=0.95,
        fp16=opt.fp16, lr_scheduler=scheduler,
        scheduler_update_every_step=True, metrics=metrics,
        use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval,
        use_loss_as_metric=False, best_mode='max',
    )

    train_loader = NeRFDataset(opt, device=device, type='train',
                               downscale=opt.downscale,
                               train_skip=opt.trainskip).dataloader()
    val_loader = NeRFDataset(opt, device=device, type='val',
                             downscale=opt.downscale).dataloader()

    max_epoch = int(np.ceil(opt.iters / max(1, len(train_loader))))
    t0 = time.time()
    trainer.train(train_loader, val_loader, max_epoch)
    total_time = time.time() - t0

    summary = write_metrics(workspace, trainer, total_time, n_params)
    print(f"[main_nerf] DONE  best_psnr={summary['best_psnr']:.3f}  "
          f"final_psnr={summary['final_psnr']:.3f}  time={total_time:.1f}s")


if __name__ == '__main__':
    main()
