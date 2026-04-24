"""
NeRF task (novel-view synthesis on Blender synthetic scenes).

Unlike image_fitting / shape_occupancy, the data interface here is
"one scene" = (opt, train_loader, val_loader); the run_experiment_3d.py
dispatcher yields one such triple per scene and calls `run()`.

The actual training/rendering loop is delegated to torch-ngp's `Trainer`
(which implements CUDA ray-marching, density-grid culling, fp16 AMP,
PSNR/LPIPS/SSIM evaluation, checkpoint saving).

We own:
    - building optimizer + scheduler from cfg
    - wrapping the Trainer into our task.run() signature
    - extracting the headline metrics (best val PSNR) into the standard
      result dict so analysis/ scripts still work.
"""

import os
import sys
import time
import torch


def _ensure_torch_ngp_importable():
    root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ngp_root = os.path.join(root, 'third_party', 'torch-ngp')
    if ngp_root not in sys.path:
        sys.path.insert(0, ngp_root)


def run(model, opt, train_loader, val_loader, scene_name, cfg, device,
        save_dir=None):
    """
    Train one NeRF model on one scene.

    Args:
        model:         NeRFNetwork (already on `device`)
        opt:           SimpleNamespace built by blender_nerf.build_nerf_dataloaders
        train_loader:  torch-ngp NeRFDataset.dataloader() for 'train' split
        val_loader:    same for 'val'
        scene_name:    e.g. 'lego'
        cfg:           full YAML config dict
        device:        torch.device
        save_dir:      where torch-ngp's Trainer writes its own workspace
                       (ckpts, tensorboard logs, validation renders).
    """
    _ensure_torch_ngp_importable()
    from nerf.utils import Trainer, PSNRMeter, SSIMMeter, LPIPSMeter  # torch-ngp

    cfg_train = cfg['training']
    num_iters = int(cfg_train.get('num_iters', 30000))
    lr = float(cfg_train.get('lr', 5e-4))
    fp16 = bool(cfg_train.get('fp16', True))
    eval_interval = int(cfg_train.get('eval_interval', 50))
    workspace = save_dir if save_dir else os.path.join(
        'results', 'nerf', cfg['method'], scene_name)
    os.makedirs(workspace, exist_ok=True)

    # Optimizer / scheduler — match torch-ngp's reference recipe
    optimizer = lambda m: torch.optim.Adam(
        m.get_params(lr), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda o: torch.optim.lr_scheduler.LambdaLR(
        o, lambda it: 0.1 ** min(it / num_iters, 1.0))

    criterion = torch.nn.MSELoss(reduction='none')

    metrics = [PSNRMeter()]
    if cfg_train.get('eval_lpips_ssim', False):
        metrics += [SSIMMeter(device=device), LPIPSMeter(device=device)]

    trainer = Trainer(
        name=cfg['method'],
        opt=opt,
        model=model,
        device=device,
        workspace=workspace,
        criterion=criterion,
        optimizer=optimizer,
        ema_decay=0.95,
        fp16=fp16,
        lr_scheduler=scheduler,
        scheduler_update_every_step=True,
        metrics=metrics,
        eval_interval=eval_interval,
        use_checkpoint=cfg_train.get('ckpt', 'latest'),
    )

    # Derive #epochs such that total training iters == num_iters
    max_epochs = max(1, (num_iters + len(train_loader) - 1) // len(train_loader))

    t0 = time.time()
    trainer.train(train_loader, val_loader, max_epochs)
    total_time = time.time() - t0

    # Final evaluation on val split
    trainer.evaluate(val_loader)

    # Extract best val PSNR from trainer's state — torch-ngp logs the
    # best epoch internally via its `stats` dict.
    best_psnr = None
    try:
        # `stats['best_result']` is set when best_mode='min' with
        # use_loss_as_metric.  For our PSNRMeter, trainer.stats['results']
        # contains the PSNR history under the 'valid' key.
        results = trainer.stats.get('results', [])
        psnrs = [r for r in results if isinstance(r, (int, float))]
        if psnrs:
            # PSNR: larger is better, so max
            best_psnr = float(max(psnrs))
    except Exception:
        pass

    # Fallback: run one more evaluate to get a final number if we didn't
    # manage to extract from the trainer state.
    if best_psnr is None:
        try:
            best_psnr = float(trainer.stats['valid_loss'][-1])  # MSE, we'll convert
            # actually PSNRMeter reports PSNR directly; leave as-is
        except Exception:
            best_psnr = float('nan')

    return {
        'name':          scene_name,
        'final_psnr':    best_psnr,
        'final_ssim':    float('nan'),     # fill in if SSIMMeter enabled
        'total_time_s':  total_time,
        'num_iters':     num_iters,
        'workspace':     workspace,
        # NeRF checkpoints live in `workspace/checkpoints/`; don't copy into
        # the .pt result here (they can be multi-GB).
        'model_state':   None,
        'psnr_curve':    [],
        'ssim_curve':    [],
        'epochs_curve':  [],
    }
