"""
Thin wrapper around torch-ngp's NeRFDataset.

Usage:
    from benchmark.datasets.blender_nerf import build_nerf_dataloaders

    for scene, opt, train_loader, val_loader in build_nerf_dataloaders(cfg, device):
        ...

`cfg` is the full YAML config dict that run_experiment_nerf.py loaded.
The returned `opt` is the SimpleNamespace that torch-ngp's Trainer and
NeRFRenderer's `.run_cuda()` both consume.

Expected data layout (Blender synthetic):
    data/blender_nerf/<scene>/
        transforms_train.json
        transforms_val.json
        transforms_test.json
        train/   r_*.png
        val/
        test/
"""

import os
import sys
import types


def _ensure_torch_ngp_importable():
    root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ngp_root = os.path.join(root, 'third_party', 'torch-ngp')
    if ngp_root not in sys.path:
        sys.path.insert(0, ngp_root)


# Default opt values — cover every attribute NeRFDataset / Trainer reads.
# Anything the user sets in cfg['dataset_kwargs'] or cfg['training']
# will override the default.
_NGP_OPT_DEFAULTS = dict(
    path=None,                # set from cfg['data_root'] + scene
    preload=True,             # load all data into GPU RAM (fast, uses more VRAM)
    scale=0.8,                # blender: 0.8 is typical
    offset=[0, 0, 0],
    bound=1.0,                # scene bounding box half-length
    fp16=True,
    downscale=1,              # 1 = full res; 4 = 200×200 (WIRE/FINER protocol)
    rand_pose=-1,
    num_rays=4096,            # rays per training step
    error_map=False,
    color_space='srgb',
    patch_size=1,
    trainskip=1,
    cuda_ray=True,
    max_steps=1024,
    num_steps=128,
    upsample_steps=0,
    update_extra_interval=16,
    max_ray_batch=4096,
    dt_gamma=0.0,
    min_near=0.2,
    density_thresh=10.0,
    bg_radius=-1,
)


def _make_opt(cfg, data_path):
    """Build a SimpleNamespace that NeRFDataset / Trainer expect."""
    opt = types.SimpleNamespace(**_NGP_OPT_DEFAULTS)
    opt.path = data_path

    # Allow overrides from cfg['training'] (raymarching knobs) and
    # cfg['dataset_kwargs'] (data-side knobs).  Unknown keys are still
    # attached — harmless if NeRFDataset ignores them.
    for k, v in (cfg.get('training', {}) or {}).items():
        if hasattr(opt, k):
            setattr(opt, k, v)
    for k, v in (cfg.get('dataset_kwargs', {}) or {}).items():
        setattr(opt, k, v)

    return opt


def build_nerf_dataloaders(cfg, device, scenes=None):
    """
    Yields (scene_name, opt, train_loader, val_loader) per scene.
    `scenes` overrides cfg['scenes'] if provided.
    """
    _ensure_torch_ngp_importable()
    from nerf.provider import NeRFDataset  # torch-ngp

    data_root = cfg.get('data_root', 'data/blender_nerf')
    scenes = scenes or cfg.get('scenes', ['lego'])

    for scene in scenes:
        data_path = os.path.join(data_root, scene)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"NeRF scene not found: {data_path}. "
                f"Put Blender synthetic data under {data_root}/<scene>/.")

        opt = _make_opt(cfg, data_path)
        train_loader = NeRFDataset(
            opt, device=device, type='train',
            downscale=opt.downscale).dataloader()
        val_loader = NeRFDataset(
            opt, device=device, type='val',
            downscale=opt.downscale).dataloader()
        yield scene, opt, train_loader, val_loader
