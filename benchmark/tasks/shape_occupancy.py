"""
3D shape occupancy task (a.k.a. the INCODE "sdf" task — it is actually
binary occupancy, not signed distance).

Fit one INR per mesh that maps a 3D coordinate in [-1, 1]^3 to a scalar
occupancy probability.  Supervision is MSE against a pre-computed voxel
occupancy grid; the reported metric is IoU at threshold 0.5.  When the
run finishes we optionally extract a mesh with scikit-image's marching
cubes and dump it as an .obj for visual inspection.

Follows the same return-dict shape as benchmark.tasks.image_fitting so
the analysis/ scripts work unchanged:
    final_psnr  ← here we reuse this key for final IoU (×100)
    final_ssim  ← here we reuse this key for binary accuracy
plus the task-specific 'final_iou', 'final_acc' for clarity.
"""

import os
import time
import numpy as np
import torch


def _make_scheduler(optimizer, cfg_train):
    sched_type = cfg_train.get('scheduler', 'cosine')
    n = cfg_train['num_epochs']
    if sched_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n, eta_min=0)
    elif sched_type == 'lambda':
        decay = cfg_train.get('lambda_decay', 0.2)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: decay ** min(ep / n, 1.0))
    elif sched_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler '{sched_type}'")


def _forward_all(model, coords, batch_size):
    """Chunked forward over all (N,3) coords.  Returns (N, out_features)."""
    N = coords.shape[0]
    if batch_size is None or batch_size == -1 or batch_size >= N:
        return model(coords)
    outs = []
    for i in range(0, N, batch_size):
        outs.append(model(coords[i:i + batch_size]))
    return torch.cat(outs, dim=0)


def _iou(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum().item()
    union = (pred_bin | gt_bin).sum().item()
    return inter / union if union > 0 else 1.0


def _binary_acc(pred_bin, gt_bin):
    return (pred_bin == gt_bin).float().mean().item()


def run(model, coords, targets, meta, cfg, device, save_dir=None):
    """
    Args:
        coords:   (N, 3) in [-1, 1]^3
        targets:  (N, 1) binary occupancy {0, 1}
        meta:     dict with 'H','W','T','name','mesh_whl'
    """
    cfg_train = cfg['training']
    H, W, T = meta['H'], meta['W'], meta['T']
    N = H * W * T
    thres = cfg_train.get('iou_threshold', 0.5)

    coords = coords.to(device)
    targets = targets.to(device)
    gt_bin = (targets > 0.5).reshape(-1)

    # INCODE's 3D harmonizer (r3d_18) is not implemented in our benchmark's
    # incode.py — it assumes 2D.  For non-INCODE methods this hook is a no-op.
    if hasattr(model, 'set_gt'):
        # Best effort: feed a 3-channel 3D volume.  If the method's harmonizer
        # cannot consume it, the user must switch to a 2D-only method.
        vol = targets.reshape(H, W, T).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,T)
        vol = vol.expand(1, 3, H, W, T)
        try:
            model.set_gt(vol)
        except Exception as e:
            print(f"  [shape_occupancy] set_gt failed ({e}); "
                  "method may not support 3D harmonizer.")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['lr'])
    scheduler = _make_scheduler(optimizer, cfg_train)

    num_epochs = cfg_train['num_epochs']
    batch_size = cfg_train.get('batch_size', 100_000)
    log_every = cfg_train.get('log_every', 50)

    iou_curve, acc_curve, epochs_curve = [], [], []
    total_time = 0.0
    best_iou = -float('inf')
    best_state = None

    print(f"  occupancy ratio (GT): {gt_bin.float().mean().item():.3f}  "
          f"grid={H}x{W}x{T}  batch={batch_size}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Shuffle then sweep once through all points in mini-batches
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for b_idx in range(0, N, batch_size):
            b_ind = perm[b_idx:b_idx + batch_size]
            pred = model(coords[b_ind])
            loss = torch.mean((pred - targets[b_ind]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()

        total_time += time.time() - t0

        if epoch % log_every == 0 or epoch == num_epochs:
            with torch.no_grad():
                pred_full = _forward_all(model, coords, batch_size)
            pred_bin = (pred_full.reshape(-1) > thres)
            iou = _iou(pred_bin, gt_bin)
            acc = _binary_acc(pred_bin, gt_bin)
            iou_curve.append(iou)
            acc_curve.append(acc)
            epochs_curve.append(epoch)
            if iou > best_iou:
                best_iou = iou
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  [{meta['name']}] epoch {epoch:5d}/{num_epochs}"
                  f"  loss={epoch_loss/max(n_batches,1):.6f}"
                  f"  IoU={iou:.4f}  acc={acc:.4f}")

    # Final evaluation from best state
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred_full = _forward_all(model, coords, batch_size)
    pred_bin = (pred_full.reshape(-1) > thres)
    final_iou = _iou(pred_bin, gt_bin)
    final_acc = _binary_acc(pred_bin, gt_bin)
    pred_vol = pred_full.reshape(H, W, T).detach().cpu().numpy()

    mesh_info = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # Save the raw predicted volume for later analysis
        np.savez_compressed(
            os.path.join(save_dir, f"{meta['name']}_pred_volume.npz"),
            pred=pred_vol.astype(np.float32),
            gt=targets.reshape(H, W, T).cpu().numpy().astype(np.uint8),
            mesh_whl=np.asarray(meta['mesh_whl'], dtype=np.float32),
        )
        # Mesh extraction via skimage marching cubes (optional; fails silently
        # if the iso-surface is empty).
        mesh_info = _try_save_mesh(
            pred_vol, meta['mesh_whl'], thres,
            os.path.join(save_dir, f"{meta['name']}_mesh.obj"))

        # Optional multi-view render of the extracted mesh.
        render_views = cfg.get('output', {}).get('render_views', 0)
        if render_views and mesh_info is not None:
            _try_multi_view_render(
                mesh_info['verts'], mesh_info['faces'],
                n_views=int(render_views),
                out_path=os.path.join(save_dir, f"{meta['name']}_views.png"),
                title=meta['name'])
        # Drop heavy arrays before returning (keep only the summary)
        if mesh_info is not None:
            mesh_info = {k: v for k, v in mesh_info.items()
                         if k not in ('verts', 'faces')}

    return {
        'name':          meta['name'],
        # Reuse the image-task keys so analysis/ scripts still work
        'psnr_curve':    [100.0 * i for i in iou_curve],      # IoU×100 for comparability
        'ssim_curve':    acc_curve,
        'epochs_curve':  epochs_curve,
        'final_psnr':    100.0 * final_iou,
        'final_ssim':    final_acc,
        'total_time_s':  total_time,
        'model_state':   best_state,
        # Task-specific
        'final_iou':     final_iou,
        'final_acc':     final_acc,
        'iou_curve':     iou_curve,
        'acc_curve':     acc_curve,
        'grid_res':      H,
        'threshold':     thres,
        'mesh_info':     mesh_info,
    }


def _try_save_mesh(occ_volume, mesh_whl, thres, out_path):
    """Extract an iso-surface with skimage marching cubes and write .obj."""
    try:
        from skimage import measure
    except ImportError:
        print("  [shape_occupancy] scikit-image not available; skipping mesh export.")
        return None
    vol = occ_volume.astype(np.float32)
    if vol.max() <= thres or vol.min() >= thres:
        print(f"  [shape_occupancy] iso-surface at {thres} is empty; skipping.")
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=thres)
    except Exception as e:
        print(f"  [shape_occupancy] marching_cubes failed: {e}")
        return None

    # verts in grid indices → normalise to [-1,1]^3, then scale by mesh_whl
    n = np.asarray(vol.shape, dtype=np.float32)
    verts = (verts / (n - 1)) * 2.0 - 1.0
    verts = verts * np.asarray(mesh_whl, dtype=np.float32)

    with open(out_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            # OBJ is 1-indexed
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"  [shape_occupancy] mesh → {out_path} "
          f"(verts={len(verts)}, faces={len(faces)})")
    return {'n_verts': int(len(verts)), 'n_faces': int(len(faces)),
            'path': out_path,
            'verts': verts, 'faces': faces}


def _try_multi_view_render(verts, faces, n_views, out_path, title=''):
    """
    Headless multi-view rendering of a triangle mesh via matplotlib's
    Poly3DCollection — no OpenGL required, works on login nodes.

    Produces one PNG with `n_views` subplots taken at evenly-spaced azimuths
    (with an extra top-down view if n_views >= 6).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')               # headless backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("  [shape_occupancy] matplotlib missing; skipping multi-view render.")
        return None

    # Build camera presets: n-1 evenly-spaced azimuths around the object,
    # plus one top-down view.  (elev, azim) in degrees.
    if n_views <= 0:
        return None
    cams = []
    horiz = max(1, n_views - 1)
    for i in range(horiz):
        cams.append((20.0, i * 360.0 / horiz))   # elev=20°, azim sweep
    cams.append((85.0, 0.0))                     # top-down
    cams = cams[:n_views]

    # Layout: try to be ~square
    ncols = int(np.ceil(np.sqrt(n_views)))
    nrows = int(np.ceil(n_views / ncols))
    fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows))

    tri = verts[faces]                          # (F, 3, 3)
    lim = float(np.abs(verts).max()) * 1.05

    for i, (elev, azim) in enumerate(cams):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        coll = Poly3DCollection(tri, facecolor=(0.8, 0.85, 0.95),
                                edgecolor=(0.2, 0.2, 0.2), linewidths=0.05)
        ax.add_collection3d(coll)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"elev={elev:.0f}°, azim={azim:.0f}°", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [shape_occupancy] multi-view render → {out_path} ({n_views} views)")
    return out_path
