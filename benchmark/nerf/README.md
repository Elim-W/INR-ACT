# NeRF activation benchmark

Standalone NeRF training stack — does **not** plug into `run_experiment_3d.py`
or `hparam_search.py`. Each method has its own `network_<method>.py`; the
shared `main_nerf.py` picks one with `--method` and writes a fixed output
layout per `(scene, method)`.

## Methods

| --method  | Activation | Encoding (x → dim, dir → dim) |
|-----------|-----------|-------------------------------|
| `finer`   | `sin(ω·(|Wx|+1)·Wx)` | None (3, 3) |
| `siren`   | `sin(ω·Wx)`          | None (3, 3) |
| `gauss`   | `exp(-(s·Wx)²)`      | None (3, 3) |
| `relu_pe` | `ReLU`               | freq (63, 27) — original NeRF (Mildenhall 2020) |

All four share the same backbone shape: 4-layer sigma MLP + 4-layer color MLP,
hidden 128, geo_feat 128. The *only* deliberate per-method difference is the
activation and (for `relu_pe`) the input encoding.

## Setup (one time)

torch-ngp's CUDA extensions JIT-build on first import. They need:

- `nvcc` available + matching CUDA toolkit
- Python deps: `pip install opencv-python tensorboardX lpips trimesh packaging tqdm rich ninja imageio imageio-ffmpeg scipy einops`

Data layout (already downloaded):

```
data/blender_nerf/
├── lego/{train,val,test}/  + transforms_*.json
├── chair/, drums/, ficus/, hotdog/, materials/, mic/, ship/
```

## Run

```bash
# from repo root
python -m benchmark.nerf.main_nerf data/blender_nerf/lego --method finer -O
python -m benchmark.nerf.main_nerf data/blender_nerf/lego --method siren -O
python -m benchmark.nerf.main_nerf data/blender_nerf/lego --method gauss -O
python -m benchmark.nerf.main_nerf data/blender_nerf/lego --method relu_pe -O
```

`-O` = `--fp16 --cuda_ray --preload` (the standard "fast" recipe).

## Output layout

```
outputs/nerf/<scene>/<method>/
├── config.json          # all CLI args
├── metrics.json         # best_psnr + per-eval curves
├── checkpoints/         # auto (torch-ngp Trainer)
├── log_<method>.txt     # auto
├── run/                 # tensorboard, auto
└── validation/          # val renders, auto
```

`metrics.json` schema:

```json
{
  "method": "finer",
  "scene":  "lego",
  "best_psnr":      28.31,
  "final_psnr":     28.07,
  "psnr_curve":     [12.4, 18.1, 23.5, ...],
  "valid_loss":     [...],
  "train_loss":     [...],
  "num_params":     198915,
  "epochs_trained": 30,
  "total_time_s":   1843.2
}
```

After training, render test split + mp4:

```bash
python -m benchmark.nerf.main_nerf data/blender_nerf/lego --method finer --test -O
```

## Smoke test

A successful run should:
1. Start training without errors
2. Have `train_loss` decreasing (visible in tqdm bar and `log_<method>.txt`)
3. Have `psnr_curve[-1] > psnr_curve[0]` after a few evals
4. Produce no NaN
5. Drop validation renders to `outputs/.../validation/` that look like the scene

Don't gate on a specific PSNR threshold for smoke — only on full runs (~30k iters).

## Known minor differences from FINER's repo

- `provider.py` and `utils.py` come straight from torch-ngp (we don't vendor a
  copy). FINER's only modification was a `train_skip` arg in `provider.py` for
  WIRE-style 25-image subsampling — unused here since we run all 4 methods on
  the full Blender protocol (100 train images).
- Trainer is constructed with `use_loss_as_metric=False, best_mode='max'`
  (FINER's `main_nerf.py` uses defaults, which silently track MSE instead of
  PSNR — see `nerf/utils.py:1001`).
- Background model (`bg_radius > 0`) is dropped: all four `network_*.py` return
  black for `background()`. Blender synthetic data has a white background, so
  if you re-enable, also wire up a real bg_net and pass `--bg_radius 32`.
