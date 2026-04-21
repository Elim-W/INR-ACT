# INR Benchmark

A clean, unified benchmark for Implicit Neural Representation (INR) methods.

## Structure

```
benchmark_root/
├── third_party/          # Reference implementations (read-only, not imported)
├── benchmark/
│   ├── methods/          # All INR methods re-implemented from scratch
│   │   ├── siren.py      SIREN (Sitzmann et al., NeurIPS 2020)
│   │   ├── wire.py       WIRE – complex Gabor (Saragadam et al., CVPR 2023)
│   │   ├── wire2d.py     WIRE-2D – 2D Gabor variant
│   │   ├── gauss.py      Gaussian activation INR
│   │   ├── finer.py      FINER (Liu et al., NeurIPS 2023)
│   │   ├── staf.py       STAF – sum of sinusoids
│   │   ├── pemlp.py      PE-MLP – positional encoding + ReLU
│   │   ├── incode.py     INCODE (Kazerouni et al., WACV 2024)
│   │   ├── sl2a.py       SL2A – ChebyKAN + low-rank ReLU
│   │   └── models.py     get_INR() factory
│   ├── tasks/
│   │   ├── image_fitting.py   ✅ implemented
│   │   ├── nerf.py            🚧 placeholder
│   │   └── sdf.py             🚧 placeholder
│   ├── datasets/
│   │   ├── kodak.py           ✅ implemented
│   │   ├── div2k.py           🚧 placeholder
│   │   └── blender_nerf.py    🚧 placeholder
│   ├── metrics/
│   │   └── image_metrics.py  PSNR, SSIM, MSE
│   ├── analysis/
│   │   ├── collect_results.py
│   │   ├── make_tables.py
│   │   └── plot_curves.py
│   └── run_experiment.py     Single entry point
├── configs/experiments/
│   └── image_fitting/        One YAML per method
├── data/                     Put datasets here
├── results/                  Outputs saved here
└── scripts/
    └── run_image_fitting.sh  Run all methods
```

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision numpy Pillow scikit-image pyyaml matplotlib tqdm
```

### 2. Prepare data

Download [Kodak dataset](http://r0k.us/graphics/kodak/) (24 PNG images) into `data/kodak/`.
Expected filenames: `kodim01.png` … `kodim24.png` (or `kodak01.png`, `01.png`).

### 3. Run one method

```bash
python benchmark/run_experiment.py \
    --config configs/experiments/image_fitting/siren.yaml
```

Run a single image:

```bash
python benchmark/run_experiment.py \
    --config configs/experiments/image_fitting/siren.yaml \
```

Override hyperparameters on the fly:

```bash
python benchmark/run_experiment.py \
    --config configs/experiments/image_fitting/siren.yaml \
    --override training.lr=5e-4 training.num_epochs=5000
```

### 4. Run all methods

```bash
bash scripts/run_image_fitting.sh
```

### 5. Analyse results

```bash
python benchmark/analysis/collect_results.py --out_json results/image_fitting/summary.json
python benchmark/analysis/make_tables.py --json results/image_fitting/summary.json
python benchmark/analysis/plot_curves.py
```

## Methods

| Method  | Activation | Key idea |
|---------|-----------|----------|
| SIREN   | sin(ω₀·Wx) | Periodic activation with careful init |
| WIRE    | Complex Gabor | Wavelet activation (complex numbers) |
| WIRE2D  | 2D Gabor   | Two orthogonal Gaussian windows |
| Gauss   | exp(-(s·Wx)²) | Gaussian radial activation |
| FINER   | sin(ω·(|x|+1)·x) | Adaptive-scale sinusoidal |
| STAF    | Σ bᵢ·sin(wᵢx+φᵢ) | Learnable multi-frequency sinusoids |
| PE-MLP  | ReLU + PE  | Positional encoding baseline |
| INCODE  | Modulated SIREN | Image-conditioned harmonizer network |
| SL2A    | ChebyKAN + LowRank ReLU | KAN first layer + low-rank hidden |

## Config Reference

Each YAML has four top-level sections:

```yaml
task: image_fitting        # task name
method: siren              # method name
dataset: kodak             # dataset name
data_root: data/kodak      # path to images
device: auto               # auto | cpu | cuda

model:                     # passed to get_INR() / INR.__init__()
  hidden_features: 256
  hidden_layers: 3
  ...

training:
  lr: 1e-4
  num_epochs: 2000
  batch_size: -1           # -1 = full image; N = random pixel batch
  scheduler: cosine        # cosine | lambda | none
  log_every: 100

output:
  save_dir: results/image_fitting/siren
  save_model: true
  save_images: true
```
