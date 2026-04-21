#!/bin/bash
# 3-D bandlimited noise benchmark (100^3, 9 bandwidths, wire2d excluded)
cd "$(dirname "$0")/.."
module load python/3.11.5 2>/dev/null || true
source .venv/bin/activate

python benchmark/run_synthetic.py \
    --signal 3d_bandlimited \
    --bandwidths 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --seeds 1234 \
    --iters 1000 \
    --out_dir results/3d_bandlimited \
    "$@"
