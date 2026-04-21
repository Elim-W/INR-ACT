#!/bin/bash
# 2-D Sierpinski triangle benchmark (depths 0–8, deterministic)
cd "$(dirname "$0")/.."
module load python/3.11.5 2>/dev/null || true
source .venv/bin/activate

python benchmark/run_synthetic.py \
    --signal 2d_sierpinski \
    --bandwidths 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --seeds 1234 \
    --iters 1000 \
    --out_dir results/2d_sierpinski \
    "$@"
