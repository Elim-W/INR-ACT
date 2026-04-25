#!/bin/bash
# 2-D Sierpinski triangle benchmark (depths 0–8, deterministic signal)
# iters / seeds / methods come from configs/experiments/synthetic/2d_sierpinski.yaml
cd "$(dirname "$0")/.."
module load python/3.11.5 2>/dev/null || true
source .venv/bin/activate

python -u benchmark/run_synthetic.py \
    --signal 2d_sierpinski \
    "$@"
