#!/bin/bash
# 2-D bandlimited noise benchmark
# iters / seeds / methods / hyperparams come from configs/experiments/synthetic/2d_bandlimited.yaml
# Pass extra CLI args to override anything, e.g.:
#   bash scripts/run_2dbandlimited.sh --methods siren relu --bandwidths 0.3 0.7 --iters 1000
cd "$(dirname "$0")/.."
module load python/3.11.5 2>/dev/null || true
source .venv/bin/activate

python -u benchmark/run_synthetic.py \
    --signal 2d_bandlimited \
    "$@"
