#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

# div2k 0002
python benchmark/search_hyper/hparam_search_image.py \
    --dataset div2k --image_idx 2 \
    --n_trials 30 --iters 4000 --max_size 512

# div2k 0132
python benchmark/search_hyper/hparam_search_image.py \
    --dataset div2k --image_idx 132 \
    --n_trials 40 --iters 4000 --max_size 512