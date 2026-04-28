#!/bin/bash
# GPU 1: finish img2 remaining [staf, pemlp, incode], then img132 [siren, wire, gauss, finer, cosmo]
# Total: 8 method-runs, 2 heavy (incode@img2, cosmo@img132)
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0  # local index, not physical GPU id

# Finish div2k img2 (remaining methods)
python benchmark/search_hyper/hparam_search_image.py \
    --dataset div2k --image_idx 2 \
    --methods cosmo \
    --n_trials 30 --iters 4000 --max_size 512

# # Start div2k img132 (part 1)
# python benchmark/search_hyper/hparam_search_image.py \
#     --dataset div2k --image_idx 132 \
#     --methods siren wire gauss finer cosmo \
#     --n_trials 40 --iters 4000 --max_size 512
