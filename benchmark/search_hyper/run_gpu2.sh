#!/bin/bash
# GPU 2:
#   (a) delete stale img2 results that used the wrong backbone (H=360 for wire/wf)
#   (b) re-run wire, wf at the correct H=256 backbone
#   (c) run sl2a, cosmo (haven't been run yet; will use the new H=256 defaults)
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

# Drop stale H=360 results so we don't confuse old and new numbers.
rm -f benchmark/search_hyper/div2k_img2_wire.txt
rm -f benchmark/search_hyper/div2k_img2_wf.txt

# Re-run the 4 methods on img2 under the unified H=256, L=3 backbone.
python benchmark/search_hyper/hparam_search_image.py \
    --dataset div2k --image_idx 2 \
    --methods wire wf sl2a cosmo \
    --n_trials 30 --iters 4000 --max_size 512
