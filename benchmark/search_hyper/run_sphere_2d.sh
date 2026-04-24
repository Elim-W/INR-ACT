#!/bin/bash
# Hyperparameter search for 2d_sphere signal.
# Runs all methods sequentially on a single GPU, nohup in background.
#
# Usage:
#   bash benchmark/search_hyper/run_sphere_2d.sh [GPU_ID]
#
# Example:
#   nohup bash benchmark/search_hyper/run_sphere_2d.sh 0 > results/hparam_search/logs/sphere_2d.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SIGNAL="2d_sphere"
STUDY_DIR="results/hparam_search/optuna_db"
N_TRIALS=40
ITERS=2000

mkdir -p results/hparam_search/logs

echo "[sphere_2d] GPU=$GPU_ID  n_trials=$N_TRIALS  iters=$ITERS  $(date)"

for METHOD in siren wire gauss finer gf wf staf pemlp incode sl2a cosmo; do
    echo "======== $METHOD  $(date) ========"
    python -u benchmark/hparam_search.py \
        --signal "$SIGNAL" \
        --methods "$METHOD" \
        --n_trials "$N_TRIALS" \
        --iters "$ITERS" \
        --study_dir "$STUDY_DIR"
done

echo "[sphere_2d] all methods done  $(date)"
