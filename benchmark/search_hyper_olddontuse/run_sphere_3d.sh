#!/bin/bash
# Hyperparameter search for 3d_sphere signal.
# Run overnight on a single GPU with nohup.
#
# Usage:
#   nohup bash benchmark/search_hyper/run_sphere_3d.sh 0 > results/hparam_search/logs/sphere_3d.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SIGNAL="3d_sphere"
STUDY_DIR="results/hparam_search/optuna_db"
N_TRIALS=20
ITERS=10000
BATCH_SIZE=65536

mkdir -p results/hparam_search/logs

echo "[sphere_3d] GPU=$GPU_ID  n_trials=$N_TRIALS  iters=$ITERS  $(date)"

for METHOD in siren wire gauss finer gf wf staf pemlp incode sl2a cosmo; do
    echo "======== $METHOD  $(date) ========"
    python -u benchmark/hparam_search.py \
        --signal "$SIGNAL" \
        --methods "$METHOD" \
        --n_trials "$N_TRIALS" \
        --iters "$ITERS" \
        --batch_size "$BATCH_SIZE" \
        --study_dir "$STUDY_DIR"
done

echo "[sphere_3d] all methods done  $(date)"
