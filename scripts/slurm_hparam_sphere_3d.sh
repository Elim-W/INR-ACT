#!/bin/bash
#SBATCH --job-name=hparam_sphere_3d
#SBATCH --partition=vvh-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/results/hparam_search/logs/sphere_3d_%j.log

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hparam_search/logs

module load python/3.11.5 2>/dev/null || true
source .venv/bin/activate

echo "[hparam_sphere_3d] start  $(date)"

for METHOD in siren wire gauss finer gf wf staf pemlp incode sl2a cosmo; do
    echo "======== $METHOD  $(date) ========"
    python -u benchmark/hparam_search.py \
        --signal 3d_sphere \
        --methods "$METHOD" \
        --n_trials 20 \
        --iters 10000 \
        --batch_size 65536 \
        --study_dir results/hparam_search/optuna_db
done

echo "[hparam_sphere_3d] done  $(date)"
