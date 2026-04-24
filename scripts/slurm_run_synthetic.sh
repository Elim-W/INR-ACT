#!/bin/bash
#SBATCH --job-name=run_synthetic
#SBATCH --partition=liyues-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/results/logs/run_synthetic_%j.log

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

module load python/3.11.5 2>/dev/null || true
source .venv/bin/activate

SIGNAL="${1:?Usage: sbatch scripts/slurm_run_synthetic.sh <signal> [extra args]}"
shift || true

echo "[run_synthetic] signal=$SIGNAL  $(date)"
python -u benchmark/run_synthetic.py \
    --signal "$SIGNAL" \
    "$@"
echo "[run_synthetic] done  $(date)"
