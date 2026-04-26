#!/bin/bash
#SBATCH --job-name=ct_recon
#SBATCH --nodes=1
#SBATCH --partition=liyues-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/results/image_ct_reconstruction/slurm_%j.out
#SBATCH --error=/scratch/liyues_root/liyues/shared_data/yiting_donghua/results/image_ct_reconstruction/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=donghua@umich.edu

# CT reconstruction with the supplied config.
# Submit example:
#   sbatch run_ct.sbatch configs/experiments/image_ct_reconstruction/siren.yaml

set -e
cd /scratch/liyues_root/liyues/shared_data/yiting_donghua
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
mkdir -p results/image_ct_reconstruction

CONFIG="${1:-configs/experiments/image_ct_reconstruction/siren.yaml}"
echo "=========================================="
echo " Job: CT reconstruction started $(date)"
echo " Host: $(hostname)   GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo " Config: $CONFIG"
echo "=========================================="

python benchmark/run_ct.py --config "$CONFIG"

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
