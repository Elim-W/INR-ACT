#!/bin/bash
#SBATCH --job-name=multi_sr
#SBATCH --nodes=1
#SBATCH --account=liyues
#SBATCH --partition=liyues-a100

#SBATCH --account=liyues1
#SBATCH --partition=spgpu

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_sr_%j.out
#SBATCH --error=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_sr_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=donghua@umich.edu

# image_super_resolution only (full HR, scale_factor=8, 3 images)

set -e
cd /scratch/liyues_root/liyues0/shared_data/INR-ACT
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "=========================================="
echo " Job: image_super_resolution started $(date)"
echo " Host: $(hostname)   GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "=========================================="

python benchmark/search_hyper_multi/hparam_search_sr_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 1 \
    --methods siren wire gauss finer gf wf staf relu incode sl2a cosmo \
    --n_trials 20 \
    --iters 4000 \
    --batch_size -1 \
    --scale_factor 8 \
    --eval_epoch 0 \
    --agg mean \
    --seed 1234 \
    --device auto

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
