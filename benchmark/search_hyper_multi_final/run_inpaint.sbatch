#!/bin/bash
#SBATCH --job-name=multi_inpaint
#SBATCH --nodes=1
#SBATCH --partition=liyues-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_inpaint_%j.out
#SBATCH --error=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_inpaint_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=donghua@umich.edu

# image_inpainting only (downscale=4, sampling_ratio=0.2, 3 images)

set -e
cd /scratch/liyues_root/liyues/shared_data/yiting_donghua
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "=========================================="
echo " Job: image_inpainting started $(date)"
echo " Host: $(hostname)   GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "=========================================="

python benchmark/search_hyper_multi/hparam_search_inpaint_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 1 \
    --methods siren wire gauss finer gf wf staf relu sl2a \
    --n_trials 20 \
    --iters 4000 \
    --batch_size -1 \
    --sampling_ratio 0.2 \
    --mask_seed 0 \
    --agg mean \
    --seed 1234 \
    --device auto

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
