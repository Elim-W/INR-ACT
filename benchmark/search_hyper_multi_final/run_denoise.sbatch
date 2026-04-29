#!/bin/bash
#SBATCH --job-name=multi_denoise
#SBATCH --nodes=1
#SBATCH --partition=liyues-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=100:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_denoise_%j.out
#SBATCH --error=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_denoise_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=donghua@umich.edu

# image_denoising only (downscale=4, Poisson-Gaussian τ=20, 3 images)

set -e
cd /scratch/liyues_root/liyues/shared_data/yiting_donghua
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "=========================================="
echo " Job: image_denoising started $(date)"
echo " Host: $(hostname)   GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "=========================================="

python benchmark/search_hyper_multi/hparam_search_denoise_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 4 \
    --methods finer gf wf staf relu incode sl2a cosmo \
    --n_trials 20 \
    --iters 4000 \
    --batch_size -1 \
    --noise_type poisson_gaussian \
    --noise_tau 20.0 \
    --noise_readout_snr 2.0 \
    --noise_seed 0 \
    --agg mean \
    --seed 1234 \
    --device auto

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
