#!/bin/bash
#SBATCH --job-name=inr_multi_search
#SBATCH --nodes=1
#SBATCH --partition=liyues-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=100:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_%j.out
#SBATCH --error=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=donghua@umich.edu

# ============================================================================
#  Multi-image hyperparameter search across 4 INR tasks
#    images: 0214 (low-mid) + 0221 (high) + 0527 (high-mid) from data/validation
#    aggregation: mean PSNR (across 3 images per trial)
#    n_trials: 20    iters/trial: 4000    seed: 1234
#
#  All default values are listed explicitly below for clarity / reproducibility.
#  Per-task settings:
#    fitting:    downscale=4
#    denoise:    downscale=4, Poisson-Gaussian τ=20, readout_snr=2
#    inpainting: downscale=4, sampling_ratio=0.2 (drop 80%, keep 20%)
#    SR:         no downscale (full DIV2K HR), scale_factor=8
#
#  Total expected runtime on A100: ~30-50 hours
# ============================================================================

set -e
cd /scratch/liyues_root/liyues/shared_data/yiting_donghua
source .venv/bin/activate

# Use whatever GPU SLURM allocates
unset CUDA_VISIBLE_DEVICES

echo "=========================================="
echo " Job started: $(date)"
echo " Hostname:    $(hostname)"
echo " GPU:         $(nvidia-smi -L 2>/dev/null | head -1)"
echo "=========================================="


# ---------------------------------------------------------------------------
# 1. IMAGE FITTING  (downscale 4)
# ---------------------------------------------------------------------------
echo ""
echo "########## [1/4] image_fitting ##########"
python benchmark/search_hyper_multi/hparam_search_image_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 4 \
    --methods siren wire gauss finer gf wf staf pemlp incode sl2a cosmo \
    --n_trials 20 \
    --iters 4000 \
    --batch_size -1 \
    --agg mean \
    --seed 1234 \
    --device auto


# ---------------------------------------------------------------------------
# 2. IMAGE DENOISING  (downscale 4, Poisson-Gaussian τ=20)
# ---------------------------------------------------------------------------
echo ""
echo "########## [2/4] image_denoising ##########"
python benchmark/search_hyper_multi/hparam_search_denoise_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 4 \
    --methods siren wire gauss finer gf wf staf pemlp incode sl2a cosmo \
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


# ---------------------------------------------------------------------------
# 3. IMAGE INPAINTING  (downscale 4, keep 20% pixels)
# ---------------------------------------------------------------------------
echo ""
echo "########## [3/4] image_inpainting ##########"
python benchmark/search_hyper_multi/hparam_search_inpaint_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 4 \
    --methods siren wire gauss finer gf wf staf pemlp incode sl2a cosmo \
    --n_trials 20 \
    --iters 4000 \
    --batch_size -1 \
    --sampling_ratio 0.2 \
    --mask_seed 0 \
    --agg mean \
    --seed 1234 \
    --device auto


# ---------------------------------------------------------------------------
# 4. IMAGE SUPER-RESOLUTION  (full HR, scale ×8)
# ---------------------------------------------------------------------------
echo ""
echo "########## [4/4] image_super_resolution ##########"
python benchmark/search_hyper_multi/hparam_search_sr_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 1 \
    --methods siren wire gauss finer gf wf staf pemlp incode sl2a cosmo \
    --n_trials 20 \
    --iters 4000 \
    --batch_size -1 \
    --scale_factor 8 \
    --eval_epoch 0 \
    --agg mean \
    --seed 1234 \
    --device auto


echo ""
echo "=========================================="
echo " All 4 searches done: $(date)"
echo "=========================================="
