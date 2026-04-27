#!/bin/bash
#SBATCH --job-name=multi_relu_rerun
#SBATCH --nodes=1
#SBATCH --partition=vvh-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_relu_rerun_%j.out
#SBATCH --error=/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper_multi/slurm_relu_rerun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=donghua@umich.edu

# Re-run relu (plain ReLU MLP, no positional encoding) for fitting / inpaint / sr.
# relu defaults were changed (use_pe / N_freqs removed), so any earlier trial
# state is invalid. denoise excluded — covered by the denoise sbatch.

set -e
cd /scratch/liyues_root/liyues/shared_data/yiting_donghua
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "=========================================="
echo " Job: relu rerun (fitting/inpaint/sr) started $(date)"
echo " Host: $(hostname)   GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "=========================================="

echo ""
echo "------ [1/3] image_fitting (relu) ------"
python benchmark/search_hyper_multi/hparam_search_image_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 4 \
    --methods relu \
    --n_trials 20 \
    --iters 4000 \
    --batch_size -1 \
    --agg mean \
    --seed 1234 \
    --device auto

echo ""
echo "------ [2/3] image_inpainting (relu) ------"
python benchmark/search_hyper_multi/hparam_search_inpaint_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 4 \
    --methods relu \
    --n_trials 20 \
    --iters 4000 \
    --batch_size -1 \
    --sampling_ratio 0.2 \
    --mask_seed 0 \
    --agg mean \
    --seed 1234 \
    --device auto

echo ""
echo "------ [3/3] image_super_resolution (relu) ------"
python benchmark/search_hyper_multi/hparam_search_sr_multi.py \
    --dataset div2k \
    --data_root data/validation \
    --image_indices 214 221 527 \
    --downscale 1 \
    --methods relu \
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
