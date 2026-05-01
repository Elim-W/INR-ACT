#!/bin/bash
#SBATCH --job-name=sr_low_finish
#SBATCH --nodes=1
#SBATCH --partition=liyues-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/liyues_root/liyues/shared_data/yiting_donghua/results/image_super_resolution_groupss/sr_low_finish_%j.out
#SBATCH --error=/scratch/liyues_root/liyues/shared_data/yiting_donghua/results/image_super_resolution_groupss/sr_low_finish_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=donghua@umich.edu

# Finish the SR sweep on low_5 only — methods that are still missing.
# Inventory at submit time:
#   low_5/cosmo  0/5    (run all)
#   low_5/incode 0/5    (run all)
#   low_5/relu   0/5    (run all)
#   low_5/sl2a   0/5    (run all)
#   low_5/staf   0/5    (run all)
#   low_5/wf     3/5    (--skip_existing will skip the 3 already done; run 2 more)
# Other groups (high_5, mid_5) and other methods on low_5 are already 5/5,
# so they aren't passed to --methods at all.

set -e
cd /scratch/liyues_root/liyues/shared_data/yiting_donghua
module load python/3.11.5
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "=========================================="
echo " Job: SR low_5 finish  started $(date)"
echo " Host: $(hostname)   GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "=========================================="

python benchmark/run_superresolution_groups.py \
    --config_dir configs/experiments/image_super_resolution_3mean \
    --out_root results/image_super_resolution_groupss \
    --downscale 1 \
    --scale_factor 8 \
    --log_every 400 \
    --skip_existing \
    --groups low_5 \
    --methods relu sl2a staf wf

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="

# Regenerate the SR summary now that low_5 is complete
python donghua/regenerate_sr_summary.py
