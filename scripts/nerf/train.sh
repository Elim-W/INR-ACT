#!/usr/bin/env bash
# Train one NeRF config on a GPU compute node.
#
# Interactive:
#   srun --gres=gpu:1 --mem=32G --time=4:00:00 --pty bash scripts/nerf/train.sh configs/experiments/nerf/siren.yaml
#
# Batch:
#   sbatch scripts/nerf/train.sh configs/experiments/nerf/siren.yaml

#SBATCH --job-name=nerf-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/nerf_%j.out

set -euo pipefail

CONFIG="${1:-configs/experiments/nerf/siren.yaml}"
shift || true

cd /scratch/liyues_root/liyues/shared_data/yiting_donghua

module load python/3.11.5
module load cuda/11.8.0
module load gcc/11.2.0

source .venv_nerf/bin/activate

# Same sys.path fix as for compile — just in case torch-ngp re-invokes nvcc
# via JIT cpp_extension for anything missing.
export CPATH="/sw/pkgs/arc/python/3.11.5/include/python3.11:${CPATH:-}"

mkdir -p logs

echo "[train] config=$CONFIG"
echo "[train] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train] overrides: $*"

python -u benchmark/run_experiment_3d.py --config "$CONFIG" ${@:+--override "$@"}
