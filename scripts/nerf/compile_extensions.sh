#!/usr/bin/env bash
# Compile torch-ngp's 4 CUDA extensions.
# MUST be run on a compute node (login node has a 4GB memory cgroup that
# nvcc cannot fit into).
#
# Usage (interactive GPU session):
#   srun --gres=gpu:1 --mem=16G --time=1:00:00 --pty bash scripts/nerf/compile_extensions.sh
#
# Or batch:
#   sbatch scripts/nerf/compile_extensions.sh

#SBATCH --job-name=nerf-compile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/compile_%j.out

set -euo pipefail

cd /scratch/liyues_root/liyues/shared_data/yiting_donghua

# Load the matched CUDA/gcc toolchain torch-ngp expects
module load python/3.11.5
module load cuda/11.8.0
module load gcc/11.2.0

source .venv_nerf/bin/activate

# Need this because the venv's sysconfig points Python.h at /usr/include
# (where it doesn't exist) rather than /sw/pkgs/arc/python/...
export CPATH="/sw/pkgs/arc/python/3.11.5/include/python3.11:${CPATH:-}"

# If set, only compile for this GPU arch (saves compile time).
# Detect from current GPU if not set.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        export TORCH_CUDA_ARCH_LIST="${cc}"
        echo "[compile] auto-detected arch: ${TORCH_CUDA_ARCH_LIST}"
    else
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
        echo "[compile] no nvidia-smi; compiling for all major archs"
    fi
fi

# Single-threaded nvcc — memory-friendly and still quick
export MAX_JOBS=2

cd third_party/torch-ngp

for ext in freqencoder gridencoder raymarching shencoder; do
    echo ""
    echo "=== Building $ext ==="
    rm -rf "$ext/build" "$ext"/*.egg-info
    (cd "$ext" && pip install --no-build-isolation . 2>&1 | tail -5)
done

echo ""
echo "=== Verifying imports ==="
python - <<'PY'
import importlib, sys
for mod in ['_freqencoder', '_gridencoder', '_raymarching', '_shencoder',
            'freqencoder', 'gridencoder', 'raymarching', 'shencoder']:
    try:
        importlib.import_module(mod)
        print(f"  OK  {mod}")
    except Exception as e:
        print(f"  FAIL {mod}: {type(e).__name__}: {e}")
        sys.exit(1)
print("All 4 extensions imported successfully.")
PY

echo ""
echo "=== Done ==="
