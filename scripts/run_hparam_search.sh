#!/bin/bash
# Hyperparameter search for all INR methods on a given signal.
# Submits one SLURM job per method (they run in parallel).
#
# Usage:
#   bash scripts/run_hparam_search.sh 2d_startarget
#   bash scripts/run_hparam_search.sh 2d_startarget --n_trials 50 --iters 3000
#
# After all jobs finish, best params are written to
#   configs/experiments/synthetic/<signal>.yaml
# and raw JSON to results/hparam_search/<signal>_best_params_<method>.json

set -euo pipefail
cd "$(dirname "$0")/.."

SIGNAL="${1:?Usage: $0 <signal> [extra args]}"
shift || true

# --- SLURM defaults (edit as needed) ---
PARTITION="${SLURM_PARTITION:-vvh-l40s}"
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS:-4}"
MEM="${SLURM_MEM:-32G}"
TIME="${SLURM_TIME:-04:00:00}"
# ----------------------------------------

# Parse --methods from args; remaining args passed through to Python
METHODS=(siren wire gauss finer gf wf staf pemlp incode sl2a cosmo)
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--methods" ]]; then
        shift
        METHODS=()
        while [[ $# -gt 0 && "$1" != --* ]]; do
            METHODS+=("$1")
            shift
        done
    else
        EXTRA_ARGS+=("$1")
        shift
    fi
done

LOG_DIR="results/hparam_search/logs"
mkdir -p "$LOG_DIR"

JOBS=()
for METHOD in "${METHODS[@]}"; do
    JOB_NAME="hps_${SIGNAL}_${METHOD}"
    LOG="${LOG_DIR}/${JOB_NAME}_%j.log"

    JOB_ID=$(sbatch \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --gres="gpu:${GPUS}" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --output="$LOG" \
        --parsable \
        --wrap="
module load python/3.11.5 2>/dev/null || true
source .venv/bin/activate
python -u benchmark/hparam_search.py \
    --signal $SIGNAL \
    --methods $METHOD \
    --study_dir results/hparam_search/optuna_db \
    ${EXTRA_ARGS[*]}
")
    echo "Submitted $JOB_NAME → job $JOB_ID  (log: $LOG)"
    JOBS+=("$JOB_ID")
done

# Submit a merge job that waits for all search jobs, then merges JSONs into YAML
DEPS=$(IFS=:; echo "${JOBS[*]}")
sbatch \
    --job-name="hps_${SIGNAL}_merge" \
    --dependency="afterok:${DEPS}" \
    --partition="${PARTITION}" \
    --cpus-per-task=1 \
    --mem=4G \
    --time=00:10:00 \
    --output="${LOG_DIR}/hps_${SIGNAL}_merge_%j.log" \
    --wrap="
module load python/3.11.5 2>/dev/null || true
source .venv/bin/activate
python benchmark/hparam_search_merge.py --signal $SIGNAL
echo 'Merge done — check configs/experiments/synthetic/${SIGNAL}.yaml'
"

echo ""
echo "All ${#JOBS[@]} search jobs submitted. A merge job will run after they finish."
echo "Monitor with:  squeue -u \$USER"
