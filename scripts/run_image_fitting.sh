#!/usr/bin/env bash
# Run image-fitting benchmark for all methods on the Kodak dataset.
# Usage:  bash scripts/run_image_fitting.sh [extra args]
#   e.g.  bash scripts/run_image_fitting.sh --override training.num_epochs=500

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

METHODS=(siren wire wire2d gauss finer staf pemlp incode sl2a)

for METHOD in "${METHODS[@]}"; do
    echo "=============================="
    echo " Running: $METHOD"
    echo "=============================="
    python benchmark/run_experiment.py \
        --config "configs/experiments/image_fitting/${METHOD}.yaml" \
        "$@"
done

echo ""
echo "All methods done. Collecting results..."
python benchmark/analysis/collect_results.py \
    --results_dir results \
    --task image_fitting \
    --out_json results/image_fitting/summary.json

echo "Plotting curves..."
python benchmark/analysis/plot_curves.py \
    --results_dir results \
    --task image_fitting \
    --out_dir results/plots

echo "Making tables..."
python benchmark/analysis/make_tables.py \
    --json results/image_fitting/summary.json \
    --out_dir results/tables
