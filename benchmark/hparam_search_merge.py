"""
Merge per-method JSON results from hparam_search.py into the signal YAML config.

hparam_search.py already writes the YAML when run for a single method,
but when multiple methods run in parallel each job only writes its own method.
This script collects all per-method JSON files and rebuilds the full YAML.

Usage:
    python benchmark/hparam_search_merge.py --signal 2d_startarget
"""

import os
import sys
import json
import glob
import argparse
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--signal', required=True)
    args = p.parse_args()

    results_dir = os.path.join(_ROOT, 'results', 'hparam_search')
    json_pattern = os.path.join(results_dir, f'{args.signal}_best_params_*.json')
    files = glob.glob(json_pattern)

    # Also load the combined file if it exists (single-run case)
    combined = os.path.join(results_dir, f'{args.signal}_best_params.json')
    if os.path.exists(combined):
        files.append(combined)

    if not files:
        print(f'No JSON files found matching {json_pattern}')
        sys.exit(1)

    best_params = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        best_params.update(data)
        print(f'  loaded {path}')

    # Load existing YAML (preserve non-methods keys like iters, seeds, etc.)
    yaml_path = os.path.join(
        _ROOT, 'configs', 'experiments', 'synthetic', f'{args.signal}.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    cfg.setdefault('methods', {})
    for method, params in best_params.items():
        cfg['methods'][method] = params

    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f'\n[merged] {len(best_params)} methods → {yaml_path}')
    for method, params in sorted(best_params.items()):
        print(f'  {method}: {params}')


if __name__ == '__main__':
    main()
