"""
Hyperparameter search for all INR methods using Optuna.

Usage:
    python benchmark/hparam_search.py --signal 2d_startarget
    python benchmark/hparam_search.py --signal 2d_startarget --methods siren wire --n_trials 30
    python benchmark/hparam_search.py --signal 2d_startarget --iters 3000 --n_trials 50

After search completes, best params are written back to the signal YAML config
under configs/experiments/synthetic/<signal>.yaml.
"""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from benchmark.run_synthetic import (
    SIGNALS, make_startarget_2d, make_coords_2d, make_coords_3d,
    make_bandlimited_2d, make_sphere_2d, make_bandlimited_3d, make_sphere_3d,
    train_one, set_seed,
)
from benchmark.methods.models import get_INR, BENCHMARK_DEFAULTS, TRAIN_KEYS


# ---------------------------------------------------------------------------
# Per-method search spaces
# Each value is a tuple: ('float_log', lo, hi) | ('float', lo, hi) |
#                        ('int', lo, hi) | ('categorical', [v1, v2, ...])
# ---------------------------------------------------------------------------

SEARCH_SPACES = {
    'siren': {
        'lr':             ('float_log', 1e-5, 1e-2),
        'first_omega_0':  ('float', 5.0, 60.0),
        'hidden_omega_0': ('float', 5.0, 60.0),
    },
    'wire': {
        'lr':             ('float_log', 1e-4, 1e-1),
        'first_omega_0':  ('float', 5.0, 40.0),
        'hidden_omega_0': ('float', 5.0, 40.0),
        'scale':          ('float_log', 1.0, 30.0),
    },
    'gauss': {
        'lr':             ('float_log', 1e-4, 1e-2),
        'scale':          ('float_log', 1.0, 30.0),
    },
    'finer': {
        'lr':             ('float_log', 1e-5, 1e-2),
        'first_omega_0':  ('float', 5.0, 60.0),
        'hidden_omega_0': ('float', 5.0, 60.0),
    },
    'gf': {
        'lr':             ('float_log', 1e-4, 1e-2),
        'scale':          ('float', 0.5, 10.0),
        'omega':          ('float', 1.0, 30.0),
        'first_bias_scale':('float', 0.1, 5.0),
    },
    'wf': {
        'lr':             ('float_log', 1e-4, 1e-2),
        'scale':          ('float', 0.5, 10.0),
        'omega_w':        ('float', 1.0, 15.0),
        'omega':          ('float', 1.0, 15.0),
        'first_bias_scale':('float', 0.1, 5.0),
    },
    'staf': {
        'lr':             ('float_log', 1e-5, 1e-2),
        'first_omega_0':  ('float', 5.0, 60.0),
        'hidden_omega_0': ('float', 5.0, 60.0),
        'tau':            ('int', 1, 10),
        'skip_conn':      ('categorical', [True, False]),
    },
    'pemlp': {
        'lr':             ('float_log', 1e-4, 1e-2),
    },
    'incode': {
        'lr':             ('float_log', 1e-5, 1e-2),
        'first_omega_0':  ('float', 5.0, 60.0),
        'hidden_omega_0': ('float', 5.0, 60.0),
    },
    'sl2a': {
        'lr':             ('float_log', 1e-4, 1e-2),
        'degree':         ('categorical', [64, 128, 256]),
        'rank':           ('categorical', [32, 64, 128]),
    },
    'cosmo': {
        'lr':             ('float_log', 1e-3, 1e-1),
        'beta0':          ('float', 0.01, 0.5),
    },
}


def suggest_params(trial, method):
    space = SEARCH_SPACES[method]
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == 'float_log':
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == 'float':
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == 'int':
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == 'categorical':
            params[name] = trial.suggest_categorical(name, spec[1])
    params['scheduler'] = 'cosine'
    return params


def run_trial(method, params, signal, coords, device, iters, batch_size, seed):
    set_seed(seed)
    is_3d = signal.ndim == 3

    train_cfg = {k: params[k] for k in TRAIN_KEYS if k in params}
    model_kw  = {k: v for k, v in params.items() if k not in TRAIN_KEYS}

    hf = model_kw.pop('hidden_features', 256)
    hl = model_kw.pop('hidden_layers', 3)

    # sl2a: init_method and linear_init_type are fixed
    if method == 'sl2a':
        model_kw.setdefault('init_method', 'xavier_uniform')
        model_kw.setdefault('linear_init_type', 'kaiming_uniform')

    in_features = 3 if is_3d else 2
    model = get_INR(
        method=method,
        in_features=in_features,
        hidden_features=hf,
        hidden_layers=hl,
        out_features=1,
        **model_kw,
    ).to(device)

    # incode / cosmo need GT image for their Harmonizer (2D ResNet, expects [1,3,H,W])
    if hasattr(model, 'set_gt'):
        gt_t = torch.from_numpy(signal)
        if gt_t.dim() == 2:
            gt_t = gt_t[None, None].expand(1, 3, -1, -1)
        elif gt_t.dim() == 3:
            mid = gt_t.shape[0] // 2
            gt_t = gt_t[mid][None, None].expand(1, 3, -1, -1)
        model.set_gt(gt_t.to(device))

    signal_flat = torch.from_numpy(signal).reshape(-1).to(device)
    best_psnr, *_ = train_one(
        model, coords, signal_flat, signal.shape,
        train_cfg, iters, max(iters // 10, 1),
        device, save_dir=None, is_3d=is_3d, batch_size=batch_size,
    )
    return best_psnr


def load_signal(signal_name, cfg, length, seed):
    fn = cfg['fn']
    return fn(length=length, bandwidth_label=0.5, seed=seed)


def save_best_to_yaml(signal_name, best_params_per_method):
    yaml_path = os.path.join(
        _ROOT, 'configs', 'experiments', 'synthetic', f'{signal_name}.yaml')

    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    data.setdefault('methods', {})
    for method, params in best_params_per_method.items():
        data['methods'][method] = params

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f'\n[saved] best params written to {yaml_path}')


def parse_args():
    p = argparse.ArgumentParser(description='Optuna hyperparameter search for INR benchmark')
    p.add_argument('--signal', required=True, choices=list(SIGNALS.keys()))
    p.add_argument('--methods', nargs='+', default=None,
                   help='Methods to tune (default: all)')
    p.add_argument('--n_trials', type=int, default=40,
                   help='Optuna trials per method (default: 40)')
    p.add_argument('--iters', type=int, default=2000,
                   help='Training iters per trial (default: 2000; use fewer for speed)')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--batch_size', type=int, default=65536,
                   help='Mini-batch size (default: 65536; set 0 for full-batch)')
    p.add_argument('--signal_length', type=int, default=None,
                   help='Override signal resolution for search (e.g. 256 for speed)')
    p.add_argument('--device', default='auto')
    p.add_argument('--study_dir', default=None,
                   help='Directory to persist Optuna studies (enables resume)')
    return p.parse_args()


def main():
    args = parse_args()

    cfg = SIGNALS[args.signal]
    is_3d = cfg['ndim'] == 3
    L = args.signal_length if args.signal_length else cfg['length']

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f'[hparam_search] signal={args.signal}  device={device}  '
          f'iters/trial={args.iters}  n_trials={args.n_trials}')

    # Generate signal once
    print(f'  generating signal ...')
    signal = load_signal(args.signal, cfg, L, args.seed)

    # Precompute coordinates
    if is_3d:
        coords = make_coords_3d(signal.shape[0]).to(device)
    else:
        H, W = signal.shape
        coords = make_coords_2d(H, W).to(device)

    methods = args.methods or list(SEARCH_SPACES.keys())
    best_params_per_method = {}

    study_storage = None
    if args.study_dir:
        os.makedirs(args.study_dir, exist_ok=True)
        study_storage = f'sqlite:///{args.study_dir}/optuna.db'

    for method in methods:
        if method not in SEARCH_SPACES:
            print(f'  [skip] {method}: no search space defined')
            continue

        print(f'\n{"="*60}')
        print(f'  Tuning {method}  ({args.n_trials} trials)')
        print(f'{"="*60}')

        def objective(trial):
            params = suggest_params(trial, method)
            try:
                psnr = run_trial(
                    method, params, signal, coords, device,
                    args.iters, args.batch_size, args.seed,
                )
            except Exception as e:
                print(f'    [trial {trial.number}] ERROR: {e}')
                raise optuna.exceptions.TrialPruned()
            print(f'    trial {trial.number:3d}  PSNR={psnr:.2f}  params={params}')
            return psnr

        study = optuna.create_study(
            direction='maximize',
            study_name=f'{args.signal}_{method}',
            storage=study_storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best = study.best_trial
        print(f'\n  >>> {method} best PSNR: {best.value:.2f} dB')
        print(f'      params: {best.params}')

        best_params_per_method[method] = best.params

    # Save results JSON — one combined file, plus per-method files (for parallel jobs)
    results_dir = os.path.join(_ROOT, 'results', 'hparam_search')
    os.makedirs(results_dir, exist_ok=True)

    for method, params in best_params_per_method.items():
        per_method_path = os.path.join(results_dir, f'{args.signal}_best_params_{method}.json')
        with open(per_method_path, 'w') as f:
            json.dump({method: params}, f, indent=2)

    json_path = os.path.join(results_dir, f'{args.signal}_best_params.json')
    with open(json_path, 'w') as f:
        json.dump(best_params_per_method, f, indent=2)
    print(f'\n[saved] {json_path}')

    # Write best params back to signal YAML
    save_best_to_yaml(args.signal, best_params_per_method)


if __name__ == '__main__':
    main()
