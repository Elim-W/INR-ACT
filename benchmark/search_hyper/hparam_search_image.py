"""
Hyperparameter search on image_fitting for all INR methods, using a single image.

Mirrors benchmark/hparam_search.py (synthetic signals) but runs the real
image_fitting training loop on one image from kodak / div2k so that the
best params can be used directly for the image_fitting benchmark.

Usage:
    python benchmark/search_hyper/hparam_search_image.py --image_idx 1
    python benchmark/search_hyper/hparam_search_image.py --image_idx 1 --methods siren wire --n_trials 30
    python benchmark/search_hyper/hparam_search_image.py --dataset kodak --image_idx 1 --iters 3000
    python benchmark/search_hyper/hparam_search_image.py --image_idx 1 --max_size 512  # resize to speed up

As each method finishes, a per-method txt record is written to the same
folder as this script (benchmark/search_hyper/). A combined JSON is written
at the end. If --write_yaml is passed, best params are also merged into
    configs/experiments/image_fitting/<method>.yaml
(only `training.lr`, `training.scheduler`, and searched model.* fields are
touched; `num_epochs`, `image_indices`, etc. are left alone).
"""

import os
import sys
import json
import argparse
import datetime
import yaml
import torch

# File is at benchmark/search_hyper/hparam_search_image.py — go up TWO dirs
# (search_hyper → benchmark → project root) to find the project root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from benchmark.methods.models import get_INR, BENCHMARK_DEFAULTS, TRAIN_KEYS
from benchmark.datasets import get_dataset
from benchmark.tasks import image_fitting


# ---------------------------------------------------------------------------
# Per-method search spaces (image-fitting specific — edit freely without
# affecting benchmark/hparam_search.py which targets synthetic signals).
#
# Tuple format: ('float_log', lo, hi) | ('float', lo, hi) |
#               ('int', lo, hi) | ('categorical', [v1, v2, ...])
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
        # tau/skip_conn fixed in BENCHMARK_DEFAULTS (5, False) — they change
        # the architecture / parameter budget, so fairer to search over
        # continuous hyperparams only and keep structure comparable.
        'lr':             ('float_log', 1e-5, 1e-2),
        'first_omega_0':  ('float', 5.0, 60.0),
        'hidden_omega_0': ('float', 5.0, 60.0),
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
        # degree/rank fixed in BENCHMARK_DEFAULTS (128/128) to keep the
        # parameter budget matched with other INR methods (~199K). Searching
        # them would break budget parity — do that in a separate ablation.
        'lr':             ('float_log', 1e-4, 1e-2),
    },
    'cosmo': {
        'lr':             ('float_log', 1e-3, 1e-1),
        'beta0':          ('float', 0.01, 0.5),
    },
}


def suggest_params(trial, method):
    """Sample one set of hyperparameters from SEARCH_SPACES[method]."""
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
    return params


# ---------------------------------------------------------------------------
# Build a per-trial cfg dict that image_fitting.run() can consume
# ---------------------------------------------------------------------------

def build_trial_cfg(method, params, iters, batch_size):
    """Merge BENCHMARK_DEFAULTS[method] with this trial's sampled params."""
    defaults = dict(BENCHMARK_DEFAULTS[method])

    model_cfg = {
        'in_features':     2,
        'hidden_features': defaults.pop('hidden_features', 256),
        'hidden_layers':   defaults.pop('hidden_layers', 3),
        'out_features':    3,
    }
    train_cfg = {
        'num_epochs': iters,
        'batch_size': batch_size,
        # Evaluate / log only at the end to minimise overhead per trial
        'log_every':  iters,
        'save_every': iters * 10,
    }

    # Fill in any method-specific defaults not covered by the search space
    for k, v in defaults.items():
        if k in TRAIN_KEYS:
            train_cfg.setdefault(k, v)
        else:
            model_cfg.setdefault(k, v)

    # Trial-suggested params override defaults
    for k, v in params.items():
        if k in TRAIN_KEYS:
            train_cfg[k] = v
        else:
            model_cfg[k] = v

    # SIREN expects outermost_linear for pixel regression
    if method == 'siren':
        model_cfg.setdefault('outermost_linear', True)

    return {
        'task':     'image_fitting',
        'method':   method,
        'model':    model_cfg,
        'training': train_cfg,
    }


def run_trial(method, params, coords, pixels, meta, device, iters, batch_size):
    cfg = build_trial_cfg(method, params, iters, batch_size)
    mcfg = cfg['model']

    standard = ('in_features', 'hidden_features', 'hidden_layers', 'out_features')
    model = get_INR(
        method=method,
        in_features=mcfg['in_features'],
        hidden_features=mcfg['hidden_features'],
        hidden_layers=mcfg['hidden_layers'],
        out_features=mcfg['out_features'],
        **{k: v for k, v in mcfg.items() if k not in standard},
    ).to(device)

    result = image_fitting.run(
        model=model, coords=coords, pixels=pixels,
        meta=meta, cfg=cfg, device=device, save_dir=None,
    )
    return result['final_psnr']


# ---------------------------------------------------------------------------
# Per-method txt record — written immediately when a method's search ends
# ---------------------------------------------------------------------------

def _count_real_params(model):
    """Count real-valued trainable parameters.

    Complex-valued tensors (e.g. WIRE/WF weights) are counted as 2× their
    `numel()`, since each complex number is (re, im) — two real parameters.
    Using bare `numel()` undercounts complex methods by exactly 2×.
    """
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if p.is_complex():
            n *= 2
        total += n
    return total


def _full_config_and_params(method, best_params, iters, batch_size):
    """Return (full_cfg_dict, n_params) by running build_trial_cfg + counting."""
    cfg = build_trial_cfg(method, best_params, iters, batch_size)
    mcfg = cfg['model']
    standard = ('in_features', 'hidden_features', 'hidden_layers', 'out_features')
    model = get_INR(
        method=method,
        in_features=mcfg['in_features'],
        hidden_features=mcfg['hidden_features'],
        hidden_layers=mcfg['hidden_layers'],
        out_features=mcfg['out_features'],
        **{k: v for k, v in mcfg.items() if k not in standard},
    )
    n_params = _count_real_params(model)
    return cfg, n_params


def save_method_txt(method, best_trial, tag, meta, n_trials, iters, batch_size):
    """Write one human-readable record next to this script, including the
    FULL trial config (searched + fixed-from-BENCHMARK_DEFAULTS) + param count."""
    cfg, n_params = _full_config_and_params(method, best_trial.params, iters, batch_size)
    searched_keys = set(best_trial.params.keys())

    txt_path = os.path.join(_SCRIPT_DIR, f'{tag}_{method}.txt')
    lines = [
        f'method:          {method}',
        f'tag:             {tag}',
        f'image:           {meta["name"]}  ({meta["H"]}x{meta["W"]})',
        f'n_trials:        {n_trials}',
        f'iters_per_trial: {iters}',
        f'timestamp:       {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'best_psnr:       {best_trial.value:.4f} dB',
        f'best_trial_#:    {best_trial.number}',
        f'n_params:        {n_params:,}',
        '',
        'searched params (found by Optuna):',
    ]
    for k, v in best_trial.params.items():
        lines.append(f'  {k}: {v}')

    lines.append('')
    lines.append('fixed params (from BENCHMARK_DEFAULTS, not searched):')
    for section in ('model', 'training'):
        for k, v in cfg[section].items():
            if k in searched_keys:
                continue
            if k in ('num_epochs', 'log_every', 'save_every', 'batch_size'):
                continue  # trial-loop bookkeeping, not a real hyperparam
            lines.append(f'  {section}.{k}: {v}')

    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  [saved txt] {txt_path}  ({n_params:,} params)')


# ---------------------------------------------------------------------------
# Write best params back into configs/experiments/image_fitting/<method>.yaml
# ---------------------------------------------------------------------------

def update_method_yaml(method, params):
    yaml_path = os.path.join(
        _ROOT, 'configs', 'experiments', 'image_fitting', f'{method}.yaml')
    if not os.path.exists(yaml_path):
        print(f'  [skip yaml] {yaml_path} not found')
        return

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault('model', {})
    data.setdefault('training', {})

    for k, v in params.items():
        if k in TRAIN_KEYS:
            data['training'][k] = v
        else:
            data['model'][k] = v

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f'  [updated] {yaml_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Optuna hyperparameter search on image_fitting (single image)')
    p.add_argument('--dataset', default='div2k', choices=['kodak', 'div2k'])
    p.add_argument('--data_root', default=None,
                   help='Default: data/<dataset>')
    p.add_argument('--image_idx', type=int, default=1,
                   help='1-based image index (default: 1)')
    p.add_argument('--max_size', type=int, default=None,
                   help='Resize longer side to N pixels (div2k only, '
                        'speeds up trials on large images)')
    p.add_argument('--methods', nargs='+', default=None,
                   help='Methods to tune (default: all in SEARCH_SPACES)')
    p.add_argument('--n_trials', type=int, default=40)
    p.add_argument('--iters', type=int, default=2000,
                   help='Training iters per trial (default: 2000)')
    p.add_argument('--batch_size', type=int, default=-1,
                   help='Mini-batch size; -1 = full image (default: -1)')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='auto')
    p.add_argument('--study_dir', default=None,
                   help='Directory to persist Optuna studies (enables resume)')
    p.add_argument('--write_yaml', action='store_true',
                   help='Merge best params into configs/experiments/image_fitting/<method>.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    data_root = args.data_root or os.path.join('data', args.dataset)
    ds_kw = {'indices': [args.image_idx], 'normalize': True}
    if args.dataset == 'div2k' and args.max_size:
        ds_kw['max_size'] = args.max_size
    dataset = get_dataset(args.dataset, data_root, **ds_kw)

    # Grab the one image
    coords, pixels, meta = next(iter(dataset.iter_images()))
    coords = coords.to(device)
    pixels = pixels.to(device)

    print(f'[hparam_search_image] dataset={args.dataset} image={meta["name"]} '
          f'({meta["H"]}×{meta["W"]})  device={device}  '
          f'iters/trial={args.iters}  n_trials={args.n_trials}')

    methods = args.methods or list(SEARCH_SPACES.keys())
    tag = f'{args.dataset}_img{args.image_idx}'

    study_storage = None
    if args.study_dir:
        os.makedirs(args.study_dir, exist_ok=True)
        study_storage = f'sqlite:///{args.study_dir}/optuna.db'

    best_per_method = {}

    for method in methods:
        if method not in SEARCH_SPACES:
            print(f'  [skip] {method}: no search space defined')
            continue
        if method not in BENCHMARK_DEFAULTS:
            print(f'  [skip] {method}: no BENCHMARK_DEFAULTS entry')
            continue

        print(f'\n{"="*60}\n  Tuning {method}  ({args.n_trials} trials)\n{"="*60}')

        def objective(trial, _method=method):
            params = suggest_params(trial, _method)
            try:
                psnr = run_trial(
                    _method, params, coords, pixels, meta, device,
                    args.iters, args.batch_size,
                )
            except Exception as e:
                print(f'    [trial {trial.number}] ERROR: {e}')
                raise optuna.exceptions.TrialPruned()
            print(f'    trial {trial.number:3d}  PSNR={psnr:.2f}  params={params}')
            return psnr

        study = optuna.create_study(
            direction='maximize',
            study_name=f'imgfit_{args.dataset}_img{args.image_idx}_{method}',
            storage=study_storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best = study.best_trial
        print(f'\n  >>> {method} best PSNR: {best.value:.2f} dB')
        print(f'      params: {best.params}')

        # Record the FULL config (searched + fixed defaults) and param count
        full_cfg, n_params = _full_config_and_params(
            method, best.params, args.iters, args.batch_size)
        best_per_method[method] = {
            'best_psnr':      best.value,
            'n_params':       n_params,
            'searched':       best.params,
            'model':          full_cfg['model'],
            'training':       {k: v for k, v in full_cfg['training'].items()
                               if k in ('lr', 'scheduler')},
        }

        # Immediately persist this method's result as a txt next to the script.
        # This way, if a later method crashes, earlier results are already on disk.
        save_method_txt(method, best, tag, meta, args.n_trials, args.iters, args.batch_size)

    # ---- Save combined JSON (merge with existing so parallel runs on
    #      different methods/GPUs don't overwrite each other's results) ----
    json_path = os.path.join(_SCRIPT_DIR, f'{tag}_best_params.json')
    merged = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                merged = json.load(f) or {}
        except Exception:
            merged = {}
    merged.update(best_per_method)   # new results overwrite only their own keys
    with open(json_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'\n[saved] {json_path}  ({len(merged)} methods total)')

    # ---- Optionally merge into per-method YAMLs ----
    if args.write_yaml:
        print('\n[write_yaml] merging best params into per-method configs...')
        for method, params in best_per_method.items():
            update_method_yaml(method, params)


if __name__ == '__main__':
    main()
