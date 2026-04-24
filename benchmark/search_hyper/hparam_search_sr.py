"""
Hyperparameter search on image_super_resolution for all INR methods.

Mirrors hparam_search_image.py but runs the SR training loop (train on the
LR grid, evaluate on HR) so the best params can be fed straight into the
SR benchmark.

Usage:
    python benchmark/search_hyper/hparam_search_sr.py --image_idx 1
    python benchmark/search_hyper/hparam_search_sr.py --image_idx 1 --methods siren incode --n_trials 30
    python benchmark/search_hyper/hparam_search_sr.py --dataset kodak --image_idx 1 --iters 2000 --scale_factor 4
    python benchmark/search_hyper/hparam_search_sr.py --dataset div2k --image_idx 2 --max_size 512

As each method finishes, a per-method txt record is written to the same
folder as this script. A combined JSON is written at the end. If
--write_yaml is passed, best params are merged into
    configs/experiments/image_super_resolution/<method>.yaml
"""

import os
import sys
import json
import argparse
import datetime
import yaml
import torch

# File is at benchmark/search_hyper/hparam_search_sr.py — go up TWO dirs
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from benchmark.methods.models import get_INR, BENCHMARK_DEFAULTS, TRAIN_KEYS
from benchmark.datasets import get_dataset
from benchmark.tasks import image_super_resolution


# ---------------------------------------------------------------------------
# Per-method search spaces (SR-specific; intentionally the same ranges as
# image_fitting since the INR architectures are identical — SR just changes
# the training signal, not the model. Adjust freely.)
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
    return params


# ---------------------------------------------------------------------------
# Build a per-trial cfg dict that image_super_resolution.run() can consume
# ---------------------------------------------------------------------------

def build_trial_cfg(method, params, iters, batch_size, scale_factor, eval_epoch):
    defaults = dict(BENCHMARK_DEFAULTS[method])

    model_cfg = {
        'in_features':     2,
        'hidden_features': defaults.pop('hidden_features', 256),
        'hidden_layers':   defaults.pop('hidden_layers', 3),
        'out_features':    3,
    }
    train_cfg = {
        'num_epochs':   iters,
        'batch_size':   batch_size,
        'log_every':    iters,   # only log at the end — per-trial overhead minimal
        'save_every':   iters * 10,
        'scale_factor': scale_factor,
        'eval_epoch':   eval_epoch,
    }

    for k, v in defaults.items():
        if k in TRAIN_KEYS:
            train_cfg.setdefault(k, v)
        else:
            model_cfg.setdefault(k, v)

    for k, v in params.items():
        if k in TRAIN_KEYS:
            train_cfg[k] = v
        else:
            model_cfg[k] = v

    if method == 'siren':
        model_cfg.setdefault('outermost_linear', True)

    return {
        'task':     'image_super_resolution',
        'method':   method,
        'model':    model_cfg,
        'training': train_cfg,
    }


def run_trial(method, params, coords, pixels, meta, device,
              iters, batch_size, scale_factor, eval_epoch):
    cfg = build_trial_cfg(method, params, iters, batch_size, scale_factor, eval_epoch)
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

    result = image_super_resolution.run(
        model=model, coords=coords, pixels=pixels,
        meta=meta, cfg=cfg, device=device, save_dir=None,
    )
    return result['final_psnr']


# ---------------------------------------------------------------------------
# Per-method txt record
# ---------------------------------------------------------------------------

def _count_real_params(model):
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if p.is_complex():
            n *= 2
        total += n
    return total


def _full_config_and_params(method, best_params, iters, batch_size, scale_factor, eval_epoch):
    cfg = build_trial_cfg(method, best_params, iters, batch_size, scale_factor, eval_epoch)
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


def _trial_log_path(tag, method):
    return os.path.join(_SCRIPT_DIR, f'{tag}_{method}_trials.txt')


def _reset_trial_log(log_path, method, param_keys, tag, meta, iters,
                     scale_factor, eval_epoch):
    with open(log_path, 'w') as f:
        f.write(f'# task:         image_super_resolution\n')
        f.write(f'# method:       {method}\n')
        f.write(f'# tag:          {tag}\n')
        f.write(f'# image:        {meta["name"]}  ({meta["H"]}x{meta["W"]})\n')
        f.write(f'# scale_factor: {scale_factor}\n')
        f.write(f'# eval_epoch:   {eval_epoch}\n')
        f.write(f'# iters/trial:  {iters}\n')
        f.write(f'# started:      {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('# columns below are tab-separated; psnr=NaN means trial errored\n')
        f.write('trial\tpsnr_HR\t' + '\t'.join(param_keys) + '\n')


def _append_trial(log_path, trial_num, psnr_val, params, param_keys):
    with open(log_path, 'a') as f:
        psnr_str = 'NaN' if psnr_val != psnr_val else f'{psnr_val:.4f}'  # NaN check
        vals = [str(trial_num), psnr_str]
        for k in param_keys:
            v = params.get(k, '')
            vals.append(f'{v:.6g}' if isinstance(v, float) else str(v))
        f.write('\t'.join(vals) + '\n')


def save_method_txt(method, best_trial, tag, meta, n_trials, iters,
                    batch_size, scale_factor, eval_epoch):
    cfg, n_params = _full_config_and_params(
        method, best_trial.params, iters, batch_size, scale_factor, eval_epoch)
    searched_keys = set(best_trial.params.keys())

    txt_path = os.path.join(_SCRIPT_DIR, f'{tag}_{method}.txt')
    lines = [
        f'task:            image_super_resolution',
        f'method:          {method}',
        f'tag:             {tag}',
        f'image:           {meta["name"]}  ({meta["H"]}x{meta["W"]})',
        f'scale_factor:    {scale_factor}',
        f'eval_epoch:      {eval_epoch}',
        f'n_trials:        {n_trials}',
        f'iters_per_trial: {iters}',
        f'timestamp:       {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'best_psnr(HR):   {best_trial.value:.4f} dB',
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
                continue
            lines.append(f'  {section}.{k}: {v}')

    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  [saved txt] {txt_path}  ({n_params:,} params)')


# ---------------------------------------------------------------------------
# Write best params back into configs/experiments/image_super_resolution/<method>.yaml
# ---------------------------------------------------------------------------

def update_method_yaml(method, params):
    yaml_path = os.path.join(
        _ROOT, 'configs', 'experiments', 'image_super_resolution', f'{method}.yaml')
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
        description='Optuna hyperparameter search on image_super_resolution (single image)')
    p.add_argument('--dataset', default='kodak', choices=['kodak', 'div2k'])
    p.add_argument('--data_root', default=None,
                   help='Default: data/<dataset>')
    p.add_argument('--image_idx', type=int, default=1,
                   help='1-based image index (default: 1)')
    p.add_argument('--max_size', type=int, default=None,
                   help='Resize longer side to N pixels (div2k only)')
    p.add_argument('--methods', nargs='+', default=None,
                   help='Methods to tune (default: all in SEARCH_SPACES)')
    p.add_argument('--n_trials', type=int, default=40)
    p.add_argument('--iters', type=int, default=2000,
                   help='Training iters per trial (default: 2000)')
    p.add_argument('--batch_size', type=int, default=-1)
    p.add_argument('--scale_factor', type=int, default=4,
                   help='HR/LR downsample factor (default: 4)')
    p.add_argument('--eval_epoch', type=int, default=0,
                   help='Epoch at which HR PSNR starts being the selection metric '
                        '(default: 0 — HR PSNR drives selection from iter 1)')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='auto')
    p.add_argument('--study_dir', default=None,
                   help='Directory to persist Optuna studies (enables resume)')
    p.add_argument('--write_yaml', action='store_true',
                   help='Merge best params into configs/experiments/image_super_resolution/<method>.yaml')
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

    coords, pixels, meta = next(iter(dataset.iter_images()))
    coords = coords.to(device)
    pixels = pixels.to(device)

    print(f'[hparam_search_sr] dataset={args.dataset} image={meta["name"]} '
          f'({meta["H"]}×{meta["W"]})  device={device}  '
          f'scale={args.scale_factor}  eval_epoch={args.eval_epoch}  '
          f'iters/trial={args.iters}  n_trials={args.n_trials}')

    methods = args.methods or list(SEARCH_SPACES.keys())
    tag = f'sr_{args.dataset}_img{args.image_idx}_x{args.scale_factor}'

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

        log_path = _trial_log_path(tag, method)
        param_keys = list(SEARCH_SPACES[method].keys())
        _reset_trial_log(log_path, method, param_keys, tag, meta, args.iters,
                         args.scale_factor, args.eval_epoch)

        def objective(trial, _method=method, _log_path=log_path, _param_keys=param_keys):
            params = suggest_params(trial, _method)
            try:
                psnr = run_trial(
                    _method, params, coords, pixels, meta, device,
                    args.iters, args.batch_size, args.scale_factor, args.eval_epoch,
                )
            except Exception as e:
                print(f'    [trial {trial.number}] ERROR: {e}')
                _append_trial(_log_path, trial.number, float('nan'), params, _param_keys)
                raise optuna.exceptions.TrialPruned()
            print(f'    trial {trial.number:3d}  HR-PSNR={psnr:.2f}  params={params}')
            _append_trial(_log_path, trial.number, psnr, params, _param_keys)
            return psnr

        study = optuna.create_study(
            direction='maximize',
            study_name=f'sr_{args.dataset}_img{args.image_idx}_x{args.scale_factor}_{method}',
            storage=study_storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best = study.best_trial
        print(f'\n  >>> {method} best HR-PSNR: {best.value:.2f} dB')
        print(f'      params: {best.params}')

        full_cfg, n_params = _full_config_and_params(
            method, best.params, args.iters, args.batch_size,
            args.scale_factor, args.eval_epoch)
        best_per_method[method] = {
            'best_psnr':    best.value,
            'n_params':     n_params,
            'searched':     best.params,
            'model':        full_cfg['model'],
            'training':     {k: v for k, v in full_cfg['training'].items()
                             if k in ('lr', 'scheduler', 'scale_factor', 'eval_epoch')},
        }

        save_method_txt(method, best, tag, meta, args.n_trials, args.iters,
                        args.batch_size, args.scale_factor, args.eval_epoch)

    # ---- Combined JSON (merge with existing) ----
    json_path = os.path.join(_SCRIPT_DIR, f'{tag}_best_params.json')
    merged = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                merged = json.load(f) or {}
        except Exception:
            merged = {}
    merged.update(best_per_method)
    with open(json_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'\n[saved] {json_path}  ({len(merged)} methods total)')

    if args.write_yaml:
        print('\n[write_yaml] merging best params into per-method configs...')
        for method, params in best_per_method.items():
            update_method_yaml(method, params)


if __name__ == '__main__':
    main()
