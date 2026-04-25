"""
Multi-image hyperparameter search on image_fitting.

Each trial fits the SAME (lr, omega, ...) on K images and aggregates the K
PSNRs into a single score that Optuna optimizes. Avoids over-fitting
hparams to a single image (which would be unfair when different image types
prefer different hparams).

Default 3-image setup: 0214 (low-mid) + 0527 (high-mid) + 0221 (high) from
data/validation/. Three points evenly span the "useful" part of the
frequency spectrum (~35-50 dB PSNR range). Drop pure low-frequency images
on purpose — they hit very high PSNR easily and dominate any naive average.

Aggregation choices (--agg):
    mean      : (p1 + p2 + p3) / 3                        ← default
    min       : min(p1, p2, p3)                           — robust to worst case
    mean_mse  : -10 * log10(mean(10**(-pi/10)))           — weights hard images more
    max       : max(p1, p2, p3)                           — only for sanity testing

Usage:
    python benchmark/search_hyper/hparam_search_image_multi.py \\
        --data_root data/validation --image_indices 214 221 527 \\
        --max_size 512 --n_trials 20 --iters 4000

As each method finishes a per-method txt is written next to this script.
A combined JSON is written at the end.
"""

import os
import sys
import json
import math
import argparse
import datetime
import yaml
import torch

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
# Search spaces (mirrors hparam_search_image.py — same INR architectures)
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


def aggregate(psnrs, mode):
    if mode == 'mean':
        return sum(psnrs) / len(psnrs)
    if mode == 'min':
        return min(psnrs)
    if mode == 'max':
        return max(psnrs)
    if mode == 'mean_mse':
        mse_each = [10 ** (-p / 10) for p in psnrs]
        mean_mse = sum(mse_each) / len(mse_each)
        return -10 * math.log10(max(mean_mse, 1e-30))
    raise ValueError(f"unknown agg '{mode}'")


# ---------------------------------------------------------------------------
# Build per-trial cfg + run a single image's fit
# ---------------------------------------------------------------------------

def build_trial_cfg(method, params, iters, batch_size):
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
        'log_every':  iters,
        'save_every': iters * 10,
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
    return {'task': 'image_fitting', 'method': method,
            'model': model_cfg, 'training': train_cfg}


def run_single_image(method, params, coords, pixels, meta, device, iters, batch_size):
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
# Per-method txt + trial log
# ---------------------------------------------------------------------------

def _count_real_params(model):
    total = 0
    for p in model.parameters():
        if not p.requires_grad: continue
        n = p.numel()
        if p.is_complex(): n *= 2
        total += n
    return total


def _full_config_and_params(method, best_params, iters, batch_size):
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
    return cfg, _count_real_params(model)


def save_method_txt(method, best_trial, tag, image_metas, n_trials, iters,
                    batch_size, agg, per_image_psnrs_at_best):
    cfg, n_params = _full_config_and_params(method, best_trial.params, iters, batch_size)
    searched_keys = set(best_trial.params.keys())

    txt_path = os.path.join(_SCRIPT_DIR, f'{tag}_{method}.txt')
    lines = [
        f'task:            image_fitting',
        f'method:          {method}',
        f'tag:             {tag}',
        f'n_images:        {len(image_metas)}',
        *[f'  image[{i}]:     {m["name"]} ({m["H"]}x{m["W"]})  best_PSNR={p:.4f}'
          for i, (m, p) in enumerate(zip(image_metas, per_image_psnrs_at_best))],
        f'agg:             {agg}',
        f'n_trials:        {n_trials}',
        f'iters_per_trial: {iters}',
        f'timestamp:       {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'best_score:      {best_trial.value:.4f}  ({agg} of per-image PSNRs)',
        f'best_trial_#:    {best_trial.number}',
        f'n_params:        {n_params:,}',
        '',
        'searched params (Optuna):',
    ]
    for k, v in best_trial.params.items():
        lines.append(f'  {k}: {v}')

    lines += ['', 'fixed params (BENCHMARK_DEFAULTS):']
    for section in ('model', 'training'):
        for k, v in cfg[section].items():
            if k in searched_keys: continue
            if k in ('num_epochs', 'log_every', 'save_every', 'batch_size'): continue
            lines.append(f'  {section}.{k}: {v}')

    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  [saved txt] {txt_path}  ({n_params:,} params)')


def _trial_log_path(tag, method):
    return os.path.join(_SCRIPT_DIR, f'{tag}_{method}_trials.txt')


def _reset_trial_log(log_path, method, param_keys, tag, image_metas, iters, agg):
    with open(log_path, 'w') as f:
        f.write(f'# task:         image_fitting (multi-image)\n')
        f.write(f'# method:       {method}\n')
        f.write(f'# tag:          {tag}\n')
        f.write(f'# images:       {[m["name"] for m in image_metas]}\n')
        f.write(f'# agg:          {agg}\n')
        f.write(f'# iters/trial:  {iters}\n')
        f.write(f'# started:      {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('# columns: trial, score, then per-image PSNRs, then searched params\n')
        per_img = '\t'.join(f'psnr_{m["name"]}' for m in image_metas)
        f.write(f'trial\tscore\t{per_img}\t' + '\t'.join(param_keys) + '\n')


def _append_trial(log_path, trial_num, score, per_image, params, param_keys):
    with open(log_path, 'a') as f:
        score_str = 'NaN' if score != score else f'{score:.4f}'
        per = '\t'.join('NaN' if (p != p) else f'{p:.4f}' for p in per_image)
        vals = [str(trial_num), score_str, per]
        for k in param_keys:
            v = params.get(k, '')
            vals.append(f'{v:.6g}' if isinstance(v, float) else str(v))
        f.write('\t'.join(vals) + '\n')


# ---------------------------------------------------------------------------
# YAML write-back
# ---------------------------------------------------------------------------

def update_method_yaml(method, params):
    yaml_path = os.path.join(_ROOT, 'configs', 'experiments',
                             'image_fitting', f'{method}.yaml')
    if not os.path.exists(yaml_path):
        print(f'  [skip yaml] {yaml_path} not found')
        return
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault('model', {}); data.setdefault('training', {})
    for k, v in params.items():
        if k in TRAIN_KEYS: data['training'][k] = v
        else:               data['model'][k] = v
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f'  [updated] {yaml_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Optuna multi-image hyperparameter search on image_fitting')
    p.add_argument('--dataset', default='div2k', choices=['kodak', 'div2k'])
    p.add_argument('--data_root', default='data/validation',
                   help='Default: data/validation (1 high + 2 mid images)')
    p.add_argument('--image_indices', nargs='+', type=int,
                   default=[214, 221, 527],
                   help='1-based image indices (default: 214 221 527)')
    p.add_argument('--max_size', type=int, default=None,
                   help='Resize longer side to N pixels (ignored if --downscale > 1)')
    p.add_argument('--downscale', type=int, default=4,
                   help='Divide W and H by this factor (BICUBIC); default 4. '
                        'Pass 1 to disable.')
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--iters', type=int, default=4000)
    p.add_argument('--batch_size', type=int, default=-1)
    p.add_argument('--agg', default='mean',
                   choices=['mean', 'min', 'max', 'mean_mse'])
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='auto')
    p.add_argument('--study_dir', default=None)
    p.add_argument('--write_yaml', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    ds_kw = {'indices': args.image_indices, 'normalize': True}
    if args.dataset == 'div2k':
        if args.downscale and args.downscale > 1:
            ds_kw['downscale'] = args.downscale
        elif args.max_size:
            ds_kw['max_size'] = args.max_size
    dataset = get_dataset(args.dataset, args.data_root, **ds_kw)

    images = []  # list of (coords_gpu, pixels_gpu, meta)
    for coords, pixels, meta in dataset.iter_images():
        images.append((coords.to(device), pixels.to(device), meta))
    if not images:
        raise SystemExit(f'No images loaded from {args.data_root}')

    image_metas = [m for _, _, m in images]
    idx_str = '_'.join(str(i) for i in args.image_indices)
    tag = f'multi_{args.dataset}_{idx_str}_{args.agg}'

    print(f'[hparam_search_image_multi] device={device}  agg={args.agg}  '
          f'n_trials={args.n_trials}  iters/trial={args.iters}')
    print(f'  images:')
    for i, m in enumerate(image_metas):
        print(f'    [{i}] {m["name"]}  ({m["H"]}x{m["W"]})')

    methods = args.methods or list(SEARCH_SPACES.keys())

    study_storage = None
    if args.study_dir:
        os.makedirs(args.study_dir, exist_ok=True)
        study_storage = f'sqlite:///{args.study_dir}/optuna.db'

    best_per_method = {}

    for method in methods:
        if method not in SEARCH_SPACES:
            print(f'  [skip] {method}: no search space'); continue
        if method not in BENCHMARK_DEFAULTS:
            print(f'  [skip] {method}: no defaults'); continue

        print(f'\n{"="*60}\n  Tuning {method}  ({args.n_trials} trials × {len(images)} images)\n{"="*60}')

        log_path = _trial_log_path(tag, method)
        param_keys = list(SEARCH_SPACES[method].keys())
        _reset_trial_log(log_path, method, param_keys, tag, image_metas, args.iters, args.agg)

        # Per-method state to remember which (per-image PSNRs) the best trial had
        per_image_at_best = {'value': -float('inf'), 'psnrs': [float('nan')] * len(images)}

        def objective(trial, _method=method, _log=log_path, _keys=param_keys, _store=per_image_at_best):
            params = suggest_params(trial, _method)
            psnrs = []
            for coords_g, pixels_g, meta in images:
                try:
                    p = run_single_image(_method, params, coords_g, pixels_g, meta,
                                         device, args.iters, args.batch_size)
                except Exception as e:
                    print(f'    trial {trial.number}  image {meta["name"]} ERROR: {e}')
                    _append_trial(_log, trial.number, float('nan'), psnrs + [float('nan')] * (len(images) - len(psnrs)), params, _keys)
                    raise optuna.exceptions.TrialPruned()
                psnrs.append(p)
            score = aggregate(psnrs, args.agg)
            print(f'    trial {trial.number:3d}  {args.agg}={score:.2f}  per_img={[f"{p:.2f}" for p in psnrs]}  params={params}')
            _append_trial(_log, trial.number, score, psnrs, params, _keys)
            if score > _store['value']:
                _store['value'] = score
                _store['psnrs'] = list(psnrs)
            return score

        study = optuna.create_study(
            direction='maximize',
            study_name=f'{tag}_{method}',
            storage=study_storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best = study.best_trial
        print(f'\n  >>> {method} best {args.agg}={best.value:.2f} '
              f'(per-image={per_image_at_best["psnrs"]})')
        print(f'      params: {best.params}')

        full_cfg, n_params = _full_config_and_params(method, best.params, args.iters, args.batch_size)
        best_per_method[method] = {
            'best_score':     best.value,
            'agg':            args.agg,
            'per_image_psnr': {m['name']: p for m, p in zip(image_metas, per_image_at_best['psnrs'])},
            'n_params':       n_params,
            'searched':       best.params,
            'model':          full_cfg['model'],
            'training':       {k: v for k, v in full_cfg['training'].items()
                               if k in ('lr', 'scheduler')},
        }

        save_method_txt(method, best, tag, image_metas, args.n_trials, args.iters,
                        args.batch_size, args.agg, per_image_at_best['psnrs'])

    # Combined JSON (merge with existing)
    json_path = os.path.join(_SCRIPT_DIR, f'{tag}_best_params.json')
    merged = {}
    if os.path.exists(json_path):
        try: merged = json.load(open(json_path)) or {}
        except Exception: merged = {}
    merged.update(best_per_method)
    with open(json_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'\n[saved] {json_path}  ({len(merged)} methods total)')

    if args.write_yaml:
        for method, params in best_per_method.items():
            update_method_yaml(method, params['searched'])


if __name__ == '__main__':
    main()
