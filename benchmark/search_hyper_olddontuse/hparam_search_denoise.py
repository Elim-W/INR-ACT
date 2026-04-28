"""
Hyperparameter search on image_denoising for all INR methods.

Mirrors hparam_search_image.py / hparam_search_inpaint.py. The INR is
trained on a noisy image and evaluated against the CLEAN ground truth.

IMPORTANT: the noise realization is seeded (default --noise_seed 0) and is
FIXED across all trials of all methods, so PSNRs from different
hyperparameter configs are directly comparable.

Usage:
    python benchmark/search_hyper/hparam_search_denoise.py --image_idx 1
    python benchmark/search_hyper/hparam_search_denoise.py --dataset div2k --image_idx 127 --max_size 512 --noise_sigma 0.1
    python benchmark/search_hyper/hparam_search_denoise.py --image_idx 1 --noise_type poisson_gaussian --noise_tau 40 --noise_readout_snr 2
    python benchmark/search_hyper/hparam_search_denoise.py --image_idx 1 --methods siren incode --n_trials 30

As each method finishes, a per-method txt record is written next to this
script. A combined JSON is written at the end. If --write_yaml is passed,
best params are merged into
    configs/experiments/image_denoising/<method>.yaml
"""

import os
import sys
import json
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
from benchmark.tasks import image_denoising


# ---------------------------------------------------------------------------
# Per-method search spaces — same ranges as the fitting/inpainting searches.
# For denoising, lr is often the most important knob (too high → overfit noise,
# too low → under-fit signal). The per-method omega/scale ranges are kept the
# same so architecture-level comparisons remain consistent.
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
# Build a per-trial cfg dict that image_denoising.run() can consume
# ---------------------------------------------------------------------------

def build_trial_cfg(method, params, iters, batch_size, noise_cfg):
    """noise_cfg: dict with keys noise_type, noise_sigma/noise_tau/..., noise_seed."""
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
    train_cfg.update(noise_cfg)

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
        'task':     'image_denoising',
        'method':   method,
        'model':    model_cfg,
        'training': train_cfg,
    }


def run_trial(method, params, coords, pixels, meta, device,
              iters, batch_size, noise_cfg):
    cfg = build_trial_cfg(method, params, iters, batch_size, noise_cfg)
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

    result = image_denoising.run(
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


def _full_config_and_params(method, best_params, iters, batch_size, noise_cfg):
    cfg = build_trial_cfg(method, best_params, iters, batch_size, noise_cfg)
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


def _reset_trial_log(log_path, method, param_keys, tag, meta, iters, noise_cfg):
    with open(log_path, 'w') as f:
        f.write(f'# task:         image_denoising\n')
        f.write(f'# method:       {method}\n')
        f.write(f'# tag:          {tag}\n')
        f.write(f'# image:        {meta["name"]}  ({meta["H"]}x{meta["W"]})\n')
        for k, v in noise_cfg.items():
            f.write(f'# {k:12s} {v}\n')
        f.write(f'# iters/trial:  {iters}\n')
        f.write(f'# started:      {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('# columns below are tab-separated; psnr=NaN means trial errored\n')
        f.write('trial\tpsnr\t' + '\t'.join(param_keys) + '\n')


def _append_trial(log_path, trial_num, psnr_val, params, param_keys):
    with open(log_path, 'a') as f:
        psnr_str = 'NaN' if psnr_val != psnr_val else f'{psnr_val:.4f}'  # NaN check
        vals = [str(trial_num), psnr_str]
        for k in param_keys:
            v = params.get(k, '')
            vals.append(f'{v:.6g}' if isinstance(v, float) else str(v))
        f.write('\t'.join(vals) + '\n')


def save_method_txt(method, best_trial, tag, meta, n_trials, iters,
                    batch_size, noise_cfg):
    cfg, n_params = _full_config_and_params(
        method, best_trial.params, iters, batch_size, noise_cfg)
    searched_keys = set(best_trial.params.keys())

    txt_path = os.path.join(_SCRIPT_DIR, f'{tag}_{method}.txt')
    lines = [
        f'task:            image_denoising',
        f'method:          {method}',
        f'tag:             {tag}',
        f'image:           {meta["name"]}  ({meta["H"]}x{meta["W"]})',
    ]
    for k, v in noise_cfg.items():
        lines.append(f'{k:16s} {v}')
    lines += [
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
                continue
            lines.append(f'  {section}.{k}: {v}')

    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  [saved txt] {txt_path}  ({n_params:,} params)')


# ---------------------------------------------------------------------------
# Write best params back into configs/experiments/image_denoising/<method>.yaml
# ---------------------------------------------------------------------------

def update_method_yaml(method, params):
    yaml_path = os.path.join(
        _ROOT, 'configs', 'experiments', 'image_denoising', f'{method}.yaml')
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
        description='Optuna hyperparameter search on image_denoising (single image)')
    p.add_argument('--dataset', default='div2k', choices=['kodak', 'div2k'])
    p.add_argument('--data_root', default=None)
    p.add_argument('--image_idx', type=int, default=1)
    p.add_argument('--max_size', type=int, default=None,
                   help='Resize longer side to N pixels (div2k only)')
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--n_trials', type=int, default=40)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=-1)

    # Noise-model knobs
    p.add_argument('--noise_type', default='gaussian',
                   choices=['gaussian', 'poisson_gaussian'])
    p.add_argument('--noise_sigma', type=float, default=0.1,
                   help='Std of Gaussian noise in [0,1] pixel range (default 0.1)')
    p.add_argument('--noise_tau', type=float, default=40.0,
                   help='Photon scale for poisson_gaussian (INCODE default 40)')
    p.add_argument('--noise_readout_snr', type=float, default=2.0,
                   help='Readout-noise std for poisson_gaussian (INCODE default 2)')
    p.add_argument('--noise_seed', type=int, default=0,
                   help='Seed for the noise realization; FIXED across all trials '
                        'so PSNRs are comparable (default: 0)')

    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='auto')
    p.add_argument('--study_dir', default=None)
    p.add_argument('--write_yaml', action='store_true')
    return p.parse_args()


def _build_noise_cfg(args):
    """Collect only the fields image_denoising.run() actually reads."""
    c = {'noise_type': args.noise_type, 'noise_seed': args.noise_seed}
    if args.noise_type == 'gaussian':
        c['noise_sigma'] = args.noise_sigma
    else:
        c['noise_tau'] = args.noise_tau
        c['noise_readout_snr'] = args.noise_readout_snr
    return c


def _noise_tag(args):
    if args.noise_type == 'gaussian':
        return f'g{args.noise_sigma:.2f}'.replace('.', 'p')  # e.g. g0p10
    return f'pg_t{int(args.noise_tau)}_r{args.noise_readout_snr:.1f}'.replace('.', 'p')


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

    noise_cfg = _build_noise_cfg(args)
    print(f'[hparam_search_denoise] dataset={args.dataset} image={meta["name"]} '
          f'({meta["H"]}×{meta["W"]})  device={device}  noise={noise_cfg}  '
          f'iters/trial={args.iters}  n_trials={args.n_trials}')

    methods = args.methods or list(SEARCH_SPACES.keys())
    tag = f'den_{args.dataset}_img{args.image_idx}_{_noise_tag(args)}'

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
        _reset_trial_log(log_path, method, param_keys, tag, meta, args.iters, noise_cfg)

        def objective(trial, _method=method, _log_path=log_path, _param_keys=param_keys):
            params = suggest_params(trial, _method)
            try:
                psnr_val = run_trial(
                    _method, params, coords, pixels, meta, device,
                    args.iters, args.batch_size, noise_cfg,
                )
            except Exception as e:
                print(f'    [trial {trial.number}] ERROR: {e}')
                _append_trial(_log_path, trial.number, float('nan'), params, _param_keys)
                raise optuna.exceptions.TrialPruned()
            print(f'    trial {trial.number:3d}  PSNR={psnr_val:.2f}  params={params}')
            _append_trial(_log_path, trial.number, psnr_val, params, _param_keys)
            return psnr_val

        study = optuna.create_study(
            direction='maximize',
            study_name=f'{tag}_{method}',
            storage=study_storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best = study.best_trial
        print(f'\n  >>> {method} best PSNR: {best.value:.2f} dB')
        print(f'      params: {best.params}')

        full_cfg, n_params = _full_config_and_params(
            method, best.params, args.iters, args.batch_size, noise_cfg)
        best_per_method[method] = {
            'best_psnr':   best.value,
            'n_params':    n_params,
            'searched':    best.params,
            'model':       full_cfg['model'],
            'training':    {k: v for k, v in full_cfg['training'].items()
                            if k in ('lr', 'scheduler') or k.startswith('noise_')},
        }

        save_method_txt(method, best, tag, meta, args.n_trials, args.iters,
                        args.batch_size, noise_cfg)

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
