"""
Multi-image hyperparameter search on image_super_resolution.

Each trial fits the SAME (lr, omega, ...) on K images at the same scale
factor, then aggregates the K HR-PSNRs into one Optuna score.

Defaults:
    images: 0214 + 0221 + 0527 from data/validation
    NO downscale (HR is the original full DIV2K resolution, ~2040×1356)
    scale_factor=8   (LR = HR / 8)
    eval_epoch=0     (HR PSNR drives selection from iter 1)
    agg=mean

Note: SR training is on the LR grid (HR/8 ≈ 255×170 pixels), so per-trial
cost is small even though HR is ~2.8M pixels. Only the final HR eval is
done at full resolution.

Usage:
    python benchmark/search_hyper_multi/hparam_search_sr_multi.py \\
        --n_trials 20 --iters 4000
"""

import os, sys, json, math, argparse, datetime, yaml, torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from benchmark.methods.models import get_INR, BENCHMARK_DEFAULTS, TRAIN_KEYS
from benchmark.datasets import get_dataset
from benchmark.tasks import image_super_resolution


SEARCH_SPACES = {
    'siren':  {'lr': ('float_log', 1e-5, 1e-2), 'first_omega_0': ('float', 5.0, 60.0), 'hidden_omega_0': ('float', 5.0, 60.0)},
    'wire':   {'lr': ('float_log', 1e-4, 1e-1), 'first_omega_0': ('float', 5.0, 40.0), 'hidden_omega_0': ('float', 5.0, 40.0), 'scale': ('float_log', 1.0, 30.0)},
    'gauss':  {'lr': ('float_log', 1e-4, 1e-2), 'scale': ('float_log', 1.0, 30.0)},
    'finer':  {'lr': ('float_log', 1e-5, 1e-2), 'first_omega_0': ('float', 5.0, 60.0), 'hidden_omega_0': ('float', 5.0, 60.0)},
    'gf':     {'lr': ('float_log', 1e-4, 1e-2), 'scale': ('float', 0.5, 10.0), 'omega': ('float', 1.0, 30.0), 'first_bias_scale': ('float', 0.1, 5.0)},
    'wf':     {'lr': ('float_log', 1e-4, 1e-2), 'scale': ('float', 0.5, 10.0), 'omega_w': ('float', 1.0, 15.0), 'omega': ('float', 1.0, 15.0), 'first_bias_scale': ('float', 0.1, 5.0)},
    'staf':   {'lr': ('float_log', 1e-5, 1e-2), 'first_omega_0': ('float', 5.0, 60.0), 'hidden_omega_0': ('float', 5.0, 60.0)},
    'relu':   {'lr': ('float_log', 1e-4, 1e-2)},
    'incode': {'lr': ('float_log', 1e-5, 1e-2), 'first_omega_0': ('float', 5.0, 60.0), 'hidden_omega_0': ('float', 5.0, 60.0)},
    'sl2a':   {'lr': ('float_log', 1e-4, 1e-2)},
    'cosmo':  {'lr': ('float_log', 1e-3, 1e-1), 'beta0': ('float', 0.01, 0.5)},
}


def suggest_params(trial, method):
    out = {}
    for name, spec in SEARCH_SPACES[method].items():
        kind = spec[0]
        if kind == 'float_log':   out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == 'float':     out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == 'int':       out[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == 'categorical': out[name] = trial.suggest_categorical(name, spec[1])
    return out


def aggregate(psnrs, mode):
    if mode == 'mean': return sum(psnrs) / len(psnrs)
    if mode == 'min':  return min(psnrs)
    if mode == 'max':  return max(psnrs)
    if mode == 'mean_mse':
        m = sum(10 ** (-p / 10) for p in psnrs) / len(psnrs)
        return -10 * math.log10(max(m, 1e-30))
    raise ValueError(f"unknown agg '{mode}'")


def build_trial_cfg(method, params, iters, batch_size, scale_factor, eval_epoch):
    defaults = dict(BENCHMARK_DEFAULTS[method])
    model_cfg = {'in_features': 2,
                 'hidden_features': defaults.pop('hidden_features', 256),
                 'hidden_layers': defaults.pop('hidden_layers', 3),
                 'out_features': 3}
    train_cfg = {'num_epochs': iters, 'batch_size': batch_size,
                 'log_every': iters, 'save_every': iters * 10,
                 'scale_factor': scale_factor, 'eval_epoch': eval_epoch}
    for k, v in defaults.items():
        (train_cfg if k in TRAIN_KEYS else model_cfg).setdefault(k, v)
    for k, v in params.items():
        if k in TRAIN_KEYS: train_cfg[k] = v
        else:               model_cfg[k] = v
    if method == 'siren':
        model_cfg.setdefault('outermost_linear', True)
    return {'task': 'image_super_resolution', 'method': method,
            'model': model_cfg, 'training': train_cfg}


def run_single_image(method, params, coords, pixels, meta, device,
                     iters, batch_size, scale_factor, eval_epoch):
    cfg = build_trial_cfg(method, params, iters, batch_size, scale_factor, eval_epoch)
    mcfg = cfg['model']
    standard = ('in_features', 'hidden_features', 'hidden_layers', 'out_features')
    model = get_INR(method=method,
                    in_features=mcfg['in_features'],
                    hidden_features=mcfg['hidden_features'],
                    hidden_layers=mcfg['hidden_layers'],
                    out_features=mcfg['out_features'],
                    **{k: v for k, v in mcfg.items() if k not in standard}).to(device)
    return image_super_resolution.run(model=model, coords=coords, pixels=pixels,
                                      meta=meta, cfg=cfg, device=device, save_dir=None)['final_psnr']


def _count_real_params(model):
    t = 0
    for p in model.parameters():
        if not p.requires_grad: continue
        n = p.numel(); t += (2 * n if p.is_complex() else n)
    return t


def _full_config_and_params(method, params, iters, batch_size, scale_factor, eval_epoch):
    cfg = build_trial_cfg(method, params, iters, batch_size, scale_factor, eval_epoch)
    mcfg = cfg['model']
    standard = ('in_features', 'hidden_features', 'hidden_layers', 'out_features')
    model = get_INR(method=method,
                    in_features=mcfg['in_features'],
                    hidden_features=mcfg['hidden_features'],
                    hidden_layers=mcfg['hidden_layers'],
                    out_features=mcfg['out_features'],
                    **{k: v for k, v in mcfg.items() if k not in standard})
    return cfg, _count_real_params(model)


def save_method_txt(method, best_trial, tag, image_metas, n_trials, iters,
                    batch_size, scale_factor, eval_epoch, agg, per_image_psnrs):
    cfg, n_params = _full_config_and_params(method, best_trial.params, iters,
                                            batch_size, scale_factor, eval_epoch)
    searched_keys = set(best_trial.params.keys())
    txt_path = os.path.join(_SCRIPT_DIR, f'{tag}_{method}.txt')
    lines = [
        f'task:            image_super_resolution',
        f'method:          {method}',
        f'tag:             {tag}',
        f'n_images:        {len(image_metas)}',
        *[f'  image[{i}]:     {m["name"]} ({m["H"]}x{m["W"]})  best_HR_PSNR={p:.4f}'
          for i, (m, p) in enumerate(zip(image_metas, per_image_psnrs))],
        f'scale_factor:    {scale_factor}',
        f'eval_epoch:      {eval_epoch}',
        f'agg:             {agg}',
        f'n_trials:        {n_trials}',
        f'iters_per_trial: {iters}',
        f'timestamp:       {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'best_score:      {best_trial.value:.4f}  ({agg} of per-image HR-PSNRs)',
        f'best_trial_#:    {best_trial.number}',
        f'n_params:        {n_params:,}',
        '',
        'searched params (Optuna):',
        *[f'  {k}: {v}' for k, v in best_trial.params.items()],
        '',
        'fixed params (BENCHMARK_DEFAULTS):',
    ]
    for section in ('model', 'training'):
        for k, v in cfg[section].items():
            if k in searched_keys: continue
            if k in ('num_epochs', 'log_every', 'save_every', 'batch_size'): continue
            lines.append(f'  {section}.{k}: {v}')
    with open(txt_path, 'w') as f: f.write('\n'.join(lines) + '\n')
    print(f'  [saved txt] {txt_path}  ({n_params:,} params)')


def _trial_log_path(tag, method): return os.path.join(_SCRIPT_DIR, f'{tag}_{method}_trials.txt')


def _reset_trial_log(log_path, method, param_keys, tag, image_metas, iters, agg, scale_factor, eval_epoch):
    with open(log_path, 'w') as f:
        f.write(f'# task: image_super_resolution (multi)\n# method: {method}\n# tag: {tag}\n')
        f.write(f'# images: {[m["name"] for m in image_metas]}\n')
        f.write(f'# scale_factor: {scale_factor}\n# eval_epoch: {eval_epoch}\n')
        f.write(f'# agg: {agg}\n# iters/trial: {iters}\n')
        f.write(f'# started: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        per_img = '\t'.join(f'psnr_{m["name"]}' for m in image_metas)
        f.write(f'trial\tscore\t{per_img}\t' + '\t'.join(param_keys) + '\n')


def _append_trial(log_path, trial_num, score, per_image, params, param_keys):
    with open(log_path, 'a') as f:
        s = 'NaN' if score != score else f'{score:.4f}'
        per = '\t'.join('NaN' if (p != p) else f'{p:.4f}' for p in per_image)
        vals = [str(trial_num), s, per] + [
            (f'{params.get(k, ""):.6g}' if isinstance(params.get(k), float) else str(params.get(k, '')))
            for k in param_keys]
        f.write('\t'.join(vals) + '\n')


def update_method_yaml(method, params):
    yaml_path = os.path.join(_ROOT, 'configs', 'experiments', 'image_super_resolution', f'{method}.yaml')
    if not os.path.exists(yaml_path):
        print(f'  [skip yaml] {yaml_path} not found'); return
    with open(yaml_path) as f: data = yaml.safe_load(f) or {}
    data.setdefault('model', {}); data.setdefault('training', {})
    for k, v in params.items():
        if k in TRAIN_KEYS: data['training'][k] = v
        else:               data['model'][k] = v
    with open(yaml_path, 'w') as f: yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f'  [updated] {yaml_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='div2k', choices=['kodak', 'div2k'])
    p.add_argument('--data_root', default='data/validation')
    p.add_argument('--image_indices', nargs='+', type=int, default=[214, 221, 527])
    p.add_argument('--max_size', type=int, default=None,
                   help='Optional: resize HR longer side (default: full original DIV2K res)')
    p.add_argument('--downscale', type=int, default=1,
                   help='Optional: downscale HR (default 1 = no downscale, full original)')
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--iters', type=int, default=4000)
    p.add_argument('--batch_size', type=int, default=-1)
    p.add_argument('--scale_factor', type=int, default=8,
                   help='HR/LR downsample factor for SR (default 8)')
    p.add_argument('--eval_epoch', type=int, default=0,
                   help='Epoch from which HR PSNR drives best selection')
    p.add_argument('--agg', default='mean', choices=['mean', 'min', 'max', 'mean_mse'])
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
        if args.downscale and args.downscale > 1: ds_kw['downscale'] = args.downscale
        elif args.max_size:                       ds_kw['max_size']  = args.max_size
    dataset = get_dataset(args.dataset, args.data_root, **ds_kw)

    images = [(c.to(device), p.to(device), m) for c, p, m in dataset.iter_images()]
    if not images: raise SystemExit(f'No images loaded from {args.data_root}')
    image_metas = [m for _, _, m in images]

    idx_str = '_'.join(str(i) for i in args.image_indices)
    tag = f'multi_sr_{args.dataset}_{idx_str}_x{args.scale_factor}_{args.agg}'

    print(f'[hparam_search_sr_multi] device={device}  agg={args.agg}  '
          f'scale={args.scale_factor}  eval_epoch={args.eval_epoch}  '
          f'n_trials={args.n_trials}  iters/trial={args.iters}')
    for i, m in enumerate(image_metas):
        print(f'    [{i}] {m["name"]}  HR=({m["H"]}x{m["W"]})  '
              f'LR≈({m["H"] // args.scale_factor}x{m["W"] // args.scale_factor})')

    methods = args.methods or list(SEARCH_SPACES.keys())
    study_storage = None
    if args.study_dir:
        os.makedirs(args.study_dir, exist_ok=True)
        study_storage = f'sqlite:///{args.study_dir}/optuna.db'

    best_per_method = {}
    for method in methods:
        if method not in SEARCH_SPACES or method not in BENCHMARK_DEFAULTS:
            print(f'  [skip] {method}'); continue

        print(f'\n{"="*60}\n  Tuning {method}  ({args.n_trials} trials × {len(images)} images)\n{"="*60}')
        log_path = _trial_log_path(tag, method)
        param_keys = list(SEARCH_SPACES[method].keys())
        _reset_trial_log(log_path, method, param_keys, tag, image_metas, args.iters,
                         args.agg, args.scale_factor, args.eval_epoch)
        store = {'value': -float('inf'), 'psnrs': [float('nan')] * len(images)}

        def objective(trial, _m=method, _log=log_path, _keys=param_keys, _store=store):
            params = suggest_params(trial, _m)
            psnrs = []
            for c, p, meta in images:
                try:
                    psnrs.append(run_single_image(_m, params, c, p, meta, device,
                                                  args.iters, args.batch_size,
                                                  args.scale_factor, args.eval_epoch))
                except Exception as e:
                    print(f'    trial {trial.number} image {meta["name"]} ERROR: {e}')
                    _append_trial(_log, trial.number, float('nan'),
                                  psnrs + [float('nan')] * (len(images) - len(psnrs)),
                                  params, _keys)
                    raise optuna.exceptions.TrialPruned()
            score = aggregate(psnrs, args.agg)
            print(f'    trial {trial.number:3d}  {args.agg}={score:.2f}  per_img={[f"{p:.2f}" for p in psnrs]}  params={params}')
            _append_trial(_log, trial.number, score, psnrs, params, _keys)
            if score > _store['value']:
                _store['value'] = score; _store['psnrs'] = list(psnrs)
            return score

        study = optuna.create_study(
            direction='maximize',
            study_name=f'{tag}_{method}',
            storage=study_storage, load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best = study.best_trial
        print(f'\n  >>> {method} best HR-{args.agg}={best.value:.2f} per_img={store["psnrs"]}')
        full_cfg, n_params = _full_config_and_params(method, best.params, args.iters,
                                                     args.batch_size, args.scale_factor, args.eval_epoch)
        best_per_method[method] = {
            'best_score':     best.value, 'agg': args.agg,
            'per_image_psnr': {m['name']: p for m, p in zip(image_metas, store['psnrs'])},
            'n_params':       n_params, 'searched': best.params,
            'model':          full_cfg['model'],
            'training':       {k: v for k, v in full_cfg['training'].items()
                               if k in ('lr', 'scheduler', 'scale_factor', 'eval_epoch')},
        }
        save_method_txt(method, best, tag, image_metas, args.n_trials, args.iters,
                        args.batch_size, args.scale_factor, args.eval_epoch, args.agg, store['psnrs'])

    json_path = os.path.join(_SCRIPT_DIR, f'{tag}_best_params.json')
    merged = {}
    if os.path.exists(json_path):
        try: merged = json.load(open(json_path)) or {}
        except Exception: merged = {}
    merged.update(best_per_method)
    with open(json_path, 'w') as f: json.dump(merged, f, indent=2)
    print(f'\n[saved] {json_path}  ({len(merged)} methods total)')

    if args.write_yaml:
        for m, info in best_per_method.items():
            update_method_yaml(m, info['searched'])


if __name__ == '__main__':
    main()
