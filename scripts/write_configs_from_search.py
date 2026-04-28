#!/usr/bin/env python
"""
Write per-method config YAMLs from a multi-image hparam search run.

Reads the per-method best-result txt files produced by
benchmark/search_hyper_multi/hparam_search_*_multi.py and writes
configs/experiments/<task>/<method>.yaml for each one.

Default behavior is the image_fitting task; pass --task /
--results_dir / --tag to point it at inpaint / denoise / sr.

Examples
--------
    # fitting (default tag)
    python scripts/write_configs_from_search.py

    # super resolution
    python scripts/write_configs_from_search.py \
        --task image_super_resolution \
        --tag multi_sr_div2k_214_221_527_x8_mean \
        --extra scale_factor=8 eval_epoch=0

    # inpaint
    python scripts/write_configs_from_search.py \
        --task image_inpainting \
        --tag multi_inp_div2k_214_221_527_r0p20_mean \
        --extra sampling_ratio=0.2 mask_seed=0
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from glob import glob

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _coerce(v: str):
    v = v.strip()
    if v == 'True':  return True
    if v == 'False': return False
    if v == 'None':  return None
    try:    return int(v)
    except: pass
    try:    return float(v)
    except: pass
    if v.startswith('[') and v.endswith(']'):
        inside = v[1:-1].strip()
        if not inside:
            return []
        return [_coerce(x) for x in inside.split(',')]
    if v.startswith('(') and v.endswith(')'):
        inside = v[1:-1].strip()
        return [_coerce(x) for x in inside.split(',') if x.strip()]
    return v


def parse_method_txt(path: str) -> dict:
    """Return {'method': str, 'searched': {...}, 'model': {...},
              'training': {...}, 'best_score': float}."""
    with open(path) as f:
        text = f.read()

    out = {'searched': {}, 'model': {}, 'training': {}, 'search_images': []}

    m = re.search(r'^method:\s*(\S+)', text, re.M)
    out['method'] = m.group(1) if m else None

    m = re.search(r'^best_score:\s*([\-\d.]+)', text, re.M)
    out['best_score'] = float(m.group(1)) if m else None

    # image[0]:  0214 (339x510)  best_PSNR=...   -> [214, 221, 527]
    for m in re.finditer(r'image\[\d+\]:\s*0*(\d+)\s', text):
        out['search_images'].append(int(m.group(1)))

    # searched params block
    m = re.search(r'searched params \(Optuna\):\n((?:[ \t]+\S.*\n?)+)', text)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            out['searched'][k.strip()] = _coerce(v)

    # fixed params block — keys look like "model.foo: bar" / "training.foo: bar"
    m = re.search(r'fixed params \(BENCHMARK_DEFAULTS\):\n((?:[ \t]+\S.*\n?)+)', text)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip()
            value = _coerce(v)
            if '.' in k:
                section, sub = k.split('.', 1)
                if section in out:
                    out[section][sub] = value
            else:
                # unknown bucket — drop into model by default
                out['model'][k] = value

    return out


# ---------------------------------------------------------------------------
# YAML construction
# ---------------------------------------------------------------------------

TRAIN_KEYS = {'lr', 'scheduler'}


def build_yaml_dict(parsed: dict, args) -> dict:
    method = parsed['method']
    model = dict(parsed['model'])
    training = dict(parsed['training'])

    for k, v in parsed['searched'].items():
        if k in TRAIN_KEYS:
            training[k] = v
        else:
            model[k] = v

    if 'num_epochs' not in training:
        training['num_epochs'] = args.iters
    training.setdefault('log_every', max(1, args.iters // 10))
    training.setdefault('save_every', args.iters)

    config_dir = args.config_dir or args.task
    data = {
        'task':          args.task,
        'method':        method,
        'dataset':       args.dataset,
        'data_root':     args.data_root,
        'image_indices': list(args.image_indices),
        'downscale':     args.downscale,
        'device':        'auto',
        'model':         model,
        'training':      training,
        'output': {
            'save_dir':    f'results/{config_dir}/{method}',
            'save_model':  True,
            'save_images': True,
        },
    }

    for kv in args.extra or []:
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        data[k.strip()] = _coerce(v)

    data['_search'] = {
        'tag':           args.tag,
        'results_dir':   args.results_dir,
        'image_indices': list(parsed.get('search_images') or []),
        'best_score':    parsed.get('best_score'),
    }

    return data


def dump_yaml(data: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Write per-method config YAMLs from a hparam search run.')
    p.add_argument('--results_dir',
                   default='benchmark/search_hyper_multi',
                   help='Folder containing the per-method result txt files.')
    p.add_argument('--tag', default='multi_div2k_214_221_527_mean',
                   help='Tag prefix used when the search ran. The script reads '
                        '<results_dir>/<tag>_<method>.txt for each method.')
    p.add_argument('--task', default='image_fitting',
                   help='Task type — written to the yaml `task:` field, used '
                        'by the runner to pick a task module.')
    p.add_argument('--config_dir', default=None,
                   help='Subfolder name under configs/experiments/ AND '
                        'results/. Defaults to --task. Use this to keep the '
                        'task module unchanged but write configs into a '
                        'different folder (e.g. image_fitting_3mean).')
    p.add_argument('--out_dir', default=None,
                   help='Override output dir entirely. Default: '
                        'configs/experiments/<config_dir>/')
    p.add_argument('--dataset', default='div2k')
    p.add_argument('--data_root', default='data/validation')
    p.add_argument('--image_indices', nargs='*', type=int, default=[],
                   help='Bake these image indices into yaml. Default: empty '
                        '(left blank for the user to fill in).')
    p.add_argument('--downscale', type=int, default=4)
    p.add_argument('--iters', type=int, default=4000,
                   help='num_epochs to bake into training: (default 4000).')
    p.add_argument('--methods', nargs='+', default=None,
                   help='Limit to these methods. Default: every txt found.')
    p.add_argument('--extra', nargs='+', default=None,
                   help='Extra top-level key=val pairs to inject (e.g. '
                        'scale_factor=8 sampling_ratio=0.2).')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.join(REPO_ROOT, args.results_dir)
    config_dir = args.config_dir or args.task
    out_dir = args.out_dir or os.path.join(REPO_ROOT, 'configs', 'experiments',
                                           config_dir)

    pattern = os.path.join(results_dir, f'{args.tag}_*.txt')
    candidates = sorted(p for p in glob(pattern)
                        if not p.endswith('_trials.txt'))
    if not candidates:
        print(f'[error] no result files match {pattern}', file=sys.stderr)
        sys.exit(1)

    print(f'[scan] {len(candidates)} result files in {results_dir}')
    print(f'[out]  {out_dir}')

    n_written = 0
    for path in candidates:
        parsed = parse_method_txt(path)
        method = parsed['method']
        if method is None:
            print(f'  [skip] {os.path.basename(path)}: no method line')
            continue
        if args.methods and method not in args.methods:
            continue

        data = build_yaml_dict(parsed, args)
        out_path = os.path.join(out_dir, f'{method}.yaml')

        score = parsed.get('best_score')
        score_s = f'  best={score:.3f}' if score is not None else ''
        print(f'  {method:8s} -> {os.path.relpath(out_path, REPO_ROOT)}{score_s}')

        if not args.dry_run:
            dump_yaml(data, out_path)
            n_written += 1

    print(f'\n[done] wrote {n_written} yaml files'
          + (' (dry run)' if args.dry_run else ''))


if __name__ == '__main__':
    main()
