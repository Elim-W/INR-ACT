"""One-off: parse div2k_img2_{method}.txt files and write image_fitting yamls
using the exact hyperparameters from the search (no rounding)."""

import os
from pathlib import Path
import yaml

SRC = Path('/scratch/liyues_root/liyues/shared_data/yiting_donghua/benchmark/search_hyper')
DST = Path('/scratch/liyues_root/liyues/shared_data/yiting_donghua/configs/experiments/image_fitting')
DST.mkdir(parents=True, exist_ok=True)

METHODS = ['siren', 'wire', 'gauss', 'finer', 'gf', 'wf',
           'staf', 'pemlp', 'incode', 'sl2a', 'cosmo']


def _cast(raw):
    s = raw.strip()
    if s == 'True':  return True
    if s == 'False': return False
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    # Python-literal tuple/list like "(0.5, 5.0)" or "[1, 2, 3]"
    if (s.startswith('(') and s.endswith(')')) or \
       (s.startswith('[') and s.endswith(']')):
        import ast
        try: return list(ast.literal_eval(s))
        except Exception: pass
    return s


def parse_txt(path):
    searched, fixed = {}, {}
    section = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('searched params'):
                section = 'searched'; continue
            if line.startswith('fixed params'):
                section = 'fixed'; continue
            if section == 'searched' and line.startswith('  '):
                k, v = line.strip().split(': ', 1)
                searched[k] = _cast(v)
            elif section == 'fixed' and line.startswith('  '):
                k, v = line.strip().split(': ', 1)
                fixed[k] = _cast(v)
    return searched, fixed


def build_yaml(method, searched, fixed):
    model = {'in_features': 2, 'hidden_features': 256,
             'hidden_layers': 3, 'out_features': 3}
    training = {'num_epochs': 4000, 'batch_size': -1,
                'scheduler': 'cosine', 'log_every': 100, 'save_every': 500}

    for k, v in fixed.items():
        if k.startswith('model.'):
            model[k[6:]] = v
        elif k.startswith('training.'):
            training[k[9:]] = v

    for k, v in searched.items():
        if k == 'lr':
            training['lr'] = v
        elif k == 'scheduler':
            training['scheduler'] = v
        else:
            model[k] = v

    return {
        'task':          'image_fitting',
        'method':        method,
        'dataset':       'div2k',
        'data_root':     'data/div2k',
        'image_indices': [2],
        'device':        'auto',
        'model':         model,
        'training':      training,
        'output': {
            'save_dir':    f'results/image_fitting/{method}',
            'save_model':  True,
            'save_images': True,
        },
    }


def _fmt(v):
    if isinstance(v, bool):  return 'true' if v else 'false'
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_fmt(x) for x in v) + ']'
    if isinstance(v, float): return repr(v)
    if isinstance(v, int):   return str(v)
    if v is None:            return 'null'
    return str(v)


def dump_yaml(cfg, path):
    order = ['task', 'method', 'dataset', 'data_root', 'image_indices',
             'device', 'model', 'training', 'output']
    with open(path, 'w') as f:
        for k in order:
            if k not in cfg: continue
            v = cfg[k]
            if isinstance(v, dict):
                f.write(f'{k}:\n')
                for sk, sv in v.items():
                    f.write(f'  {sk}: {_fmt(sv)}\n')
                f.write('\n')
            else:
                f.write(f'{k}: {_fmt(v)}\n')


for m in METHODS:
    txt = SRC / f'div2k_img2_{m}.txt'
    if not txt.exists():
        print(f'  [skip] {m}: {txt.name} not found')
        continue
    searched, fixed = parse_txt(txt)
    cfg = build_yaml(m, searched, fixed)
    out = DST / f'{m}.yaml'
    dump_yaml(cfg, out)
    print(f'  [write] {out}  (lr={cfg["training"]["lr"]})')

print('\ndone.')
