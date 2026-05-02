"""
Shared helpers for the benchmark entry scripts (run_experiment.py,
run_experiment_3d.py).

Everything that 2D and 3D dispatchers both need lives here:
    - load_config / apply_overrides  (YAML + dot-notation overrides)
    - get_device                     (auto / cpu / cuda selection)
    - build_model                    (method factory from cfg['model'])
    - make_save_dir                  (output directory from cfg['output'])
    - parse_cli                      (the shared argparse spec)

Anything task-family-specific (dispatch loops, per-task summary printing)
stays in the corresponding run_experiment*.py file.
"""

import argparse
import os
import torch
import yaml


# Note: each entry script must put the project root on sys.path *before*
# importing from this module, since this module itself lives inside the
# benchmark/ package.  See the 3-line prelude in run_experiment{,_3d}.py.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli(description='INR Benchmark'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', required=True,
                        help='Path to YAML config file')
    parser.add_argument('--override', nargs='*', default=[],
                        metavar='KEY=VAL',
                        help='Override any config value, e.g. training.lr=1e-3 '
                             'image_indices=[1,2,3]')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_overrides(cfg, overrides):
    """
    Apply dot-notation overrides, e.g. 'training.lr=1e-3'.
    Supports nested keys and numeric / bool / string coercion (via yaml.safe_load).
    """
    for ov in overrides:
        key, _, val_str = ov.partition('=')
        keys = key.split('.')
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            val = yaml.safe_load(val_str)
        except Exception:
            val = val_str
        node[keys[-1]] = val
    return cfg


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(cfg):
    req = cfg.get('device', 'auto')
    if req == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(req)


# ---------------------------------------------------------------------------
# Model factory from config
# ---------------------------------------------------------------------------

# Project-level task name (cfg['task']) → upstream INCODE task string used
# inside benchmark/methods/incode.py to switch feature_extractor / LayerNorm.
# All entries match what the upstream notebooks pass as MLP_configs['task'].
_INCODE_TASK_MAP = {
    'image_fitting':           'image',
    'image_super_resolution':  'image',
    'image_denoising':         'denoising',
    'image_inpainting':        'inpainting',
    'image_ct_reconstruction': 'image',
    'sdf':                     'shape',
    'shape_occupancy':         'shape',
}


def build_model(cfg):
    """
    Build an INR by reading cfg['method'] and cfg['model'].  Model hyper-
    parameters beyond the four standard ones are forwarded as **kwargs to
    the method's INR.__init__ (e.g. first_omega_0, scale, tau, ...).

    For INCODE specifically, the project task name (cfg['task']) is mapped
    to the upstream task string and injected into model kwargs so the
    Harmonizer picks the right feature_extractor / LayerNorm path. Callers
    can override by setting `model.task` explicitly in the YAML.
    """
    # Lazy import so `import benchmark._runner_common` has no heavy deps.
    from benchmark.methods.models import get_INR

    method = cfg['method']
    mcfg = dict(cfg.get('model', {}))
    standard = ('in_features', 'hidden_features', 'hidden_layers', 'out_features')

    if method == 'incode' and 'task' not in mcfg:
        proj_task = cfg.get('task')
        if proj_task in _INCODE_TASK_MAP:
            mcfg['task'] = _INCODE_TASK_MAP[proj_task]

    return get_INR(
        method=method,
        in_features=mcfg.get('in_features', 2),
        hidden_features=mcfg.get('hidden_features', 256),
        hidden_layers=mcfg.get('hidden_layers', 3),
        out_features=mcfg.get('out_features', 3),
        **{k: v for k, v in mcfg.items() if k not in standard},
    )


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def make_save_dir(cfg, task_name):
    out_cfg = cfg.get('output', {})
    save_dir = out_cfg.get(
        'save_dir',
        os.path.join('results', task_name, cfg['method']))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, out_cfg
