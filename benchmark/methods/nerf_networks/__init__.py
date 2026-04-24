"""
NeRF-side method adapters.

Each method re-uses the activation/Layer classes we already have under
benchmark/methods/*.py, wraps them into torch-ngp's NeRFRenderer subclass,
and exposes a single factory:

    from benchmark.methods.nerf_networks import build_nerf_network
    model = build_nerf_network('siren', num_layers=4, hidden_dim=256, ...)

At import time this module tells Python where to find torch-ngp's
NeRFRenderer / get_encoder / trunc_exp, so the rest of the benchmark
code does not have to care about sys.path gymnastics.
"""

import os
import sys


def _ensure_torch_ngp_importable():
    """
    torch-ngp is a git clone (not a package), its NeRFRenderer lives in
    third_party/torch-ngp/nerf/renderer.py and imports sibling files like
    `from encoding import get_encoder`.  Prepend that directory so those
    imports resolve.
    """
    root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ngp_root = os.path.join(root, 'third_party', 'torch-ngp')
    if ngp_root not in sys.path:
        sys.path.insert(0, ngp_root)


_ensure_torch_ngp_importable()

# NOTE: `factory` / `network` are imported lazily (only when a caller
# actually needs build_nerf_network).  This keeps the package importable
# on CPU / login nodes — torch-ngp's raymarching JIT-compiles on first
# `import raymarching`, which needs nvcc + enough RAM.

def build_nerf_network(*args, **kwargs):
    from .factory import build_nerf_network as _build
    return _build(*args, **kwargs)


__all__ = ['build_nerf_network']
