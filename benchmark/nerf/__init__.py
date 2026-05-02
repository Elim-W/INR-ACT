"""
Standalone NeRF training stack for activation comparison.

Layout follows FINER's nerf/ folder: one network_<method>.py per activation,
one main_nerf.py CLI entry that picks the right network. Imports torch-ngp's
NeRFRenderer / NeRFDataset / Trainer / get_encoder / trunc_exp directly via
sys.path injection — we do not vendor those files.
"""

import os
import sys


def ensure_torch_ngp_importable():
    """
    Prepend third_party/torch-ngp/ so `from nerf.provider import NeRFDataset`,
    `from encoding import get_encoder`, etc. resolve to torch-ngp's source.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ngp_root = os.path.join(root, 'third_party', 'torch-ngp')
    if ngp_root not in sys.path:
        sys.path.insert(0, ngp_root)


ensure_torch_ngp_importable()
