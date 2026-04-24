"""
Parameterised NeRFNetwork that subclasses torch-ngp's NeRFRenderer.

Same architecture as FINER's network_finer.py (no positional encoding,
stacked activation layers for sigma and color heads), but the activation
type is chosen at construction time rather than hard-coded.

Usage:
    from benchmark.methods.nerf_networks import build_nerf_network
    model = build_nerf_network('siren', num_layers=4, hidden_dim=256,
                               geo_feat_dim=256, num_layers_color=4,
                               hidden_dim_color=256, bound=1, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# These imports resolve because benchmark/methods/nerf_networks/__init__.py
# prepended third_party/torch-ngp to sys.path.
from nerf.renderer import NeRFRenderer  # type: ignore
from encoding import get_encoder        # type: ignore
from activation import trunc_exp        # type: ignore

from .activation_layers import build_mlp_stack, available_activations


class NeRFNetwork(NeRFRenderer):
    """
    NeRF = (density MLP + color MLP) with a custom activation function.

    Input: x in [-bound, bound]^3, d (view direction) normalised to [-1, 1].
    Output: (sigma, rgb).  Volume rendering is done by NeRFRenderer.
    """

    def __init__(self,
                 activation='siren',
                 num_layers=4,
                 hidden_dim=256,
                 geo_feat_dim=256,
                 num_layers_color=4,
                 hidden_dim_color=256,
                 bound=1.0,
                 # activation-specific knobs (all optional)
                 first_omega_0=30.0,
                 hidden_omega_0=30.0,
                 first_bias_scale=None,
                 scale_req_grad=False,
                 scale=10.0,
                 **kwargs,  # cuda_ray, min_near, density_thresh, bg_radius, ...
                 ):
        super().__init__(bound, **kwargs)

        if activation not in available_activations():
            raise ValueError(
                f"activation '{activation}' not supported for NeRF. "
                f"Available: {available_activations()}")
        self.activation = activation
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        # No positional encoding — raw (x, y, z) and (dx, dy, dz) go in
        self.encoder, self.in_dim = get_encoder(encoding='None')
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding='None')

        # Density head: in_dim → hidden → ... → (1 + geo_feat_dim)
        sigma_dims = ([self.in_dim]
                      + [hidden_dim] * (num_layers - 1)
                      + [1 + geo_feat_dim])
        self.sigma_net = build_mlp_stack(
            activation, sigma_dims,
            first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0,
            last_linear=True,
            first_bias_scale=first_bias_scale,
            scale_req_grad=scale_req_grad,
            scale=scale,
        )

        # Color head: (in_dim_dir + geo_feat_dim) → hidden_color → ... → 3
        color_dims = ([self.in_dim_dir + geo_feat_dim]
                      + [hidden_dim_color] * (num_layers_color - 1)
                      + [3])
        self.color_net = build_mlp_stack(
            activation, color_dims,
            first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0,
            last_linear=True,
            first_bias_scale=first_bias_scale,
            scale_req_grad=scale_req_grad,
            scale=scale,
        )

        # No background model (keep it simple; add later if needed)
        self.bg_net = None

    # -----------------------------------------------------------------
    # NeRFRenderer API: forward / density / color / background / get_params
    # -----------------------------------------------------------------

    def _run_sigma(self, x):
        h = self.encoder(x, bound=self.bound)
        for layer in self.sigma_net:
            h = layer(h)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        return sigma, geo_feat

    def _run_color(self, d, geo_feat):
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for layer in self.color_net:
            h = layer(h)
        return torch.sigmoid(h)

    def forward(self, x, d):
        sigma, geo_feat = self._run_sigma(x)
        color = self._run_color(d, geo_feat)
        return sigma, color

    def density(self, x):
        sigma, geo_feat = self._run_sigma(x)
        return {'sigma': sigma, 'geo_feat': geo_feat}

    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]
        rgbs_active = self._run_color(d, geo_feat)
        if mask is not None:
            rgbs[mask] = rgbs_active.to(rgbs.dtype)
            return rgbs
        return rgbs_active

    def background(self, x, d):
        # No background model — return black.
        return torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)

    # -----------------------------------------------------------------
    # Optimizer parameter groups (NeRFRenderer expects this)
    # -----------------------------------------------------------------

    def get_params(self, lr):
        params = [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
        return params
