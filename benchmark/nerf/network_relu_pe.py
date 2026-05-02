"""
ReLU + positional encoding NeRF — original NeRF (Mildenhall et al. 2020).

Same backbone shape as the other network_*.py (4-layer sigma + 4-layer color),
but with frequency-encoded inputs:
  x   →  10-frequency PE  (input_dim = 3 + 3*10*2 = 63)
  dir →   4-frequency PE  (input_dim = 3 + 3*4*2  = 27)

Hidden activation: ReLU. Last layer of each branch: plain Linear.

Note: we use the pure-PyTorch FreqEncoder defined at the top of
torch-ngp/encoding.py (not the CUDA `freqencoder` extension) to avoid
the JIT-build dependency.
"""

from . import ensure_torch_ngp_importable
ensure_torch_ngp_importable()

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import FreqEncoder
from activation import trunc_exp
from nerf.renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 num_layers=4, hidden_dim=128, geo_feat_dim=128,
                 num_layers_color=4, hidden_dim_color=128,
                 bound=1,
                 multires_x=10, multires_dir=4,
                 **kwargs):
        super().__init__(bound, **kwargs)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        self.encoder = FreqEncoder(
            input_dim=3, max_freq_log2=multires_x - 1,
            N_freqs=multires_x, log_sampling=True)
        self.in_dim = self.encoder.output_dim

        self.encoder_dir = FreqEncoder(
            input_dim=3, max_freq_log2=multires_dir - 1,
            N_freqs=multires_dir, log_sampling=True)
        self.in_dim_dir = self.encoder_dir.output_dim

        sigma_net = []
        for l in range(num_layers):
            in_f = self.in_dim if l == 0 else hidden_dim
            out_f = (1 + geo_feat_dim) if l == num_layers - 1 else hidden_dim
            sigma_net.append(nn.Linear(in_f, out_f))
        self.sigma_net = nn.ModuleList(sigma_net)

        color_net = []
        for l in range(num_layers_color):
            in_f = (self.in_dim_dir + geo_feat_dim) if l == 0 else hidden_dim_color
            out_f = 3 if l == num_layers_color - 1 else hidden_dim_color
            color_net.append(nn.Linear(in_f, out_f))
        self.color_net = nn.ModuleList(color_net)

        self.bg_net = None

    def _run_sigma(self, x):
        h = self.encoder(x)
        for i, layer in enumerate(self.sigma_net):
            h = layer(h)
            if i != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        return sigma, geo_feat

    def _run_color(self, d, geo_feat):
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for i, layer in enumerate(self.color_net):
            h = layer(h)
            if i != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        return torch.sigmoid(h)

    def forward(self, x, d):
        sigma, geo_feat = self._run_sigma(x)
        return sigma, self._run_color(d, geo_feat)

    def density(self, x):
        sigma, geo_feat = self._run_sigma(x)
        return {'sigma': sigma, 'geo_feat': geo_feat}

    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)
            if not mask.any():
                return rgbs
            x = x[mask]; d = d[mask]; geo_feat = geo_feat[mask]
        rgbs_active = self._run_color(d, geo_feat)
        if mask is not None:
            rgbs[mask] = rgbs_active.to(rgbs.dtype)
            return rgbs
        return rgbs_active

    def background(self, x, d):
        return torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)

    def get_params(self, lr):
        return [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
