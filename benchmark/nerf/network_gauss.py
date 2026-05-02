"""
Gaussian-activation NeRF — same architecture as network_finer.py,
layer swapped to GaussLayer.

GaussLayer: exp(-(scale * Wx)^2)
  - All layers use default Linear init (kaiming uniform).
  - Scale defaults to 10 for NeRF (smaller than image baselines, which
    use 30; Gaussian collapses easily on 3D coordinate inputs).
  - Last layer is plain linear (no activation).
"""

from . import ensure_torch_ngp_importable
ensure_torch_ngp_importable()

import torch
import torch.nn as nn

from encoding import get_encoder
from activation import trunc_exp
from nerf.renderer import NeRFRenderer


class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 scale=10.0, is_first=False, is_last=False):
        super().__init__()
        self.scale = scale
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if self.is_last:
            return self.linear(x)
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 num_layers=4, hidden_dim=128, geo_feat_dim=128,
                 num_layers_color=4, hidden_dim_color=128,
                 bound=1,
                 gauss_scale=10.0,
                 **kwargs):
        super().__init__(bound, **kwargs)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        self.encoder, self.in_dim = get_encoder(encoding='None')
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding='None')

        sigma_net = []
        for l in range(num_layers):
            in_f = self.in_dim if l == 0 else hidden_dim
            out_f = (1 + geo_feat_dim) if l == num_layers - 1 else hidden_dim
            sigma_net.append(GaussLayer(
                in_f, out_f,
                scale=gauss_scale,
                is_first=(l == 0),
                is_last=(l == num_layers - 1),
            ))
        self.sigma_net = nn.ModuleList(sigma_net)

        color_net = []
        for l in range(num_layers_color):
            in_f = (self.in_dim_dir + geo_feat_dim) if l == 0 else hidden_dim_color
            out_f = 3 if l == num_layers_color - 1 else hidden_dim_color
            color_net.append(GaussLayer(
                in_f, out_f,
                scale=gauss_scale,
                is_first=(l == 0),
                is_last=(l == num_layers_color - 1),
            ))
        self.color_net = nn.ModuleList(color_net)

        self.bg_net = None

    def forward(self, x, d):
        h = self.encoder(x, bound=self.bound)
        for layer in self.sigma_net:
            h = layer(h)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for layer in self.color_net:
            h = layer(h)
        return sigma, torch.sigmoid(h)

    def density(self, x):
        h = self.encoder(x, bound=self.bound)
        for layer in self.sigma_net:
            h = layer(h)
        return {'sigma': trunc_exp(h[..., 0]), 'geo_feat': h[..., 1:]}

    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)
            if not mask.any():
                return rgbs
            x = x[mask]; d = d[mask]; geo_feat = geo_feat[mask]
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for layer in self.color_net:
            h = layer(h)
        h = torch.sigmoid(h)
        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
            return rgbs
        return h

    def background(self, x, d):
        return torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)

    def get_params(self, lr):
        return [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
