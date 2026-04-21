"""
FINER++ family  (Liu et al., NeurIPS 2023 + FINER++ follow-up).

Three variants all share the FINER adaptive-frequency core:
  alpha(z) = |z| + 1   (no gradient)
  finer(z) = sin(omega * alpha(z) * z)

Variant       Outer activation          Class
-----------   -----------------------   -------
Finer++Sine   sin(omega*(|z|+1)*z)      FinerPPSine   ← replaces finer.py
Finer++Gauss  exp(-(s*finer(z))^2)      FinerPPGauss  ← replaces gf.py
Finer++Wave   Wire(finer(z))            FinerPPWave   ← replaces wf.py

Use via models.py:
    get_INR('finer', ...)   → FinerPPSine
    get_INR('gf', ...)      → FinerPPGauss
    get_INR('wf', ...)      → FinerPPWave
"""

import numpy as np
import torch
from torch import nn


# Shared activation primitives
def _finer_real(z, omega):
    """FINER on real tensor: sin(omega * (|z|+1) * z). No grad through alpha."""
    with torch.no_grad():
        alpha = torch.abs(z) + 1
    return torch.sin(omega * alpha * z)


def _finer_complex(z, omega):
    """FINER applied separately to real/imag parts of a complex tensor."""
    with torch.no_grad():
        alpha_r = torch.abs(z.real) + 1
        alpha_i = torch.abs(z.imag) + 1
    return torch.sin(omega * torch.complex(z.real * alpha_r, z.imag * alpha_i))


def _wire(x, scale, omega_w):
    """Complex Gabor: exp(j*omega_w*x - |scale*x|^2)."""
    return torch.exp(1j * omega_w * x - torch.abs(scale * x) ** 2)



# Weight initialisation helpers
def _init_siren(linear, omega, is_first):
    with torch.no_grad():
        n = linear.weight.shape[1]
        if is_first:
            linear.weight.uniform_(-1 / n, 1 / n)
        else:
            bound = np.sqrt(6 / n) / max(omega, 1e-12)
            linear.weight.uniform_(-bound, bound)



# Finer++Sine
class _FinerSineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, is_last=False,
                 omega=30.0, first_bias_scale=None):
        super().__init__()
        self.omega = omega
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        _init_siren(self.linear, omega, is_first)
        if is_first and first_bias_scale is not None:
            with torch.no_grad():
                self.linear.bias.uniform_(-first_bias_scale, first_bias_scale)

    def forward(self, x):
        z = self.linear(x)
        return z if self.is_last else _finer_real(z, self.omega)


class FinerPPSine(nn.Module):
    """
    Finer++Sine: sin(omega * (|z|+1) * z)

    Defaults: first_omega=30, hidden_omega=30, lr=1e-4
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 first_omega_0=30.0, hidden_omega_0=30.0,
                 first_bias_scale=None, **kwargs):
        super().__init__()
        layers = [_FinerSineLayer(in_features, hidden_features,
                                  is_first=True, omega=first_omega_0,
                                  first_bias_scale=first_bias_scale)]
        for _ in range(hidden_layers):
            layers.append(_FinerSineLayer(hidden_features, hidden_features,
                                          omega=hidden_omega_0))
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / max(hidden_omega_0, 1e-12)
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)


# Finer++Gauss
class _GFLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, is_last=False,
                 scale=3.0, omega=10.0, first_bias_scale=None):
        super().__init__()
        self.scale = scale
        self.omega = omega
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        _init_siren(self.linear, omega, is_first)
        if is_first and first_bias_scale is not None:
            with torch.no_grad():
                self.linear.bias.uniform_(-first_bias_scale, first_bias_scale)

    def forward(self, x):
        z = self.linear(x)
        if self.is_last:
            return z
        return torch.exp(-(self.scale * _finer_real(z, self.omega)) ** 2)


class FinerPPGauss(nn.Module):
    """
    Finer++Gauss: exp(-(scale * sin(omega*(|z|+1)*z))^2)

    Defaults: scale=3, omega=10, first_bias_scale=1, lr=1e-3
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 scale=3.0, omega=10.0, first_bias_scale=1.0, **kwargs):
        super().__init__()
        layers = [_GFLayer(in_features, hidden_features,
                           is_first=True, scale=scale, omega=omega,
                           first_bias_scale=first_bias_scale)]
        for _ in range(hidden_layers):
            layers.append(_GFLayer(hidden_features, hidden_features,
                                   scale=scale, omega=omega))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)



# Finer++Wave (WF)
class _WFLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, is_last=False,
                 scale=2.0, omega_w=4.0, omega=5.0, first_bias_scale=None):
        super().__init__()
        self.scale = scale
        self.omega_w = omega_w
        self.omega = omega
        self.is_last = is_last
        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)
        _init_siren(self.linear, omega * omega_w, is_first)
        if is_first and first_bias_scale is not None:
            with torch.no_grad():
                self.linear.bias.uniform_(-first_bias_scale, first_bias_scale)

    def forward(self, x):
        z = self.linear(x)
        if self.is_last:
            return z
        finer_out = _finer_complex(z, self.omega) if z.is_complex() \
                    else _finer_real(z, self.omega)
        return _wire(finer_out, self.scale, self.omega_w)


class FinerPPWave(nn.Module):
    """
    Finer++Wave (WF): Wire(FINER(z)) — complex Gabor over adaptive sinusoid.

    Hidden layers are complex; hidden_features halved to keep param count fair.
    Defaults: scale=2, omega_w=4, omega=5, first_bias_scale=1, lr=1e-3
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 scale=2.0, omega_w=4.0, omega=5.0, first_bias_scale=1.0, **kwargs):
        super().__init__()
        h = int(hidden_features / np.sqrt(2))   # halve for complex parity
        layers = [_WFLayer(in_features, h, is_first=True,
                           scale=scale, omega_w=omega_w, omega=omega,
                           first_bias_scale=first_bias_scale)]
        for _ in range(hidden_layers):
            layers.append(_WFLayer(h, h, scale=scale, omega_w=omega_w, omega=omega))
        layers.append(_WFLayer(h, out_features, is_last=True,
                               scale=scale, omega_w=omega_w, omega=omega))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords).real
