"""
Activation layers re-used for NeRF.

Each class mirrors the single-layer building block from
benchmark/methods/<method>.py, with one small adaptation: NeRF networks
need a "last" linear layer (no activation) after the sigma head and
before color output.  The `build_*_stack` helpers below assemble the
right sequence for a given activation.

Keeping this file in one place means that if we improve an activation
(e.g. new init), both the image-task INR and the NeRF NeRFNetwork pick
it up automatically.
"""

import math
import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# SIREN
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = np.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# ---------------------------------------------------------------------------
# FINER
# ---------------------------------------------------------------------------

class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0,
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.scale_req_grad = scale_req_grad
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
                if first_bias_scale is not None:
                    self.linear.bias.uniform_(-first_bias_scale, first_bias_scale)
            else:
                bound = np.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        z = self.linear(x)
        if self.scale_req_grad:
            scale = torch.abs(z) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(z) + 1
        return torch.sin(self.omega_0 * scale * z)


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 scale=10.0, is_first=False):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


# ---------------------------------------------------------------------------
# WIRE (real-valued variant: sin * gauss envelope instead of complex Gabor,
# because NeRF downstream code expects real tensors)
# ---------------------------------------------------------------------------

class WireLayer(nn.Module):
    """
    Real-valued Gabor wavelet: exp(-(scale * Wx)**2) * cos(omega_0 * Wx)
    Equivalent to the real part of WIRE's complex Gabor, so downstream
    code does not have to handle complex tensors.
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=20.0, scale=10.0):
        super().__init__()
        self.omega_0 = omega_0
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        z = self.linear(x)
        return torch.exp(-(self.scale * z) ** 2) * torch.cos(self.omega_0 * z)


# ---------------------------------------------------------------------------
# ReLU + positional encoding baseline — the sigma_net has a simple
# `Linear + ReLU` stack; positional encoding is applied outside.
# ---------------------------------------------------------------------------

class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return torch.relu(self.linear(x))


# ---------------------------------------------------------------------------
# Identity passthrough — used for the final linear layer in each head
# (no activation).
# ---------------------------------------------------------------------------

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Stack builder
# ---------------------------------------------------------------------------

_ACTIVATION_FACTORY = {
    'siren':  lambda in_f, out_f, **kw: SineLayer(
        in_f, out_f, is_first=kw.get('is_first', False),
        omega_0=kw.get('omega_0', 30.0)),
    'finer':  lambda in_f, out_f, **kw: FinerLayer(
        in_f, out_f, is_first=kw.get('is_first', False),
        omega_0=kw.get('omega_0', 30.0),
        first_bias_scale=kw.get('first_bias_scale', None),
        scale_req_grad=kw.get('scale_req_grad', False)),
    'gauss':  lambda in_f, out_f, **kw: GaussLayer(
        in_f, out_f, scale=kw.get('scale', 10.0),
        is_first=kw.get('is_first', False)),
    'wire':   lambda in_f, out_f, **kw: WireLayer(
        in_f, out_f, is_first=kw.get('is_first', False),
        omega_0=kw.get('omega_0', 20.0), scale=kw.get('scale', 10.0)),
    'relu':   lambda in_f, out_f, **kw: ReLULayer(
        in_f, out_f, is_first=kw.get('is_first', False)),
}


def available_activations():
    return sorted(_ACTIVATION_FACTORY.keys())


def make_layer(activation, in_f, out_f, **kwargs):
    if activation not in _ACTIVATION_FACTORY:
        raise ValueError(
            f"Unknown activation '{activation}'. "
            f"Available: {available_activations()}")
    return _ACTIVATION_FACTORY[activation](in_f, out_f, **kwargs)


def build_mlp_stack(activation, dims, first_omega_0=30.0, hidden_omega_0=30.0,
                    last_linear=True, **act_kwargs):
    """
    Build a Sequential MLP using the named activation.

    `dims` is a list like [in, h1, h2, ..., out].  All intermediate layers
    use the activation; if `last_linear` is True, the output layer is a
    plain `nn.Linear` (no activation) — standard for NeRF heads.
    """
    assert len(dims) >= 2, "dims must contain at least [in, out]"
    layers = nn.ModuleList()
    n = len(dims) - 1
    for i in range(n):
        in_f, out_f = dims[i], dims[i + 1]
        is_last = (i == n - 1)
        if is_last and last_linear:
            layers.append(LinearLayer(in_f, out_f))
        else:
            omega = first_omega_0 if i == 0 else hidden_omega_0
            layers.append(make_layer(
                activation, in_f, out_f,
                is_first=(i == 0),
                omega_0=omega,
                **act_kwargs,
            ))
    return layers
