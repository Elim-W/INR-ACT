import numpy as np
import torch
from torch import nn


class ComplexGaborLayer2D(nn.Module):
    """
    2D complex Gabor activation layer.

    Extends WIRE by using two orthogonal Gaussian windows (scale_x, scale_y)
    to capture 2D spatial structure more directly.
    Gaussian: exp(-sigma^2 * (|lin_x|^2 + |lin_orth|^2))
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0, trainable=False):
        super().__init__()
        self.is_first = is_first
        self.in_features = in_features

        dtype = torch.float if is_first else torch.cfloat

        self.omega_0 = nn.Parameter(torch.tensor(omega0), requires_grad=trainable)
        self.scale_0 = nn.Parameter(torch.tensor(sigma0), requires_grad=trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.scale_orth = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x):
        lin = self.linear(x)
        scale_y = self.scale_orth(x)

        freq_term = torch.exp(1j * self.omega_0 * lin)
        gauss_term = torch.exp(-self.scale_0 ** 2 * (lin.abs().square() + scale_y.abs().square()))
        return freq_term * gauss_term


class INR(nn.Module):
    """
    WIRE-2D: 2D variant of WIRE using two orthogonal Gaussian windows.

    Hidden dim is halved (not sqrt(2)) because the 2D layer has two linear
    projections per neuron.
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=10.0, hidden_omega_0=10.0, scale=10.0,
                 **kwargs):
        super().__init__()
        self.wavelet = 'gabor'

        hidden_features = hidden_features // 2

        layers = []
        layers.append(ComplexGaborLayer2D(in_features, hidden_features,
                                          is_first=True,
                                          omega0=first_omega_0, sigma0=scale,
                                          trainable=False))

        for _ in range(hidden_layers):
            layers.append(ComplexGaborLayer2D(hidden_features, hidden_features,
                                              is_first=False,
                                              omega0=hidden_omega_0, sigma0=scale,
                                              trainable=False))

        final = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        out = self.net(coords)
        return out.real
