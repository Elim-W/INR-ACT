import numpy as np
import torch
from torch import nn


class ComplexGaborLayer(nn.Module):
    """
    Complex Gabor wavelet activation layer (WIRE).

    First layer uses real (float) input; subsequent layers operate in complex (cfloat).
    Output: exp(j*omega_0*lin - (sigma_0*lin).abs()^2)
    This is a complex Gabor: a sinusoid modulated by a Gaussian envelope.

    omega_0 and sigma_0 are learnable scalars (frozen by default).
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0, trainable=False):
        super().__init__()
        self.is_first = is_first
        self.in_features = in_features

        dtype = torch.float if is_first else torch.cfloat

        self.omega_0 = nn.Parameter(torch.tensor(omega0), requires_grad=trainable)
        self.scale_0 = nn.Parameter(torch.tensor(sigma0), requires_grad=trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x):
        lin = self.linear(x)
        return torch.exp(1j * self.omega_0 * lin - (self.scale_0 * lin).abs().square())


class INR(nn.Module):
    """
    WIRE: Wavelet Implicit neural REpresentations.
    Saragadam et al., CVPR 2023.

    Uses complex Gabor wavelets. Hidden dim is reduced by sqrt(2) so parameter
    count matches a real-valued network with the same nominal width.
    Output is the real part of the final complex linear layer.
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=10.0, hidden_omega_0=10.0, scale=10.0,
                 **kwargs):
        super().__init__()
        self.wavelet = 'gabor'

        # Complex numbers encode two reals → halve width to keep param parity
        hidden_features = int(hidden_features / np.sqrt(2))

        layers = []
        layers.append(ComplexGaborLayer(in_features, hidden_features,
                                        is_first=True,
                                        omega0=first_omega_0, sigma0=scale,
                                        trainable=False))

        for _ in range(hidden_layers):
            layers.append(ComplexGaborLayer(hidden_features, hidden_features,
                                            is_first=False,
                                            omega0=hidden_omega_0, sigma0=scale,
                                            trainable=False))

        final = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        out = self.net(coords)
        return out.real
