import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """
    One layer of SIREN: linear + sin(omega_0 * x).

    If is_first=True, weights ~ U(-1/n, 1/n).
    Otherwise weights ~ U(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0).
    This ensures activations stay unit-variance throughout the network.
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class INR(nn.Module):
    """
    SIREN: Implicit Neural Representations with Periodic Activation Functions.
    Sitzmann et al., NeurIPS 2020.

    Architecture: SineLayer (first) → [SineLayer] × hidden_layers → Linear
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30.0, hidden_omega_0=30.0,
                 **kwargs):
        super().__init__()

        layers = []
        layers.append(SineLayer(in_features, hidden_features,
                                is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / max(hidden_omega_0, 1e-12)
                final.weight.uniform_(-bound, bound)
            layers.append(final)
        else:
            layers.append(SineLayer(hidden_features, out_features,
                                    is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)
