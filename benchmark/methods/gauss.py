import torch
from torch import nn


class GaussLayer(nn.Module):
    """
    Gaussian activation layer: exp(-(scale * Wx + b)^2).
    Scale is a fixed hyperparameter controlling the bandwidth.
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0, scale=10.0):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


class INR(nn.Module):
    """
    Gaussian activation INR.
    Ramasinghe & Lucey, ECCV 2022 / used as baseline in WIRE.

    Architecture: GaussLayer × (1 + hidden_layers) → Linear
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30.0, hidden_omega_0=30.0, scale=10.0,
                 **kwargs):
        super().__init__()

        layers = []
        layers.append(GaussLayer(in_features, hidden_features,
                                 is_first=True, scale=scale))

        for _ in range(hidden_layers):
            layers.append(GaussLayer(hidden_features, hidden_features,
                                     is_first=False, scale=scale))

        if outermost_linear:
            layers.append(nn.Linear(hidden_features, out_features))
        else:
            layers.append(GaussLayer(hidden_features, out_features, scale=scale))

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)
