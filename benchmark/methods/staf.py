import math
import torch
from torch import nn


class StafLayer(nn.Module):
    """
    STAF layer: sum of tau sinusoids with learnable frequencies, phases, amplitudes.

    output = sum_{i=1}^{tau}  b_i * sin(w_i * Wx + phi_i)

    ws  ~ omega_0 * Uniform(0,1)       (learnable)
    phis ~ Uniform(-pi, pi)             (learnable)
    bs   ~ signed-sqrt of Laplace(0, 1/(2*tau))  (learnable, Laplace diversity init)
    Optional skip connection for hidden layers (acts as a residual path).
    """

    def __init__(self, in_features, out_features, tau=5,
                 bias=True, is_first=False, omega_0=30.0,
                 scale=10.0, skip_conn=False):
        super().__init__()
        self.tau = tau
        self.omega_0 = omega_0
        self.is_first = is_first
        self.skip_conn = skip_conn
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_params()

    def _init_params(self):
        ws = self.omega_0 * torch.rand(self.tau)
        self.ws = nn.Parameter(ws)

        phis = -math.pi + 2 * math.pi * torch.rand(self.tau)
        self.phis = nn.Parameter(phis)

        diversity = 1.0 / (2 * self.tau)
        samples = torch.distributions.Laplace(0, diversity).sample((self.tau,))
        self.bs = nn.Parameter(torch.sign(samples) * torch.sqrt(torch.abs(samples)))

    def forward(self, x):
        lin = self.linear(x)
        # lin: (..., out), ws/phis/bs: (tau,)
        # unsqueeze for broadcasting → (..., out, tau)
        act = (self.bs * torch.sin(self.ws * lin.unsqueeze(-1) + self.phis)).sum(-1)
        if not self.is_first and self.skip_conn:
            return act + lin
        return act


class INR(nn.Module):
    """
    STAF: Sinusoidal Trainable Activation Functions for INRs.

    Architecture: StafLayer (first) → [StafLayer] × hidden_layers → Linear
    skip_conn enables residual connections in hidden layers.
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30.0, hidden_omega_0=30.0,
                 scale=10.0, tau=5, skip_conn=False,
                 **kwargs):
        super().__init__()

        layers = []
        layers.append(StafLayer(in_features, hidden_features,
                                tau=tau, is_first=True,
                                omega_0=first_omega_0, scale=scale))

        for _ in range(hidden_layers):
            layers.append(StafLayer(hidden_features, hidden_features,
                                    tau=tau, is_first=False,
                                    omega_0=hidden_omega_0, scale=scale,
                                    skip_conn=skip_conn))

        if outermost_linear:
            layers.append(nn.Linear(hidden_features, out_features))
        else:
            layers.append(StafLayer(hidden_features, out_features,
                                    tau=tau, is_first=False,
                                    omega_0=hidden_omega_0, skip_conn=skip_conn))

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)
