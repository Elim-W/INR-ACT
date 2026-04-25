import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Nerf-style).

    Maps input x → [x, sin(2^0 x), cos(2^0 x), ..., sin(2^(L-1) x), cos(2^(L-1) x)]
    out_dim = in_features * (2 * N_freqs + 1)
    """

    def __init__(self, in_features, N_freqs=10, logscale=True):
        super().__init__()
        self.in_features = in_features
        self.N_freqs = N_freqs
        self.out_dim = in_features * (2 * N_freqs + 1)

        if logscale:
            freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class INR(nn.Module):
    """
    ReLU MLP baseline. Set use_pe=True for PE-MLP (positional encoding + ReLU).
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 use_pe=False, N_freqs=10, **kwargs):
        super().__init__()

        if use_pe:
            self.pe = PositionalEncoding(in_features, N_freqs)
            first_in = self.pe.out_dim
        else:
            self.pe = None
            first_in = in_features

        layers = []
        layers.append(nn.Linear(first_in, hidden_features))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        x = self.pe(coords) if self.pe is not None else coords
        return self.net(x)
