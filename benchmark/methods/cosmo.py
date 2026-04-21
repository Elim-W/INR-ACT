"""
COSMO-INR: Complex Sinusoidal Modulated Raised-Cosine INR
Pandula et al., arXiv 2505.11640

Core activation: raised cosine impulse response with complex modulation
    f(x) = (1/T) * sinc(x/T) * cos(π*β*x/T) / (1-(2β*x/T)²) * exp(2π*c*x*j)

T and c are predicted per-layer from GT image by a Harmonizer (ResNet + MLP).
"""

import numpy as np
import torch
from torch import nn
import torchvision.models as tv_models


class Harmonizer(nn.Module):
    """Truncated ResNet + MLP → per-layer (T, c) modulation parameters."""

    def __init__(self, num_layers, backbone='resnet18', truncate_at=6,
                 feat_channels=128, mlp_bias=0.1,
                 T_range=(0.5, 5.0), c_range=(0.0, 3.0)):
        super().__init__()
        self.T_range = T_range
        self.c_range = c_range
        self.num_layers = num_layers  # number of RC layers (first + hidden)

        base = getattr(tv_models, backbone)(weights=None)
        self.feature_extractor = nn.Sequential(*list(base.children())[:truncate_at])
        self.gap = nn.AdaptiveAvgPool2d(1)

        out_dim = num_layers * 2  # T and c for each layer
        self.mlp = nn.Sequential(
            nn.Linear(feat_channels, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, out_dim),
        )
        self._init_mlp(mlp_bias)

    def _init_mlp(self, bias_val):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, bias_val)

    def forward(self, img):
        """img: (1, 3, H, W). Returns T_list, c_list each of length num_layers."""
        feats = self.feature_extractor(img)
        pooled = self.gap(feats).flatten(1)       # (1, feat_channels)
        coef = self.mlp(pooled)[0]                # (num_layers * 2,)
        coef = coef.view(self.num_layers, 2)       # (num_layers, 2)

        T = torch.sigmoid(coef[:, 0]) * (self.T_range[1] - self.T_range[0]) + self.T_range[0]
        c = torch.sigmoid(coef[:, 1]) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        return T, c


class RaisedCosineLayer(nn.Module):
    """
    Raised cosine impulse response layer with complex modulation.

    For the first layer: real input → complex output
    For subsequent layers: complex input (normalized to unit circle) → complex output
    Final layer handled separately (linear, real output).
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False,
                 beta0=0.05, eps=1e-8):
        super().__init__()
        self.beta0 = beta0
        self.eps = eps
        self.is_first = is_first

        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        nn.init.uniform_(self.linear.weight, -1 / in_features, 1 / in_features)

    def forward(self, x, T, c):
        """
        x: input tensor
        T: bandwidth scalar (from Harmonizer)
        c: frequency shift scalar (from Harmonizer)
        """
        z = self.linear(x)

        if not self.is_first:
            z = z / (torch.abs(z) + self.eps)  # normalize to unit circle

        f1 = (1.0 / T) * torch.sinc(z / T) * torch.cos(torch.pi * self.beta0 * z / T)
        f2 = 1.0 - (2.0 * self.beta0 * z / T) ** 2 + self.eps
        theta = 2.0 * torch.pi * c * z * 1j

        out = (f1 / f2) * torch.exp(theta)

        if not self.is_first:
            out = out / (torch.abs(out) + self.eps)

        return out


class INR(nn.Module):
    """
    COSMO-INR (COSMO-RC variant).

    Call set_gt(img_tensor) before training — the Harmonizer extracts
    per-layer (T, c) modulation params from the GT image each forward pass.

    Args:
        beta0:       roll-off factor for raised cosine (default 0.05)
        T_range:     (min, max) range for T after sigmoid projection
        c_range:     (min, max) range for c after sigmoid projection
        backbone:    torchvision backbone for Harmonizer feature extractor
        truncate_at: layer index to truncate backbone
        feat_channels: output channels of truncated backbone
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 beta0=0.05, T_range=(0.5, 5.0), c_range=(0.0, 3.0),
                 backbone='resnet18', truncate_at=6, feat_channels=128,
                 **kwargs):
        super().__init__()

        self.hidden_layers = hidden_layers
        num_rc_layers = hidden_layers + 1  # first layer + hidden layers

        self.harmonizer = Harmonizer(
            num_layers=num_rc_layers,
            backbone=backbone,
            truncate_at=truncate_at,
            feat_channels=feat_channels,
            T_range=T_range,
            c_range=c_range,
        )

        self.rc_layers = nn.ModuleList()
        self.rc_layers.append(
            RaisedCosineLayer(in_features, hidden_features, is_first=True, beta0=beta0)
        )
        for _ in range(hidden_layers):
            self.rc_layers.append(
                RaisedCosineLayer(hidden_features, hidden_features, is_first=False, beta0=beta0)
            )

        # Final linear layer (real output)
        self.final_linear = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)

        self._gt = None

    def set_gt(self, img_tensor):
        """img_tensor: (1, C, H, W) float tensor."""
        self._gt = img_tensor

    def forward(self, coords):
        assert self._gt is not None, "Call set_gt(img) before forward()."

        T_list, c_list = self.harmonizer(self._gt)

        x = coords
        for i, layer in enumerate(self.rc_layers):
            x = layer(x, T_list[i], c_list[i])

        return self.final_linear(x).real
