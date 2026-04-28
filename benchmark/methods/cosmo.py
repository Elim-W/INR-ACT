"""
COSMO-INR: Complex Sinusoidal Modulated Raised-Cosine INR
Pandula et al., arXiv 2505.11640

Core activation: raised cosine impulse response with complex modulation
    f(x) = (1/T) * sinc(x/T) * cos(π*β*x/T) / (1-(2β*x/T)²) * exp(2π*c*x*j)

T and c are predicted per-layer from GT by a Harmonizer:
  image-based tasks:  ResNet-34   (first 5 layers) + AdaptiveAvgPool2d
  3D occupancy task:  ResNet3D-18 (first 5 layers) + AdaptiveAvgPool3d
"""

import numpy as np
import torch
from torch import nn
import torchvision.models as tv_models
import torchvision.models.video as tv_video


class Harmonizer(nn.Module):
    """Truncated ResNet (2D or 3D) + MLP → per-layer (T, c) modulation parameters."""

    def __init__(self, num_layers, is_3d=False, truncate_at=5,
                 mlp_bias=0.1, T_range=(0.5, 5.0), c_range=(0.0, 3.0)):
        super().__init__()
        self.T_range = T_range
        self.c_range = c_range
        self.num_layers = num_layers
        self.is_3d = is_3d

        if is_3d:
            base = tv_video.r3d_18(weights=None)
            feat_channels = 512   # output channels of layer4 in r3d_18
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            base = tv_models.resnet34(weights=None)
            feat_channels = 64    # output channels of layer1 in resnet34
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.feature_extractor = nn.Sequential(*list(base.children())[:truncate_at])

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
        """
        img: (1, 3, H, W) for 2D  or  (1, 3, D, H, W) for 3D.
        Returns T_list, c_list each of length num_layers.
        """
        feats = self.feature_extractor(img)
        if self.is_3d:
            pooled = self.gap(feats)[:, :, 0, 0, 0]   # (1, feat_channels)
        else:
            pooled = self.gap(feats).flatten(1)         # (1, feat_channels)
        coef = self.mlp(pooled)[0]                      # (num_layers * 2,)
        coef = coef.view(self.num_layers, 2)

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

    Backbone selection follows the original paper:
      in_features == 2  →  ResNet-34   (image tasks, first 5 layers)
      in_features == 3  →  ResNet3D-18 (3D occupancy, first 5 layers)

    Call set_gt(gt_tensor) before training/inference:
      2D: gt_tensor shape (1, 3, H, W)
      3D: gt_tensor shape (1, 3, D, H, W)
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 beta0=0.05, T_range=(0.5, 5.0), c_range=(0.0, 3.0),
                 **kwargs):
        super().__init__()

        self.hidden_layers = hidden_layers
        is_3d = (in_features == 3)
        num_rc_layers = hidden_layers + 1  # first layer + hidden layers

        self.harmonizer = Harmonizer(
            num_layers=num_rc_layers,
            is_3d=is_3d,
            truncate_at=5,
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
        """
        2D: img_tensor (1, 3, H, W)
        3D: img_tensor (1, 3, D, H, W)
        """
        self._gt = img_tensor

    def forward(self, coords):
        assert self._gt is not None, "Call set_gt(img) before forward()."

        T_list, c_list = self.harmonizer(self._gt)

        x = coords
        for i, layer in enumerate(self.rc_layers):
            x = layer(x, T_list[i], c_list[i])

        return self.final_linear(x).real
