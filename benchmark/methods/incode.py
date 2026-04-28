import numpy as np
import torch
from torch import nn
import torchvision.models as tv_models
import torchvision.models.video as tv_video


# ---------------------------------------------------------------------------
# Harmonizer (GT → 4 modulation scalars)
# ---------------------------------------------------------------------------

class Harmonizer(nn.Module):
    """
    image-based tasks:  truncated ResNet-34   (first 5 layers) + AdaptiveAvgPool1d
    3D occupancy task:  truncated ResNet3D-18 (first 5 layers) + AdaptiveAvgPool3d

    Follows the original INCODE third-party implementation.
    """

    def __init__(self, is_3d=False, truncate_at=5,
                 mlp_hidden_channels=(64, 32, 4), mlp_bias=0.1):
        super().__init__()
        self.is_3d = is_3d

        if is_3d:
            base = tv_video.r3d_18(weights=None)
            feat_channels = 512   # layer4 output of r3d_18
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            base = tv_models.resnet34(weights=None)
            feat_channels = 64    # layer1 output of resnet34
            self.gap = nn.AdaptiveAvgPool1d(1)

        self.feature_extractor = nn.Sequential(*list(base.children())[:truncate_at])

        layers = []
        in_dim = feat_channels
        for h in mlp_hidden_channels[:-1]:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, mlp_hidden_channels[-1]))
        self.mlp = nn.Sequential(*layers)
        self._init_mlp(mlp_bias)

    def _init_mlp(self, bias_val):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, bias_val)

    def forward(self, img):
        feats = self.feature_extractor(img)
        if self.is_3d:
            # (1, C, D', H', W') → AdaptiveAvgPool3d → (1, C, 1, 1, 1) → (1, C)
            pooled = self.gap(feats)[:, :, 0, 0, 0]
        else:
            # (1, C, H', W') → flatten spatial → AdaptiveAvgPool1d → (1, C)
            pooled = self.gap(
                feats.view(feats.size(0), feats.size(1), -1)
            )[..., 0]
        coef = self.mlp(pooled)   # (1, 4)
        return coef[0]            # (4,)


# ---------------------------------------------------------------------------
# Composer SineLayer
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    """
    SIREN-style layer modulated by (a, b, c, d):
        output = exp(a) * sin(exp(b) * omega_0 * Wx + c) + d
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

    def forward(self, x, a, b, c, d):
        z = self.linear(x)
        return torch.exp(a) * torch.sin(torch.exp(b) * self.omega_0 * z + c) + d


# ---------------------------------------------------------------------------
# INR
# ---------------------------------------------------------------------------

class INR(nn.Module):
    """
    INCODE: Implicit Neural Conditioning with Prior Knowledge Encodings.
    Kazerouni et al., WACV 2024.

    Backbone selection follows the original paper / third-party code:
      in_features == 2  →  ResNet-34   (image tasks, first 5 layers)
      in_features == 3  →  ResNet3D-18 (3D occupancy, first 5 layers)

    Call set_gt(gt_tensor) before training/inference:
      2D: gt_tensor shape (1, 3, H, W)
      3D: gt_tensor shape (1, 3, D, H, W)
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30.0, hidden_omega_0=30.0,
                 mlp_hidden_channels=(64, 32, 4),
                 mlp_bias=0.1,
                 **kwargs):
        super().__init__()

        is_3d = (in_features == 3)
        self.hidden_layers = hidden_layers

        self.harmonizer = Harmonizer(
            is_3d=is_3d,
            truncate_at=5,
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_bias=mlp_bias,
        )

        self.composer = nn.ModuleList()
        self.composer.append(SineLayer(in_features, hidden_features,
                                       is_first=True, omega_0=first_omega_0))
        for _ in range(hidden_layers):
            self.composer.append(SineLayer(hidden_features, hidden_features,
                                           is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / max(hidden_omega_0, 1e-12)
                final.weight.uniform_(-bound, bound)
            self.composer.append(final)
        else:
            self.composer.append(SineLayer(hidden_features, out_features,
                                           is_first=False, omega_0=hidden_omega_0))

        self.outermost_linear = outermost_linear
        self._gt = None

    def set_gt(self, gt_tensor):
        """
        2D: gt_tensor (1, 3, H, W)
        3D: gt_tensor (1, 3, D, H, W)
        """
        self._gt = gt_tensor

    def forward(self, coords):
        assert self._gt is not None, "Call set_gt(gt) before forward()."

        a, b, c, d = self.harmonizer(self._gt)

        x = coords
        for i in range(self.hidden_layers + 1):
            x = self.composer[i](x, a, b, c, d)

        if self.outermost_linear:
            x = self.composer[self.hidden_layers + 1](x)
        else:
            x = self.composer[self.hidden_layers + 1](x, a, b, c, d)

        return x
