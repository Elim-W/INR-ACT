import numpy as np
import torch
from torch import nn
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Harmonizer (image → 4 modulation scalars)
# ---------------------------------------------------------------------------

class Harmonizer(nn.Module):
    """
    Extracts image-level features with a truncated ResNet and maps them to
    4 scalar modulation parameters (a, b, c, d) for the composer network.

    a, b: log-scale amplitude and frequency multipliers  → exp(a), exp(b)
    c:    phase shift
    d:    output bias
    """

    def __init__(self, backbone='resnet18', truncate_at=6,
                 feat_channels=128,
                 mlp_hidden_channels=(64, 32, 4),
                 mlp_bias=0.1):
        super().__init__()

        base = getattr(tv_models, backbone)(weights=None)
        # children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        children = list(base.children())
        self.feature_extractor = nn.Sequential(*children[:truncate_at])
        self.gap = nn.AdaptiveAvgPool2d(1)

        # small MLP: feat_channels → ... → 4
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
        """
        img: (1, C, H, W) ground-truth image tensor.
        Returns: (a, b, c, d) each scalar.
        """
        feats = self.feature_extractor(img)       # (1, C', h', w')
        pooled = self.gap(feats).flatten(1)        # (1, C')
        coef = self.mlp(pooled)                    # (1, 4)
        return coef[0]                             # (4,)


# ---------------------------------------------------------------------------
# Composer SineLayer  (accepts 4 modulation params)
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

    Two-module design:
      - Harmonizer: truncated ResNet + MLP → 4 modulation scalars from GT image
      - Composer: modulated SIREN using those 4 scalars

    The GT image must be set via set_gt(img) before training/inference.

    Args:
        gt_channels:        channels of GT image (default 3 for RGB)
        backbone:           torchvision backbone for feature extraction
        truncate_at:        truncate backbone at this child index
        feat_channels:      output channels of the truncated backbone
        mlp_hidden_channels: hidden dims of harmonizer MLP (last must be 4)
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30.0, hidden_omega_0=30.0,
                 gt_channels=3,
                 backbone='resnet18', truncate_at=6,
                 feat_channels=128,
                 mlp_hidden_channels=(64, 32, 4),
                 mlp_bias=0.1,
                 **kwargs):
        super().__init__()

        self.hidden_layers = hidden_layers

        self.harmonizer = Harmonizer(
            backbone=backbone,
            truncate_at=truncate_at,
            feat_channels=feat_channels,
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_bias=mlp_bias,
        )

        # Composer network (manually iterated — not nn.Sequential because
        # each layer needs the (a,b,c,d) params from the harmonizer)
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
        self._gt = None   # set by set_gt()

    def set_gt(self, img_tensor):
        """
        Register the ground-truth image used by the harmonizer.
        img_tensor: (1, C, H, W) float tensor in [0,1] or [-1,1].
        """
        self._gt = img_tensor

    def forward(self, coords):
        assert self._gt is not None, "Call set_gt(img) before forward()."

        a, b, c, d = self.harmonizer(self._gt)

        x = coords
        for i in range(self.hidden_layers + 1):
            x = self.composer[i](x, a, b, c, d)

        # final layer (linear or sine)
        if self.outermost_linear:
            x = self.composer[self.hidden_layers + 1](x)
        else:
            x = self.composer[self.hidden_layers + 1](x, a, b, c, d)

        return x
