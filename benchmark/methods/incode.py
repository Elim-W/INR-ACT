"""
INCODE: Implicit Neural Conditioning with Prior Knowledge Encodings.
Kazerouni et al., WACV 2024.

Faithful port of third_party/INCODE-main/modules/incode.py with two
intentional deviations:
  1. Positional encoding is omitted.
  2. forward(coords) returns just `output` (not [output, coef]) so existing
     callers `pred = model(coords).clamp(...)` keep working. The 4 modulation
     scalars are exposed as `self.last_coef` after every forward; apply the
     INCODE auxiliary regularization in your training loop, e.g.
         pred = model(coords)
         loss = recon(pred, gt) + lam * (model.last_coef ** 2).sum()

Audio task is also dropped (it pulled in a torchaudio dependency unused by
the image / 3D benchmark).
"""

import numpy as np
import torch
from torch import nn
import torchvision.models as tv_models
import torchvision.models.video as tv_video


# ---------------------------------------------------------------------------
# Auxiliary MLP (Harmonizer head)
# ---------------------------------------------------------------------------

class MLP(nn.Sequential):
    """
    Mirrors third_party MLP exactly:
      Linear → (LayerNorm if task == 'denoising') → activation
        × (len(hidden_channels) - 1)
      Linear (final)
      Dropout
    Weights init: trunc_normal(std=0.001), bias = mlp_bias.
    """

    def __init__(self, in_channels, hidden_channels, mlp_bias,
                 task=None, activation_layer=nn.SiLU,
                 bias=True, dropout=0.0):
        super().__init__()
        self.mlp_bias = mlp_bias

        layers = []
        in_dim = in_channels
        for h in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, h, bias=bias))
            if task == 'denoising':
                layers.append(nn.LayerNorm(h))
            layers.append(activation_layer())
            in_dim = h
        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, self.mlp_bias)

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# 1D conv feature extractor used for inpainting in original INCODE
# ---------------------------------------------------------------------------

class Custom1DFeatureExtractor(nn.Module):
    """
    Mirrors third_party/INCODE-main:Custom1DFeatureExtractor.
    Expects GT shape (1, im_chans, L).
    """

    def __init__(self, im_chans=3, out_chans=(32, 64, 64)):
        super().__init__()
        c1, c2, c3 = out_chans
        self.conv1 = nn.Conv1d(im_chans, c1, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, stride=1, padding=1, groups=c1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=7, stride=1, padding=1, groups=c2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_channels = c3

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.gap(x).flatten(1)   # (B, C)
        return x


# ---------------------------------------------------------------------------
# Composer SineLayer
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    """
    SIREN-style layer modulated by (a, b, c, d):
        output = exp(a) * sin(exp(b) * omega_0 * Wx + c) + d
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
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
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                     np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x, a_param, b_param, c_param, d_param):
        z = self.linear(x)
        return torch.exp(a_param) * torch.sin(
            torch.exp(b_param) * self.omega_0 * z + c_param
        ) + d_param


# ---------------------------------------------------------------------------
# INR
# ---------------------------------------------------------------------------

class INR(nn.Module):
    """
    Backbone selection (mirrors third party):
      task == 'inpainting'                : Custom1DFeatureExtractor
      task == 'shape'  or in_features==3  : tv_video.<model_3d>()  (default r3d_18)
      otherwise                           : tv_models.<model_2d>() (default resnet34)

    Call set_gt(gt_tensor) before forward():
      2D image:    (1, 3, H, W)
      3D shape:    (1, 3, D, H, W)
      inpainting:  (1, 3, L)
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30,
                 task=None,
                 model_2d='resnet34', model_3d='r3d_18',
                 truncated_layer=5,
                 mlp_in_channels=None,
                 mlp_hidden_channels=(64, 32, 4),
                 mlp_bias=0.1,
                 mlp_dropout=0.0,
                 mlp_activation=nn.SiLU,
                 **kwargs):
        super().__init__()

        is_3d = (in_features == 3) or (task == 'shape')
        self.task = task
        self.hidden_layers = hidden_layers
        self.nonlin = SineLayer

        # ---- Feature extractor ------------------------------------------------
        if task == 'inpainting':
            self.feature_extractor = Custom1DFeatureExtractor(
                im_chans=3, out_chans=(32, 64, 64)
            )
            inferred_in = 64
        elif is_3d:
            base = getattr(tv_video, model_3d)(weights=None)
            self.feature_extractor = nn.Sequential(
                *list(base.children())[:truncated_layer]
            )
            inferred_in = 512   # r3d_18 layer4 output channels
        else:
            base = getattr(tv_models, model_2d)(weights=None)
            self.feature_extractor = nn.Sequential(
                *list(base.children())[:truncated_layer]
            )
            inferred_in = 64    # resnet34 layer1 output channels

        if mlp_in_channels is None:
            mlp_in_channels = inferred_in

        self.aux_mlp = MLP(
            in_channels=mlp_in_channels,
            hidden_channels=list(mlp_hidden_channels),
            mlp_bias=mlp_bias,
            task=task,
            activation_layer=mlp_activation,
            dropout=mlp_dropout,
        )

        # ---- Pooling for backbone features -----------------------------------
        if is_3d:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif task != 'inpainting':
            self.gap = nn.AdaptiveAvgPool1d(1)
        else:
            self.gap = None    # Custom1D pools internally

        # ---- Composer network -------------------------------------------------
        net = []
        net.append(self.nonlin(in_features, hidden_features,
                               is_first=True, omega_0=first_omega_0))
        for _ in range(hidden_layers):
            net.append(self.nonlin(hidden_features, hidden_features,
                                   is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-bound, bound)
            net.append(final_linear)
        else:
            net.append(self.nonlin(hidden_features, out_features,
                                   is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*net)
        self.outermost_linear = outermost_linear

        self._gt = None
        self.last_coef = None

    def set_gt(self, gt_tensor):
        self._gt = gt_tensor

    def forward(self, coords):
        assert self._gt is not None, "Call set_gt(gt) before forward()."

        feats = self.feature_extractor(self._gt)

        if self.task == 'inpainting':
            coef = self.aux_mlp(feats)                     # (1, 4)
        elif isinstance(self.gap, nn.AdaptiveAvgPool3d):
            pooled = self.gap(feats)[:, :, 0, 0, 0]        # (1, C)
            coef = self.aux_mlp(pooled)
        else:
            pooled = self.gap(
                feats.view(feats.size(0), feats.size(1), -1)
            )[..., 0]                                      # (1, C)
            coef = self.aux_mlp(pooled)

        self.last_coef = coef                              # (1, 4)
        a_param, b_param, c_param, d_param = coef[0]

        output = self.net[0](coords, a_param, b_param, c_param, d_param)
        for i in range(1, self.hidden_layers + 1):
            output = self.net[i](output, a_param, b_param, c_param, d_param)

        if self.outermost_linear:
            output = self.net[self.hidden_layers + 1](output)
        else:
            output = self.net[self.hidden_layers + 1](
                output, a_param, b_param, c_param, d_param
            )

        return output
