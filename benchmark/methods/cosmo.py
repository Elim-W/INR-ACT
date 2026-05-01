import numpy as np
import torch
from torch import nn
import torchvision.models as tv_models
import torchvision.models.video as tv_video


# ---------------------------------------------------------------------------
# Auxiliary MLP (prior knowledge embedder head)
# Mirrors the original `MLP` class.
# ---------------------------------------------------------------------------

class _MLP(nn.Sequential):
    def __init__(self, in_channels, hidden_channels, mlp_bias=0.1,
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
# Raised cosine impulse response layer with complex modulation (paper Eq. 8)
# Mirrors `RaisedCosineImpulseResponseLayer` from BandRC.py.
# ---------------------------------------------------------------------------

class RaisedCosineImpulseResponseLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False,
                 beta0=0.05, eps=1e-8, out_real=False):
        super().__init__()

        # β fixed (non-trainable) — matches original requires_grad=False.
        self.beta0 = nn.Parameter(torch.tensor(beta0, dtype=torch.float),
                                  requires_grad=False)
        self.eps = eps
        self.is_first = is_first
        self.out_real = out_real

        self.in_features = in_features
        self.out_features = out_features

        dtype = torch.float if self.is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        nn.init.uniform_(self.linear.weight,
                         -1 / self.in_features, 1 / self.in_features)

    def forward(self, input, t0, c0):
        # Move input to module's device, matching original behavior
        input = input.to(next(self.parameters()).device)
        lin = self.linear(input)

        # eps placement matches original: |lin + eps| (not |lin| + eps)
        if not self.is_first:
            lin = lin / torch.abs(lin + self.eps)

        f1 = (1.0 / t0) * torch.sinc(lin / t0) \
             * torch.cos(torch.pi * self.beta0 * lin / t0)
        f2 = 1.0 - (2.0 * self.beta0 * lin / t0) ** 2 + self.eps
        theta = 2.0 * torch.pi * c0 * lin * 1j

        rc = f1 / f2
        out = rc * torch.exp(theta)

        if not self.is_first:
            out = out / torch.abs(out + self.eps)

        return out.real if self.out_real else out


# ---------------------------------------------------------------------------
# INR
# ---------------------------------------------------------------------------

class INR(nn.Module):
    """
    Backbone selection (matches paper Section 3.2):
      in_features == 2  →  ResNet-34   (first 5 layers, ImageNet pretrained)
      in_features == 3  →  ResNet3D-18 (first 5 layers, Kinetics pretrained)

    Call set_gt(gt_tensor) before forward():
      2D image:  (1, 3, H, W)
      3D shape:  (1, 3, D, H, W)
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 T_range=(0.5, 5.0), c_range=(0.0, 3.0),
                 truncated_layer=5, mlp_bias=0.1, mlp_dropout=0.0,
                 mlp_hidden=(64, 32),
                 **kwargs):
        super().__init__()

        is_3d = (in_features == 3)
        self.is_3d = is_3d
        self.T_range = T_range
        self.c_range = c_range
        self.hidden_layers = hidden_layers
        # number of RC layers = first layer + hidden layers (paper Fig.1)
        self.num_rc_layers = hidden_layers + 1

        # ---- Feature extractor ------------------------------------------------
        if is_3d:
            # 3D occupancy branch (paper Sec. 3.2: "we instead input the 3D
            # voxel/point vector into the first five layers of ResNet3D-18")
            try:
                base = tv_video.r3d_18(weights='DEFAULT')
            except Exception:
                import warnings
                warnings.warn(
                    'r3d_18 pretrained weights unavailable; falling back to '
                    'random init for the 3D Harmonizer.')
                base = tv_video.r3d_18(weights=None)
            feat_channels = 512   # output of layer4 in r3d_18
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            base = tv_models.resnet34(weights='DEFAULT')
            feat_channels = 64    # output of layer1 in resnet34
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.feature_extractor = nn.Sequential(
            *list(base.children())[:truncated_layer]
        )

        # ---- Prior MLP (outputs T,c per RC layer) ----------------------------
        # Original paper: 2 values × 4 layers = 8 outputs (hard-coded view(4,2)).
        # Generalised here to 2 × num_rc_layers so we can change depth.
        self.prior = _MLP(
            in_channels=feat_channels,
            hidden_channels=list(mlp_hidden) + [self.num_rc_layers * 2],
            mlp_bias=mlp_bias,
            task=None,
            activation_layer=nn.SiLU,
            dropout=mlp_dropout,
        )

        # ---- RC composer -----------------------------------------------------
        self.nonlin = RaisedCosineImpulseResponseLayer
        self.net = nn.ModuleList()
        self.net.append(self.nonlin(in_features, hidden_features, is_first=True))
        for _ in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features))

        # Final complex linear → take .real → sigmoid
        self.final_linear = nn.Linear(hidden_features, out_features,
                                      dtype=torch.cfloat)

        self._gt = None

    # ---------- API ---------------------------------------------------------
    def set_gt(self, gt_tensor):
        """
        2D: gt_tensor (1, 3, H, W)
        3D: gt_tensor (1, 3, D, H, W)
        """
        self._gt = gt_tensor

    # ---------- forward -----------------------------------------------------
    def forward(self, coords):
        assert self._gt is not None, "Call set_gt(gt) before forward()."

        feats = self.feature_extractor(self._gt)

        # Spatial pooling → (1, C)
        if self.is_3d:
            pooled = self.gap(feats)[:, :, 0, 0, 0]                  # (1, C)
        else:
            # feats: (1, C, H', W') → AdaptiveAvgPool2d(1) → (1, C, 1, 1) → flatten → (1, C)
            pooled = self.gap(feats).flatten(1)                      # (1, C)

        # Prior MLP → (num_rc_layers, 2)
        coef = self.prior(pooled).view(self.num_rc_layers, 2)
        t0, c0 = torch.unbind(coef, dim=1)

        # Sigmoid projection to keep T,c in valid ranges
        t0 = torch.sigmoid(t0) * (self.T_range[1] - self.T_range[0]) + self.T_range[0]
        c0 = torch.sigmoid(c0) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]

        output = coords
        for i, lyr in enumerate(self.net):
            output = lyr(output, t0[i], c0[i])
        output = self.final_linear(output).real
        return torch.sigmoid(output)