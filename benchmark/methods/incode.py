"""
INCODE: Implicit Neural Conditioning with Prior Knowledge Encodings.
Kazerouni et al., WACV 2024.

Faithful port of third_party/INCODE-main/modules/incode.py. Class layout,
MLP_configs dict style, SineLayer math, and forward control flow all mirror
the upstream file. Three intentional deviations are kept:

  1. Positional encoding is omitted (`pos_encode_configs` argument removed,
     no Encoding module imported).
  2. Audio task is dropped (it pulled in a torchaudio dependency unused by
     the image / 3D benchmark).
  3. forward(coords) returns just `output` (not [output, coef]) so existing
     callers `pred = model(coords).clamp(...)` keep working. The 4 modulation
     scalars are exposed as `self.last_coef` after every forward, and the
     INCODE auxiliary regularization is provided as a model-side method
     `aux_loss()`. Training loops add it via:

         pred = model(coords)
         loss = recon(pred, gt)
         if hasattr(model, 'aux_loss'):
             loss = loss + model.aux_loss()

The Harmonizer ground-truth tensor is supplied via `model.set_context(gt)`
(alias `set_gt`) rather than `MLP_configs['GT']`, because the benchmark
builds models before the GT tensor is materialised on device.
"""

import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torchvision.models.video as video


# Default per-coefficient ReLU(-·) penalty weights from the upstream
# train_*.ipynb defaults (image / sr / denoising / inpainting / sdf / ct
# all use the same numbers).
INCODE_REG_DEFAULTS = (0.1993, 0.0196, 0.0588, 0.0269)


# ---------------------------------------------------------------------------
# Auxiliary MLP (Harmonizer head)
# ---------------------------------------------------------------------------

class MLP(torch.nn.Sequential):
    '''
    Args:
        in_channels (int): Number of input channels or features.
        hidden_channels (list of int): List of hidden layer sizes. The last element is the output size.
        mlp_bias (float): Value for initializing bias terms in linear layers.
        activation_layer (torch.nn.Module, optional): Activation function applied between hidden layers. Default is SiLU.
        bias (bool, optional): If True, the linear layers include bias terms. Default is True.
        dropout (float, optional): Dropout probability applied after the last hidden layer. Default is 0.0 (no dropout).
    '''
    def __init__(self, MLP_configs, bias=True, dropout=0.0):
        super().__init__()

        in_channels = MLP_configs['in_channels']
        hidden_channels = MLP_configs['hidden_channels']
        self.mlp_bias = MLP_configs['mlp_bias']
        activation_layer = MLP_configs['activation_layer']

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if MLP_configs['task'] == 'denoising':
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.001)
            torch.nn.init.constant_(m.bias, self.mlp_bias)

    def forward(self, x):
        out = self.layers(x)
        return out


# ---------------------------------------------------------------------------
# 1D conv feature extractor used for inpainting in original INCODE
# ---------------------------------------------------------------------------

class Custom1DFeatureExtractor(nn.Module):
    def __init__(self, im_chans, out_chans):
        super(Custom1DFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=im_chans, out_channels=out_chans[0], kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=out_chans[1], kernel_size=5, stride=1, padding=1, groups=32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=out_chans[2], kernel_size=7, stride=1, padding=1, groups=64)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


# ---------------------------------------------------------------------------
# Composer SineLayer
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    '''
    SineLayer is a custom PyTorch module that applies a modified Sinusoidal activation function to the output of a linear transformation
    with adjustable parameters.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.
        is_first (bool, optional): If True, initializes the weights with a narrower range. Default is False.
        omega_0 (float, optional): Frequency scaling factor for the sinusoidal activation. Default is 30.

    Additional Parameters:
        a_param (float): Exponential scaling factor for the sine function. Controls the amplitude.
        b_param (float): Exponential scaling factor for the frequency.
        c_param (float): Phase shift parameter for the sine function.
        d_param (float): Bias term added to the output.
    '''
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input, a_param, b_param, c_param, d_param):
        output = self.linear(input)
        output = torch.exp(a_param) * torch.sin(torch.exp(b_param) * self.omega_0 * output + c_param) + d_param
        return output


# ---------------------------------------------------------------------------
# INR (Composer + Harmonizer)
# ---------------------------------------------------------------------------

class INR(nn.Module):
    """
    Mirrors third_party/INCODE-main:INR. See module docstring for the three
    deviations. Constructor accepts flat kwargs (benchmark calling convention
    via `get_INR(...)`); they are assembled into the same `MLP_configs` dict
    that upstream uses, so MLP / feature_extractor / pooling branches behave
    identically to the third_party reference.

    Args (relevant ones):
        task: upstream task string. Affects two switch points:
              - 'denoising'  → adds LayerNorm to aux MLP
              - 'inpainting' → uses Custom1DFeatureExtractor
              - 'shape'      → uses 3D video backbone + 3D pool
              - anything else (incl. 'image', 'sr') → 2D image backbone + 1D pool
        reg_weights: per-coef weights for the aux ReLU(-·) penalty. Defaults
              to the upstream notebook values.
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30,
                 task='image',
                 model_2d='resnet34', model_3d='r3d_18',
                 truncated_layer=None,
                 mlp_in_channels=None,
                 mlp_hidden_channels=None,
                 mlp_bias=None,
                 mlp_activation=nn.SiLU,
                 mlp_dropout=0.0,
                 reg_weights=INCODE_REG_DEFAULTS,
                 **kwargs):
        super().__init__()

        # Per-task defaults from upstream train_*.ipynb. Users can override via
        # explicit kwargs. Values verified by reading the upstream notebooks.
        if task == 'shape':
            # train_sdf.ipynb: r3d_18 truncated@3 → 128 channels.
            if truncated_layer is None: truncated_layer = 3
            if mlp_in_channels  is None: mlp_in_channels  = 128
            if mlp_hidden_channels is None: mlp_hidden_channels = (64, 32, 4)
            if mlp_bias is None: mlp_bias = 0.3120
        elif task == 'denoising':
            # train_denoising.ipynb: resnet34 truncated@5 → 64 ch, deeper aux MLP.
            if truncated_layer is None: truncated_layer = 5
            if mlp_in_channels  is None: mlp_in_channels  = 64
            if mlp_hidden_channels is None: mlp_hidden_channels = (32, 16, 8, 4)
            if mlp_bias is None: mlp_bias = 0.0005
        elif task == 'inpainting':
            # train_inpainting.ipynb: Custom1DFeatureExtractor → 64 ch, no
            # truncated_layer (Custom1D doesn't use a torchvision backbone).
            if mlp_in_channels  is None: mlp_in_channels  = 64
            if mlp_hidden_channels is None: mlp_hidden_channels = (64, 32, 4)
            if mlp_bias is None: mlp_bias = 0.3120
        else:
            # image / sr / ct all use the same image notebook config.
            if truncated_layer is None: truncated_layer = 5
            if mlp_in_channels  is None: mlp_in_channels  = 64
            if mlp_hidden_channels is None: mlp_hidden_channels = (64, 32, 4)
            if mlp_bias is None: mlp_bias = 0.3120

        MLP_configs = {
            'task': task,
            'model': model_3d if task == 'shape' else model_2d,
            'truncated_layer': truncated_layer,
            'in_channels': mlp_in_channels,
            'hidden_channels': list(mlp_hidden_channels),
            'mlp_bias': mlp_bias,
            'activation_layer': mlp_activation,
        }

        self.task = task
        self.nonlin = SineLayer
        self.hidden_layers = hidden_layers
        self.outermost_linear = outermost_linear
        self.reg_weights = tuple(reg_weights)

        # Harmonizer network — task-conditional feature extractor
        if MLP_configs['task'] == 'shape':
            model_ft = getattr(video, MLP_configs['model'])(weights=None)
            self.feature_extractor = nn.Sequential(*list(model_ft.children())[:MLP_configs['truncated_layer']])
        elif MLP_configs['task'] == 'inpainting':
            self.feature_extractor = Custom1DFeatureExtractor(im_chans=3, out_chans=[32, 64, 64])
        else:
            model_ft = getattr(models, MLP_configs['model'])(weights=None)
            self.feature_extractor = nn.Sequential(*list(model_ft.children())[:MLP_configs['truncated_layer']])

        self.aux_mlp = MLP(MLP_configs, dropout=mlp_dropout)

        if MLP_configs['task'] == 'shape':
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            if MLP_configs['task'] != 'inpainting':
                self.gap = nn.AdaptiveAvgPool1d(1)

        # Composer Network
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)

            with torch.no_grad():
                const = np.sqrt(6 / hidden_features) / max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)

            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

        # Harmonizer conditioning signal — set via set_context() before forward.
        self.ground_truth = None
        # Cached modulation coefficients from the most recent forward pass —
        # consumed by aux_loss().
        self.last_coef = None

    def set_context(self, ctx_tensor):
        """Provide the Harmonizer's conditioning signal (replaces upstream
        MLP_configs['GT']). Stores the tensor only — no precomputation, since
        feature_extractor / aux_mlp are trainable and `coef` must be recomputed
        every forward.

        Expected shapes (all batched at dim 0 with C=3 broadcast):
            task='image' / 'denoising'  → (1, 3, H, W)
            task='inpainting'           → (1, 3, L)
            task='shape'                → (1, 3, D, H, W)
        """
        self.ground_truth = ctx_tensor

    # Backwards-compatible alias used by existing runners (set_gt was the
    # previous adapter name; kept so we don't churn callers in this pass).
    set_gt = set_context

    def aux_loss(self):
        """INCODE auxiliary regularization on the modulation coefficients.
        Returns a 0-d tensor on the same device as the model parameters.
        """
        if self.last_coef is None:
            return torch.zeros((), device=next(self.parameters()).device)
        a, b, c, d = self.last_coef[0]
        wa, wb, wc, wd = self.reg_weights
        return (wa * torch.relu(-a) + wb * torch.relu(-b)
              + wc * torch.relu(-c) + wd * torch.relu(-d))

    def forward(self, coords):
        assert self.ground_truth is not None, \
            "INCODE INR: call set_context(gt) before forward()."

        extracted_features = self.feature_extractor(self.ground_truth)
        if self.task == 'shape':
            gap = self.gap(extracted_features)[:, :, 0, 0, 0]
            coef = self.aux_mlp(gap)
        elif self.task == 'inpainting':
            coef = self.aux_mlp(extracted_features)
        else:
            gap = self.gap(extracted_features.view(extracted_features.size(0), extracted_features.size(1), -1))
            coef = self.aux_mlp(gap[..., 0])

        self.last_coef = coef
        a_param, b_param, c_param, d_param = coef[0]

        output = self.net[0](coords, a_param, b_param, c_param, d_param)

        for i in range(1, self.hidden_layers + 1):
            output = self.net[i](output, a_param, b_param, c_param, d_param)

        if self.outermost_linear:
            output = self.net[self.hidden_layers + 1](output)
        else:
            output = self.net[self.hidden_layers + 1](output, a_param, b_param, c_param, d_param)

        return output
