from . import siren
from . import wire
from . import gauss
from . import finerpp
from . import staf
from . import relu as pemlp
from . import incode
from . import sl2a
from . import cosmo

# ---------------------------------------------------------------------------
# Model registry: method name → class
# ---------------------------------------------------------------------------

_model_dict = {
    'siren':   siren.INR,
    'wire':    wire.INR,
    'gauss':   gauss.INR,
    'finer':   finerpp.FinerPPSine,    # Finer++Sine
    'gf':      finerpp.FinerPPGauss,   # Finer++Gauss
    'wf':      finerpp.FinerPPWave,    # Finer++Wave
    'staf':    staf.INR,
    'pemlp':   pemlp.INR,
    'relu':    pemlp.INR,
    'incode':  incode.INR,
    'sl2a':    sl2a.INR,
    'cosmo':   cosmo.INR,
}

# ---------------------------------------------------------------------------
# Benchmark default hyperparameters
#
# Keys in TRAIN_KEYS are consumed by the training loop (lr, scheduler).
# All other keys are forwarded to the model constructor.
# wire2d excluded from DEFAULTS_3D (2D-specific activation).
# ---------------------------------------------------------------------------

TRAIN_KEYS = {'lr', 'scheduler'}

BENCHMARK_DEFAULTS = {
    'siren': dict(
        hidden_features=256, hidden_layers=5,
        first_omega_0=30.0, hidden_omega_0=30.0,
        lr=1e-4, scheduler='cosine',
    ),
    'wire': dict(
        hidden_features=300, hidden_layers=3,
        first_omega_0=20.0, hidden_omega_0=20.0, scale=10.0,
        lr=5e-3, scheduler='lambda',
    ),
    'gauss': dict(
        hidden_features=256, hidden_layers=3, scale=10.0,
        lr=1e-3, scheduler='cosine',
    ),
    'finer': dict(
        hidden_features=256, hidden_layers=3,
        first_omega_0=30.0, hidden_omega_0=30.0,
        lr=1e-4, scheduler='cosine',
    ),
    'gf': dict(
        hidden_features=256, hidden_layers=3,
        scale=3.0, omega=10.0, first_bias_scale=1.0,
        lr=1e-3, scheduler='cosine',
    ),
    'wf': dict(
        hidden_features=300, hidden_layers=3,
        scale=2.0, omega_w=4.0, omega=5.0, first_bias_scale=1.0,
        lr=1e-3, scheduler='cosine',
    ),
    'staf': dict(
        hidden_features=256, hidden_layers=3,
        first_omega_0=30.0, hidden_omega_0=30.0, tau=5, skip_conn=False,
        lr=1e-4, scheduler='cosine',
    ),
    'relu': dict(
        hidden_features=256, hidden_layers=4,
        lr=5e-4, scheduler='cosine',
    ),
    'incode': dict(
        hidden_features=256, hidden_layers=3,
        first_omega_0=30.0, hidden_omega_0=30.0,
        lr=1e-4, scheduler='cosine',
    ),
    'sl2a': dict(
        hidden_features=256, hidden_layers=3,
        degree=256, rank=128, init_method='xavier_uniform',
        linear_init_type='kaiming_uniform', nonlinearity='relu',
        lr=1e-3, scheduler='cosine',
    ),
    'cosmo': dict(
        hidden_features=256, hidden_layers=3,
        beta0=0.05, T_range=(0.5, 5.0), c_range=(0.0, 3.0),
        lr=1e-2, scheduler='cosine',
    ),
}

# 3D excludes wire2d
BENCHMARK_DEFAULTS_3D = {k: v for k, v in BENCHMARK_DEFAULTS.items()
                         if k != 'wire2d'}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_INR(method, in_features, hidden_features, hidden_layers,
            out_features, **kwargs):
    """
    Instantiate an INR model by name.

    Args:
        method:          one of the registered method names
        in_features:     input coordinate dimension (2 for image, 3 for volume)
        hidden_features: nominal hidden width
        hidden_layers:   number of hidden layers
        out_features:    output channels (3 for RGB, 1 for grayscale/occupancy)
        **kwargs:        method-specific hyperparameters

    Returns:
        nn.Module instance
    """
    if method not in _model_dict:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Available: {sorted(_model_dict.keys())}")
    return _model_dict[method](
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        **kwargs,
    )
