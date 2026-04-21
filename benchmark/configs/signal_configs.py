"""
Per-signal hyperparameter overrides.

Structure:
    SIGNAL_CONFIGS[signal_name][method_name] = dict(hyperparams)

Only list the signals/methods you want to tune.
For any (signal, method) pair not listed here, BENCHMARK_DEFAULTS is used as fallback.
"""

SIGNAL_CONFIGS = {
    '2d_bandlimited': {
        # 'siren': dict(
        #     hidden_features=256, hidden_layers=5,
        #     first_omega_0=30.0, hidden_omega_0=30.0,
        #     lr=5e-4, scheduler='cosine',
        # ),
    },
    '2d_sierpinski': {
    },
    '2d_sphere': {
    },
    '2d_startarget': {
    },
    '3d_bandlimited': {
    },
    '3d_sphere': {
    },
}
