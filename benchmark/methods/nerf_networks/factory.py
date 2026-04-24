"""
Factory for building a NeRFNetwork with a named activation.
"""

from .network import NeRFNetwork
from .activation_layers import available_activations


def build_nerf_network(activation, **kwargs):
    """
    Args:
        activation: one of available_activations() — 'siren', 'finer',
                    'gauss', 'wire', 'relu'
        **kwargs:   forwarded to NeRFNetwork (num_layers, hidden_dim,
                    geo_feat_dim, num_layers_color, hidden_dim_color,
                    bound, cuda_ray, min_near, density_thresh, bg_radius,
                    activation-specific: first_omega_0, hidden_omega_0,
                    first_bias_scale, scale_req_grad, scale, ...)
    """
    return NeRFNetwork(activation=activation, **kwargs)


__all__ = ['build_nerf_network', 'available_activations']
