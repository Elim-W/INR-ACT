from . import image_fitting
from . import image_denoising
from . import image_inpainting
from . import image_super_resolution
from . import image_ct_reconstruction
from . import shape_occupancy
from . import sdf
# `nerf` is imported lazily in get_task() because it requires torch-ngp's
# CUDA extensions to be compiled, which we only do on GPU compute nodes.

_task_dict = {
    'image_fitting':            image_fitting,
    'image_denoising':          image_denoising,
    'image_inpainting':         image_inpainting,
    'image_super_resolution':   image_super_resolution,
    'image_ct_reconstruction':  image_ct_reconstruction,
    'shape_occupancy':          shape_occupancy,
    'sdf':                      sdf,
}


def get_task(name):
    if name == 'nerf':
        # Import lazily so CPU-only / login-node runs of other tasks
        # don't fail when torch-ngp's CUDA extensions aren't compiled.
        from . import nerf
        return nerf
    if name not in _task_dict:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(_task_dict.keys()) + ['nerf']}")
    return _task_dict[name]
