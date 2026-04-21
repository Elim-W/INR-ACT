from . import image_fitting
from . import image_denoising
from . import image_inpainting
from . import image_super_resolution
from . import shape_occupancy
from . import nerf
from . import sdf

_task_dict = {
    'image_fitting':          image_fitting,
    'image_denoising':        image_denoising,
    'image_inpainting':       image_inpainting,
    'image_super_resolution': image_super_resolution,
    'shape_occupancy':        shape_occupancy,
    'nerf':                   nerf,
    'sdf':                    sdf,
}


def get_task(name):
    if name not in _task_dict:
        raise ValueError(f"Unknown task '{name}'. Available: {list(_task_dict.keys())}")
    return _task_dict[name]
