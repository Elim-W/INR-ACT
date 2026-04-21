# Placeholder — Blender (Synthetic NeRF) dataset loader (to be implemented)
#
# Blender dataset: 8 synthetic scenes (lego, drums, ship, …) with
# posed RGB images and camera intrinsics/extrinsics in transforms_*.json.
# Reference: Mildenhall et al., ECCV 2020.
#
# Expected structure:
#   data/blender_nerf/
#       lego/
#           transforms_train.json
#           transforms_val.json
#           transforms_test.json
#           train/  r_*.png
#           val/    r_*.png
#           test/   r_*.png


class BlenderNeRFDataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BlenderNeRF dataset is not yet implemented.")
