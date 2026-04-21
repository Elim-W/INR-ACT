import os
import glob
import numpy as np
import torch
from PIL import Image


class KodakDataset:
    """
    Kodak lossless true color image suite (24 images, 768×512 or 512×768).
    Images should be placed under data_root as kodak01.png … kodak24.png
    or im01.png … im24.png (both naming conventions accepted).

    Each item returns:
        coords: (H*W, 2)  float32 in [-1, 1]
        pixels: (H*W, 3)  float32 in [0, 1]
        meta:   dict with 'path', 'H', 'W', 'name'
    """

    _PATTERNS = ['kodak{:02d}.png', 'im{:02d}.png',
                 'kodim{:02d}.png', '{:02d}.png']

    def __init__(self, data_root, indices=None, normalize=True):
        """
        Args:
            data_root:  path to the folder containing Kodak images
            indices:    list of 1-based image indices to include (default: all found)
            normalize:  if True, pixel values are in [0,1]; if False, [0,255]
        """
        self.data_root = data_root
        self.normalize = normalize
        self.paths = self._discover(data_root, indices)
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No Kodak images found in '{data_root}'. "
                "Expected filenames like kodak01.png, kodim01.png, or 01.png.")

    def _discover(self, root, indices):
        found = []
        # Try pattern-based search first
        for i in (indices or range(1, 25)):
            for pat in self._PATTERNS:
                p = os.path.join(root, pat.format(i))
                if os.path.exists(p):
                    found.append(p)
                    break
        # Fallback: grab any .png/.jpg in the folder
        if not found:
            found = sorted(
                glob.glob(os.path.join(root, '*.png')) +
                glob.glob(os.path.join(root, '*.jpg'))
            )
            if indices is not None:
                found = [found[i - 1] for i in indices if i - 1 < len(found)]
        return found

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
        if self.normalize:
            img = img / 255.0

        H, W, C = img.shape
        pixels = torch.from_numpy(img).reshape(-1, C)  # (H*W, 3)

        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        meta = {'path': path, 'H': H, 'W': W, 'C': C,
                'name': os.path.splitext(os.path.basename(path))[0]}
        return coords, pixels, meta

    def iter_images(self):
        for i in range(len(self)):
            yield self[i]
