"""
DIV2K dataset loader (Agustsson & Timofte, CVPRW 2017).

High-resolution 2K images — 800 training, 100 validation.
Accepted layouts under `data_root`:

    data/div2k/0001.png … 0800.png                 (flat; this repo's current layout)
    data/div2k/train/0001.png … 0800.png           (official split)
    data/div2k/val/0801.png   … 0900.png

The loader auto-detects the layout and returns the same
(coords, pixels, meta) tuple as KodakDataset, so every 2D task works
transparently against either dataset.
"""

import os
import glob
import numpy as np
import torch
from PIL import Image


class DIV2KDataset:
    """
    Each item returns:
        coords: (H*W, 2)  float32 in [-1, 1]
        pixels: (H*W, 3)  float32 in [0, 1]
        meta:   dict with 'path', 'H', 'W', 'C', 'name'
    """

    _PATTERNS = ['{:04d}.png', '{:04d}.jpg']
    _SUBDIRS = ['', 'train', 'val', 'DIV2K_train_HR', 'DIV2K_valid_HR']

    def __init__(self, data_root, indices=None, normalize=True,
                 max_size=None):
        """
        Args:
            data_root:  path to the DIV2K folder
            indices:    list of 1-based image indices; default = all found
            normalize:  if True pixel values are in [0,1]; else [0,255]
            max_size:   if set, longer side is resized to this many pixels
                        (DIV2K images are ~2040 px, which can OOM on CPU —
                        set e.g. 512 for quick tests)
        """
        self.data_root = data_root
        self.normalize = normalize
        self.max_size = max_size
        self.paths = self._discover(data_root, indices)
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No DIV2K images found under '{data_root}'. "
                "Expected files like 0001.png (flat) or train/0001.png.")

    def _discover(self, root, indices):
        found = []
        iter_range = indices if indices is not None else range(1, 901)
        for i in iter_range:
            for sub in self._SUBDIRS:
                for pat in self._PATTERNS:
                    p = os.path.join(root, sub, pat.format(i)) if sub \
                        else os.path.join(root, pat.format(i))
                    if os.path.exists(p):
                        found.append(p)
                        break
                if found and found[-1].endswith(('.png', '.jpg')) and \
                   os.path.basename(found[-1]).startswith(f'{i:04d}'):
                    break
        # Fallback: glob every png/jpg in root (+ subdirs)
        if not found:
            candidates = sorted(
                glob.glob(os.path.join(root, '*.png')) +
                glob.glob(os.path.join(root, '*.jpg')) +
                glob.glob(os.path.join(root, '*', '*.png')) +
                glob.glob(os.path.join(root, '*', '*.jpg'))
            )
            if indices is not None:
                found = [candidates[i - 1] for i in indices
                         if 0 <= i - 1 < len(candidates)]
            else:
                found = candidates
        return found

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')

        if self.max_size is not None:
            w, h = img.size
            longer = max(w, h)
            if longer > self.max_size:
                scale = self.max_size / longer
                new_wh = (int(round(w * scale)), int(round(h * scale)))
                img = img.resize(new_wh, Image.BICUBIC)

        arr = np.array(img, dtype=np.float32)
        if self.normalize:
            arr = arr / 255.0

        H, W, C = arr.shape
        pixels = torch.from_numpy(arr).reshape(-1, C)

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
