"""
Stanford 3D Scan dataset — PLY meshes → voxel occupancy grid.

Loads .ply meshes with trimesh, computes a cubic occupancy grid via
ray-casting `contains()`, and caches the result next to the ply file so
subsequent runs are instantaneous.

Each item yields:
    coords:     (N, 3) float32 in [-1, 1]^3
    occupancy:  (N, 1) float32  — 0 or 1
    meta:       dict with 'path', 'name', 'grid_res',
                'H', 'W', 'T', 'mesh_whl' (half-extents of the bbox),
                'n_points'

Layout under data_root:
    stanford_3d_models/
        bunny/reconstruction/bun_zipper_res4.ply
        dragon_recon/dragon_vrip_res4.ply

`mesh_ids` is a list of logical names (e.g. ['bunny','dragon']); each is
resolved to a .ply file via the _MESH_MAP table.  Unknown names are
resolved by globbing (any .ply found under a folder with that name).
"""

import os
import glob
import numpy as np
import torch


# Default ply file per logical mesh name.  Add more as the dataset grows.
_MESH_MAP = {
    'bunny':  ['bunny/reconstruction/bun_zipper_res4.ply',
               'bunny/reconstruction/bun_zipper_res3.ply',
               'bunny/reconstruction/bun_zipper.ply'],
    'dragon': ['dragon_recon/dragon_vrip_res4.ply',
               'dragon_recon/dragon_vrip_res3.ply',
               'dragon_recon/dragon_vrip.ply'],
}


class Stanford3DDataset:

    def __init__(self, data_root, mesh_ids=None, grid_res=64,
                 chunk_points=4096, cache=True, **_):
        """
        Args:
            data_root:    path to `stanford_3d_models/`
            mesh_ids:     list of mesh names (default: ['bunny','dragon'])
            grid_res:     side length of the voxel grid (default 64 → 262k pts)
            chunk_points: batch size for mesh.contains() to control RAM
            cache:        if True, cache occupancy grid to .npz next to ply
        """
        self.data_root = data_root
        self.mesh_ids = mesh_ids or list(_MESH_MAP.keys())
        self.grid_res = int(grid_res)
        self.chunk_points = int(chunk_points)
        self.cache = cache
        self.paths = [self._resolve(name) for name in self.mesh_ids]

    def _resolve(self, name):
        # Try the predefined map first
        candidates = []
        if name in _MESH_MAP:
            candidates.extend(os.path.join(self.data_root, rel)
                              for rel in _MESH_MAP[name])
        # Fallback: glob any .ply under a folder of that name
        candidates.extend(sorted(
            glob.glob(os.path.join(self.data_root, name, '*.ply')) +
            glob.glob(os.path.join(self.data_root, name, '**/*.ply'),
                      recursive=True)))
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"No .ply found for mesh '{name}' under '{self.data_root}'.")

    def _cache_path(self, ply_path):
        base, _ = os.path.splitext(ply_path)
        return f"{base}_occ{self.grid_res}.npz"

    def _build_occupancy(self, ply_path):
        """Returns (occupancy_bool (N,N,N), mesh_whl (3,))."""
        cache_path = self._cache_path(ply_path)
        if self.cache and os.path.exists(cache_path):
            data = np.load(cache_path)
            return data['occ'].astype(bool), data['mesh_whl'].astype(np.float32)

        # Lazy import so `get_dataset('kodak', ...)` doesn't require trimesh.
        import trimesh
        mesh = trimesh.load(ply_path, force='mesh', skip_materials=True)

        # Center at origin
        mins = mesh.vertices.min(axis=0)
        maxs = mesh.vertices.max(axis=0)
        center = (mins + maxs) / 2
        mesh.vertices = mesh.vertices - center
        mesh_whl = ((maxs - mins) / 2 * 1.02).astype(np.float32)  # 2% margin

        # Build voxel grid in world space
        N = self.grid_res
        xs = np.linspace(-mesh_whl[0], mesh_whl[0], N)
        ys = np.linspace(-mesh_whl[1], mesh_whl[1], N)
        zs = np.linspace(-mesh_whl[2], mesh_whl[2], N)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)

        # contains() is memory-heavy (ray cast per face); chunk it.
        occ_flat = np.zeros(pts.shape[0], dtype=bool)
        for i in range(0, pts.shape[0], self.chunk_points):
            occ_flat[i:i + self.chunk_points] = mesh.contains(
                pts[i:i + self.chunk_points])
        occ = occ_flat.reshape(N, N, N)

        if self.cache:
            np.savez_compressed(cache_path, occ=occ, mesh_whl=mesh_whl)

        return occ, mesh_whl

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ply_path = self.paths[idx]
        occ, mesh_whl = self._build_occupancy(ply_path)
        N = occ.shape[0]

        # Normalised coords in [-1, 1]^3 — keep the same orientation as the grid
        xs = torch.linspace(-1, 1, N)
        ys = torch.linspace(-1, 1, N)
        zs = torch.linspace(-1, 1, N)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
        coords = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)

        occupancy = torch.from_numpy(occ.astype(np.float32)).reshape(-1, 1)
        name = self.mesh_ids[idx]
        meta = {
            'path': ply_path,
            'name': name,
            'grid_res': N,
            'H': N, 'W': N, 'T': N,
            'mesh_whl': mesh_whl,
            'n_points': coords.shape[0],
        }
        return coords, occupancy, meta

    def iter_shapes(self):
        for i in range(len(self)):
            yield self[i]

    # Alias so the run_experiment.py dispatcher can stay uniform if desired
    iter_images = iter_shapes
