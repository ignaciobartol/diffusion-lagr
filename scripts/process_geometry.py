#!/usr/bin/env python
"""
process_geometry.py - Computes the signed distance function map (SDF)
and adds the metadata.

Example
-------
    python scripts/
"""

import argparse
import numpy as np
import json
from scipy.ndimage import distance_transform_edt
from stl_to_numpy import STLToNumpyConverter, VoxelizationConfig

class MetadataVoxelizer(STLToNumpyConverter):
    """Subclass that exposes the grid metadata (origin, spacing)
    after voxelization.
    """
    def convert_with_metadata(self, stl_path):
        poly = self._read_stl(stl_path)
        if self.cfg.require_watertight:
            self._assert_watertight(poly)
        
        origin, spacing, dims = self._compute_grid(poly)
        self._guard_memory(dims, self.cfg.dtype)
        vol = self._voxelize(poly, origin, spacing, dims)

        return vol, origin, spacing
    
    def generate_sdf(self, binary_volume, spacing):
        """
        Computes the signed distance function (SDF)
        from a binary volume.
        
        :param binary_volume: [D, W, H] binary numpy array
        :param spacing: tuple (dx, dy, dz) voxel spacing.
        """
        # Need to implement anisotropic spacing
        avg_spacing = np.mean(spacing)

        dist_inside = distance_transform_edt(binary_volume,
                                             sampling=avg_spacing)
        dist_outside = distance_transform_edt(1 - binary_volume,
                                              sampling=avg_spacing)
        return dist_outside - dist_inside

def main():
    parser = argparse.ArgumentParser(
        description = "Process STL to compute SDF and add metadata.")
    parser.add_argument("input_stl", type=str, help="Input STL file path.")
    parser.add_argument("output_prefix", type=str, help="e.g. 'data/patient_01'")
    parser.add_argument("--dim", type=int, default=64, help="Voxel grid dimension.")
    args = parser.parse_args()

    cfg = VoxelizationConfig(
        target_max_dim=args.dim,
        dtype=np.uint8,
        fill_value=1,
        background_value=0,
    )

    converter = MetadataVoxelizer(cfg)
    binary_vol, origin, spacing = converter.convert_with_metadata(args.input_stl)
    sdf_vol = converter.generate_sdf(binary_vol, spacing)
    np.savez_compressed(f"{args.output_prefix}_geometry.npz",
                        binary=binary_vol,
                        sdf=sdf_vol.astype(np.float32),
                        origin=np.array(origin, dtype=np.float32),
                        spacing=np.array(spacing, dtype=np.float32))
    print(f"Saved geometry and metadata to {args.output_prefix}_geometry.npz")
    print(f"Origin: {origin}, Spacing: {spacing}")

if __name__ == "__main__":
    main()