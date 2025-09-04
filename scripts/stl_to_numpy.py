#!/usr/bin/env python3
"""
STL to numpy voxelizer (binary volume) using VTK.

- Requires a *watertight* surface.
- Fills the closed interior with ones; exterior is zeros.
- Choose either --voxel-size or --max-dim (auto-spacing).

Example:
    python stl_to_numpy.py in.stl out.npy --max-dim 256
    python stl_to_numpy.py in.stl out.npy --voxel-size 0.25 --margin-voxels 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import vtk  # VTK is robust for rasterizing closed surfaces
    from vtkmodules.util import numpy_support
except Exception as e:
    print("ERROR: This script requires VTK (pip install vtk).", file=sys.stderr)
    raise

# ----------------------------- Config ----------------------------- #

@dataclass(frozen=True)
class VoxelizationConfig:
    """Configuration for the voxelization process."""
    # Grid definition:
    voxel_size: Optional[float] = None         # physical spacing in STL units
    target_max_dim: Optional[int] = 256        # fit longest bbox side to this many voxels

    # Padding and data type:
    margin_voxels: int = 2                     # extra voxels around the object
    dtype: np.dtype = np.uint8                 # 0/1 output type

    # Fill values:
    fill_value: int = 1
    background_value: int = 0

    # Safety:
    memory_limit_gb: float = 16.0

    # Validation:
    require_watertight: bool = True            # fail if surface has boundary edges


# ------------------------------- Converter -------------------------------- #

class STLToNumpyConverter:
    """Converts a watertight STL to a dense 3D NumPy array using VTK stenciling."""

    def __init__(self, config: VoxelizationConfig) -> None:
        self.cfg = config

    # ---- Public API ---- #
    def convert(self, stl_path: str) -> np.ndarray:
        poly = self._read_stl(stl_path)
        if self.cfg.require_watertight:
            self._assert_watertight(poly)

        origin, spacing, dims = self._compute_grid(poly)
        self._guard_memory(dims, self.cfg.dtype)

        vol = self._voxelize(poly, origin, spacing, dims)
        return vol

    # ---- Implementation ---- #
    def _read_stl(self, path: str) -> "vtk.vtkPolyData":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"STL not found: {path}")

        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()

        if poly is None or poly.GetNumberOfPoints() == 0 or poly.GetNumberOfPolys() == 0:
            raise ValueError(f"Failed to read valid triangles from STL: {path}")

        return poly

    def _assert_watertight(self, poly: "vtk.vtkPolyData") -> None:
        """Fail if the surface has open boundaries or non-manifold edges."""
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(poly)
        fe.BoundaryEdgesOn()
        fe.NonManifoldEdgesOn()
        fe.FeatureEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()
        edge_poly = fe.GetOutput()
        open_edges = edge_poly.GetNumberOfCells()
        if open_edges > 0:
            raise ValueError(
                f"Surface is not watertight (boundary/non-manifold edges: {open_edges}). "
                "Close the mesh before voxelization."
            )

    def _compute_grid(self, poly: "vtk.vtkPolyData") -> Tuple[Tuple[float, float, float],
                                                              Tuple[float, float, float],
                                                              Tuple[int, int, int]]:
        """Compute origin, spacing, and integer dimensions for vtkImageData."""
        bounds = [0.0] * 6
        poly.GetBounds(bounds)
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        extents = (xmax - xmin, ymax - ymin, zmax - zmin)
        max_extent = max(extents)
        if max_extent <= 0:
            raise ValueError("Degenerate bounds: zero extent in at least one dimension.")

        # Decide spacing
        if self.cfg.voxel_size is not None:
            spacing = (float(self.cfg.voxel_size),) * 3
            dims = tuple(max(1, int(math.ceil(extents[i] / spacing[i])) + 1 + 2 * self.cfg.margin_voxels)
                         for i in range(3))
        elif self.cfg.target_max_dim is not None:
            # Fit the longest side to target_max_dim (add margin after)
            s = max_extent / float(self.cfg.target_max_dim)
            spacing = (s, s, s)
            dims = tuple(max(1, int(math.ceil(extents[i] / s)) + 1 + 2 * self.cfg.margin_voxels)
                         for i in range(3))
        else:
            raise ValueError("Provide either voxel_size or target_max_dim in VoxelizationConfig.")

        # Origin: shift min corner outward by margin*spacing
        origin = (xmin - self.cfg.margin_voxels * spacing[0],
                  ymin - self.cfg.margin_voxels * spacing[1],
                  zmin - self.cfg.margin_voxels * spacing[2])

        return origin, spacing, dims

    def _guard_memory(self, dims: Tuple[int, int, int], dtype: np.dtype) -> None:
        """Abort if the array would exceed the configured memory limit."""
        voxels = np.int64(dims[0]) * np.int64(dims[1]) * np.int64(dims[2])
        bytes_per_voxel = np.dtype(dtype).itemsize
        gigabytes = (voxels * bytes_per_voxel) / (1024 ** 3)
        if gigabytes > self.cfg.memory_limit_gb:
            raise MemoryError(
                f"Requested grid {dims} of dtype {dtype} ≈ {gigabytes:.2f} GiB "
                f"exceeds limit {self.cfg.memory_limit_gb} GiB. "
                "Increase spacing / reduce target_max_dim, or raise memory_limit_gb."
            )

    def _voxelize(
        self,
        poly: "vtk.vtkPolyData",
        origin: Tuple[float, float, float],
        spacing: Tuple[float, float, float],
        dims: Tuple[int, int, int],
    ) -> np.ndarray:
        """Rasterize the closed surface into a binary volume using a stencil."""
        # Prepare image filled with FILL (not background!)
        img = vtk.vtkImageData()
        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDimensions(dims)  # (nx, ny, nz)
        vtk_dtype = self._numpy_dtype_to_vtk(self.cfg.dtype)
        img.AllocateScalars(vtk_dtype, 1)

        # IMPORTANT: fill input image with "inside" value
        in_arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
        in_arr[:] = self.cfg.fill_value

        # Build stencil from surface
        poly2st = vtk.vtkPolyDataToImageStencil()
        poly2st.SetInputData(poly)
        poly2st.SetOutputOrigin(origin)
        poly2st.SetOutputSpacing(spacing)
        poly2st.SetOutputWholeExtent(img.GetExtent())
        poly2st.Update()

        # Apply stencil: keep input (fill) inside, write background outside
        stencil = vtk.vtkImageStencil()
        stencil.SetInputData(img)
        stencil.SetStencilConnection(poly2st.GetOutputPort())
        stencil.ReverseStencilOff()  # inside = pass-through, outside = background
        stencil.SetBackgroundValue(self.cfg.background_value)
        stencil.Update()

        # Extract NumPy array
        out_img = stencil.GetOutput()
        out_arr = numpy_support.vtk_to_numpy(out_img.GetPointData().GetScalars()).copy()

        # VTK stores image data in z, y, x order; reshape + transpose to (nx, ny, nz)
        nx, ny, nz = dims
        out_arr = out_arr.reshape((nz, ny, nx))
        volume = np.transpose(out_arr, (2, 1, 0)).astype(self.cfg.dtype, copy=False)

        return volume

    @staticmethod
    def _numpy_dtype_to_vtk(dtype: np.dtype) -> int:
        """Map NumPy dtype → VTK scalar type."""
        dt = np.dtype(dtype)
        if dt == np.uint8:
            return vtk.VTK_UNSIGNED_CHAR
        if dt == np.int8:
            return vtk.VTK_CHAR
        if dt == np.uint16:
            return vtk.VTK_UNSIGNED_SHORT
        if dt == np.int16:
            return vtk.VTK_SHORT
        if dt == np.uint32:
            return vtk.VTK_UNSIGNED_INT
        if dt == np.int32:
            return vtk.VTK_INT
        if dt == np.float32:
            return vtk.VTK_FLOAT
        if dt == np.float64:
            return vtk.VTK_DOUBLE
        raise TypeError(f"Unsupported dtype for VTK image: {dtype!r}")


# ---------------------------------- CLI ----------------------------------- #

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Voxelize a watertight STL into a NumPy .npy binary volume (1=inside, 0=outside)."
    )
    p.add_argument("input_stl", help="Path to input .stl (must be watertight)")
    p.add_argument("output_npy", help="Path to output .npy (will be overwritten)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--voxel-size", type=float, default=None,
                       help="Voxel spacing in STL units (e.g., mm).")
    group.add_argument("--max-dim", dest="target_max_dim", type=int, default=256,
                       help="Fit the longest bbox side into this many voxels (default: 256).")
    p.add_argument("--margin-voxels", type=int, default=2, help="Padding around object (default: 2).")
    p.add_argument("--dtype", choices=["uint8", "uint16", "float32"], default="uint8",
                   help="Output dtype (default: uint8).")
    p.add_argument("--fill-value", type=float, default=1, help="Value for inside voxels (default: 1).")
    p.add_argument("--background-value", type=float, default=0, help="Value for outside voxels (default: 0).")
    p.add_argument("--memory-limit-gb", type=float, default=8.0,
                   help="Abort if array would exceed this size (default: 8 GB).")
    p.add_argument("--no-watertight-check", action="store_true",
                   help="Skip watertight validation (not recommended).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    np_dtype = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}[args.dtype]

    cfg = VoxelizationConfig(
        voxel_size=args.voxel_size,
        target_max_dim=args.target_max_dim,
        margin_voxels=args.margin_voxels,
        dtype=np_dtype,
        fill_value=np_dtype(args.fill_value),
        background_value=np_dtype(args.background_value),
        memory_limit_gb=args.memory_limit_gb,
        require_watertight=not args.no_watertight_check,
    )

    converter = STLToNumpyConverter(cfg)

    try:
        volume = converter.convert(args.input_stl)
        inside_fraction = float(volume.sum()) / (volume.size * (cfg.fill_value or 1))
        print(f"Inside voxels: {inside_fraction:.4%}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Save as .npy
    np.save(args.output_npy, volume)
    voxels = volume.size
    gb = volume.nbytes / (1024 ** 3)
    print(f"Saved {args.output_npy} with shape {tuple(volume.shape)} "
          f"({voxels} voxels, {gb:.2f} GiB, dtype={volume.dtype}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
