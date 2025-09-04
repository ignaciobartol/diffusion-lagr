#!/usr/bin/env python3
"""
Simple 3D plot of a binary NumPy volume using Matplotlib's voxel renderer.

- Expects a .npy array shaped (nx, ny, nz) with 1s inside the object and 0s outside.
- Use --step to downsample large volumes for faster plotting.
- Use --save to write a PNG (helpful on headless HPC nodes).
"""

import argparse
import os
import sys
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="3D voxel plot of a binary NumPy volume (.npy).")
    p.add_argument("npy_path", help="Path to .npy file (volume with 0/1 values)")
    p.add_argument("--step", type=int, default=1,
                   help="Stride for downsampling (plot every Nth voxel along each axis). Default: 1 (no downsampling).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Value > threshold is considered 'solid'. Default: 0.5")
    p.add_argument("--alpha", type=float, default=0.7, help="Voxel face transparency (0..1). Default: 0.7")
    p.add_argument("--color", default="#ff7f0e", help="Voxel color (e.g., '#ff7f0e' or 'steelblue').")
    p.add_argument("--save", metavar="PNG_PATH", default=None,
                   help="If provided, save the figure to this PNG and exit (no GUI needed).")
    p.add_argument("--dpi", type=int, default=200, help="PNG DPI when using --save. Default: 200")
    return p.parse_args()

def main():
    args = parse_args()

    # Headless-friendly: if saving or no DISPLAY, use Agg backend
    if args.save or os.environ.get("DISPLAY", "") == "":
        import matplotlib
        matplotlib.use("Agg")  # no GUI
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

    vol = np.load(args.npy_path)
    if vol.ndim != 3:
        print(f"ERROR: expected a 3D array, got shape {vol.shape}", file=sys.stderr)
        return 2

    # Binarize (just in case it's not exactly 0/1)
    solid = vol > args.threshold

    # Downsample for speed if requested
    step = max(1, int(args.step))
    if step > 1:
        solid = solid[::step, ::step, ::step]

    nx, ny, nz = solid.shape
    voxels_to_plot = int(solid.sum())
    total_voxels = solid.size
    if voxels_to_plot == 0:
        print("WARNING: nothing to plot (no voxels above threshold).", file=sys.stderr)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((nx, ny, nz))  # preserve proportions

    # colors/alpha arrays matching the mask
    facecolors = np.empty(solid.shape, dtype=object)
    facecolors[:] = args.color
    filled = solid

    # Plot voxels; edgecolor lightly to see structure
    ax.voxels(filled, facecolors=facecolors, edgecolor="k", linewidth=0.1, alpha=args.alpha)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{os.path.basename(args.npy_path)}  |  shape={vol.shape}  step={step}  filled={voxels_to_plot}/{total_voxels}")

    # Tidy view
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()

    if args.save:
        out = args.save
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()
    return 0

if __name__ == "__main__":
    sys.exit(main())

