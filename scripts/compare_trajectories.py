#!/usr/bin/env python
"""
compare_trajectories.py - overlay CFPD vs diffusion predictions (2D + 3D).

Example
-------
    python scripts/compare_trajectories.py \
        --gt-npy   datasets/bb-part-0.42.npy \
        --pred-npz results/samples_128x1024x3.npz \
        --out      figs/compare.png
"""
from __future__ import annotations
try:
    from typing import Tuple, Literal, Optional
except ImportError:
    from typing import Tuple, Optional
    from typing_extensions import Literal

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fastdep.plotting import plot_xy_tracks, plot_3d_tracks
from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
from skimage import measure, draw

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Compare GT vs diffusion samples.")
parser.add_argument("--gt-npy", required=True, type=Path)
parser.add_argument("--pred-npz", required=True, type=Path)
parser.add_argument("--out", type=Path, help="Save figure instead of showing")
parser.add_argument("--n-train", type=int, default=512)
parser.add_argument("--n-sample", type=int, default=128)
parser.add_argument("--stl", type=Path, help="Optional STL mesh for 3D projection")
args = parser.parse_args()

gt = np.moveaxis(np.load(args.gt_npy), 1, 0)       # (P,T,3)
pred = np.load(args.pred_npz)["arr_0"]             # (P,T,3)

def plot_stl_projection(ax, stl_mesh, xy=(0,1), alpha=0.2, color="gray", stride=1):
    """
    Fast 2D projection using a single scatter call.
    Optionally subsample vertices with `stride` (e.g., 10 → keep every 10th point).
    """
    verts = stl_mesh.vectors.reshape(-1, 3)
    if stride > 1:
        verts = verts[::stride]

    x, y = verts[:, xy[0]], verts[:, xy[1]]

    # One scatter call only
    ax.scatter(x, y, s=0.1, color=color, alpha=alpha, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")

def plot_stl_edges_projection(ax, stl_mesh, xy=(0,1), alpha=0.8, color="gray", stride=1):
    """
    Plots triangle edges in 2D using a single LineCollection.
    """
    tri = stl_mesh.vectors[::stride] if stride > 1 else stl_mesh.vectors
    # tri shape: (N_tri, 3, 3); build edges (3 per tri)
    edges = np.concatenate([tri[:, [0,1]], tri[:, [1,2]], tri[:, [2,0]]], axis=0)
    segs = edges[:, :, xy]  # keep only xy coords -> shape (3*N_tri, 2, 2)

    lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=0.2, rasterized=True)
    ax.add_collection(lc)
    ax.set_aspect("equal", adjustable="datalim")

def plot_stl_3d(
    ax: plt.Axes,
    stl_mesh: mesh.Mesh,
    alpha: float = 0.2,
    color: str = "gray",
    linewidth: float = 0.0,
) -> None:
    """
    Vectorized 3D plot: add all triangles at once.
    """
    # stl_mesh.vectors: (N_triangles, 3, 3)
    collection = Poly3DCollection(
        stl_mesh.vectors,
        facecolors=color,
        alpha=alpha,
        linewidths=linewidth,
        edgecolors='none',  # set to color if you want edges
        rasterized=True     # helps when saving to PDF/SVG
    )
    ax.add_collection3d(collection)

    # Optional tight bounds:
    verts = stl_mesh.vectors.reshape(-1, 3)
    ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])


def rasterize_stl_projection(
    stl_mesh,
    plane: Tuple[int, int] = (0, 1),
    pixel_size: float = 1e-3,
    margin: float = 0.0,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize the STL mesh projection into a 2D binary mask.

    Parameters
    ----------
    stl_mesh : numpy-stl mesh
        Mesh with `.vectors` of shape (N_tri, 3, 3).
    plane : tuple[int, int]
        Axes to keep for projection (e.g., (0,1)=XY, (0,2)=XZ).
    pixel_size : float
        Size of each pixel in the projected plane units (e.g. meters).
    margin : float
        Extra padding (in world units) added around the bounding box.
    stride : int or None
        Optional triangle decimation factor (keep every `stride`-th tri).

    Returns
    -------
    mask : np.ndarray, dtype=bool
        2D binary mask with True where the STL projects.
    x_coords : np.ndarray
        X coordinates of mask columns (world units).
    y_coords : np.ndarray
        Y coordinates of mask rows (world units).
    """
    tris = stl_mesh.vectors if stride is None else stl_mesh.vectors[::stride]
    # Project to 2D
    tris_2d = tris[:, :, plane]  # (N_tri, 3, 2)

    # Bounding box
    xy_min = tris_2d.reshape(-1, 2).min(axis=0) - margin
    xy_max = tris_2d.reshape(-1, 2).max(axis=0) + margin

    width  = xy_max[0] - xy_min[0]
    height = xy_max[1] - xy_min[1]

    nx = int(np.ceil(width  / pixel_size))
    ny = int(np.ceil(height / pixel_size))

    if nx <= 0 or ny <= 0:
        raise ValueError("Invalid pixel_size or bounding box computed.")

    mask = np.zeros((ny, nx), dtype=bool)

    # Fill each triangle on the grid
    for tri in tris_2d:
        # Convert world coords → pixel indices
        col = (tri[:, 0] - xy_min[0]) / pixel_size  # x → columns
        row = (tri[:, 1] - xy_min[1]) / pixel_size  # y → rows
        rr, cc = draw.polygon(row, col, shape=mask.shape)
        mask[rr, cc] = True

    # Build coordinate arrays: cell centers
    x_coords = xy_min[0] + (np.arange(nx) + 0.5) * pixel_size
    y_coords = xy_min[1] + (np.arange(ny) + 0.5) * pixel_size

    return mask, x_coords, y_coords


def extract_outline_from_mask(
    mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    level: float = 0.5
) -> list[np.ndarray]:
    """
    Find iso-contours in the binary mask and map them back to world coordinates.

    Parameters
    ----------
    mask : np.ndarray
        Binary 2D mask where the mesh projects.
    x_coords : np.ndarray
        X coords of mask columns.
    y_coords : np.ndarray
        Y coords of mask rows.
    level : float
        Level passed to `find_contours` (0.5 for binary masks).

    Returns
    -------
    contours_world : list[np.ndarray]
        Each entry is (N_points, 2) array in world coordinates.
    """
    contours_pix = measure.find_contours(mask.astype(float), level=level)
    contours_world = []
    for c in contours_pix:
        # c[:, 0] are row indices (y), c[:, 1] are col indices (x)
        ys = np.interp(c[:, 0], np.arange(len(y_coords)), y_coords)
        xs = np.interp(c[:, 1], np.arange(len(x_coords)), x_coords)
        contours_world.append(np.column_stack([xs, ys]))
    return contours_world


def plot_stl_outline_image(
    ax,
    stl_mesh,
    plane: Tuple[int, int] = (0, 1),
    pixel_size: float = 1e-3,
    margin: float = 0.0,
    stride: Optional[int] = None,
    method_level: float = 0.5,
    color: str = "black",
    linewidth: float = 0.8,
    alpha: float = 1.0,
    show_mask: bool = False
):
    """
    Convenience wrapper: rasterize → contour → plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target 2D axes.
    stl_mesh : numpy-stl mesh
    plane : tuple[int, int]
        Axes to keep for projection.
    pixel_size : float
        Resolution of raster (smaller = finer, slower).
    margin : float
        Extra padding around bbox.
    stride : int or None
        Triangle decimation (speed vs fidelity).
    method_level : float
        Level for `find_contours` (0.5 for binary).
    color : str
    linewidth : float
    alpha : float
    show_mask : bool
        If True, display the rasterized mask as background.

    Returns
    -------
    contours_world : list[np.ndarray]
        Each array is a polyline (N_pts, 2) in world coords.
    """
    mask, xs, ys = rasterize_stl_projection(
        stl_mesh, plane=plane, pixel_size=pixel_size, margin=margin, stride=stride
    )
    contours_world = extract_outline_from_mask(mask, xs, ys, level=method_level)

    if show_mask:
        # Note imshow expects rows (y) downwards; origin='lower' to match coordinates
        ax.imshow(mask, extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  origin='lower', cmap='gray', alpha=0.3)

    for contour in contours_world:
        ax.plot(contour[:, 0], contour[:, 1], color=color,
                linewidth=linewidth, alpha=alpha)

    ax.set_aspect("equal", adjustable="datalim")
    return contours_world


if hasattr(args, "stl") and args.stl:
    stl_mesh: Optional[mesh.Mesh] = mesh.Mesh.from_file(str(args.stl))
else:
    stl_mesh: Optional[mesh.Mesh] = None
if hasattr(args, "stl") and args.stl:
    stl_mesh = mesh.Mesh.from_file(str(args.stl))
else:
    stl_mesh = None

fig = plt.figure(figsize=(20, 5))
xy_pairs = [(-3, -2), (-3, -1), (-2, -1)]  # (Z,Y), (Z,X), (Y,X)
labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

for i, (xy, (xlabel, ylabel)) in enumerate(zip(xy_pairs, labels), start=1):
    ax = fig.add_subplot(1, 4, i)
    plot_xy_tracks(gt[: args.n_train], range(args.n_train), xy=xy, ax=ax,
                   alpha=0.2, color="red", label="Ground Truth", 
                   linestyle="--")
    plot_xy_tracks(pred[: args.n_sample], range(args.n_sample), xy=xy, ax=ax,
                   alpha=0.4, color="blue", label="Diffusion",
                   linestyle="-")
    if stl_mesh is not None:
        plot_stl_outline_image(
            ax, stl_mesh, plane=xy, pixel_size=1e-4, margin=0.0001,
            stride=None, method_level=0.5, color="black", linewidth=0.8,
            alpha=0.8, show_mask=False
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel}-{ylabel}")
    handles = [
        Line2D([0], [0], color="red", linestyle="--", label="Ground Truth"),
        Line2D([0], [0], color="blue", linestyle="-", label="Diffusion"),
    ]
    ax.legend(handles=handles, loc="upper right")

ax3d = fig.add_subplot(1, 4, 4, projection="3d")
plot_3d_tracks(gt[: args.n_train], range(args.n_train), ax=ax3d,
               alpha=0.2, color="red", label="Ground Truth",
               linestyle="--")
plot_3d_tracks(pred[: args.n_sample], range(args.n_sample), ax=ax3d,
               alpha=0.4, color="blue", label="Diffusion",
               linestyle="-")
if stl_mesh is not None:
    plot_stl_3d(ax3d, stl_mesh, alpha=0.2, color="gray")
ax3d.set_title("3D")

fig.tight_layout()
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"), format="pdf")
    log.info("Figure saved to %s", args.out)
else:
    plt.show()
