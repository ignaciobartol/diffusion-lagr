#!/usr/bin/env python
"""compare_trajectories.py

Overlay CFPD ground-truth trajectories vs diffusion predictions (2D projections + 3D).

Key features
------------
- Loads trajectories from: .npy, .npz, or .h5
- Preserves particle track structure (PTC: particles x timesteps x coords)
- Optional geometry-based rescaling of predictions from normalized/voxel coordinates to world
- Optional STL visualization:
    * 2D: silhouette/outline from rasterized projection
    * 3D: wireframe edges using Line3DCollection for efficiency

Examples
--------
# legacy CLI (still supported)
python scripts/compare_trajectories.py \
  --gt-npy datasets/bb-part-0.42.npy \
  --pred-npz results/samples_256x1024x3.npz \
  --out figs/compare.png

# new CLI
python scripts/compare_trajectories.py \
  --gt datasets/bb-part-16384.h5 --gt-key train \
  --pred results/12/bb_samples_guided.npz --pred-key arr_0 \
  --geometry-npz geometry/processed/bb_geometry.npz \
  --normalization-space world \
  --stl geometry/raw/bb-geom.stl \
  --out results/12/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import draw, measure

# --- local imports (repo-root on path) ---
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from fastdep.plotting import plot_3d_tracks, plot_xy_tracks  # noqa: E402

# STL loading (numpy-stl)
try:
    from stl import mesh as stl_mesh  # type: ignore
except Exception as e:  # pragma: no cover
    stl_mesh = None


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("compare")


# ----------------------------- Trajectory I/O ----------------------------- #

def _npz_pick_key(npz: np.lib.npyio.NpzFile, preferred: Optional[str] = None) -> str:
    keys = list(npz.files)
    if preferred is not None:
        if preferred not in keys:
            raise KeyError(f"Requested key '{preferred}' not in NPZ. Available keys: {keys}")
        return preferred
    for k in ("arr_0", "samples", "train", "traj", "trajectories"):
        if k in keys:
            return k
    if len(keys) == 1:
        return keys[0]
    raise ValueError(f"NPZ contains multiple arrays {keys}. Provide --pred-key / --gt-key.")


def to_PTC(arr: np.ndarray, layout: str) -> np.ndarray:
    """Convert a trajectory tensor to PTC layout: (P, T, 3)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 tensor, got shape {arr.shape}")

    if layout == "PTC":
        out = arr
    elif layout == "TPC":
        out = np.moveaxis(arr, 0, 1)  # (T,P,3)->(P,T,3)
    elif layout == "PCT":
        out = np.transpose(arr, (0, 2, 1))  # (P,3,T)->(P,T,3)
    elif layout == "CPT":
        out = np.transpose(arr, (1, 2, 0))  # (3,P,T)->(P,T,3)
    else:
        raise ValueError(f"Unknown layout '{layout}'. Choose from PTC,TPC,PCT,CPT")

    if out.shape[-1] != 3:
        raise ValueError(f"Expected last dim=3 after conversion, got {out.shape}")
    return out.astype(np.float32, copy=False)


def load_trajectories(
    path: Path,
    *,
    key: Optional[str],
    layout: str,
) -> np.ndarray:
    """Load trajectories from .npy/.npz/.h5 and return (P,T,3) float32."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()

    if suf == ".npy":
        # Legacy convention in this repo: GT .npy stored as (T,P,3)
        if layout == "auto":
            layout_eff = "TPC"
        else:
            layout_eff = layout
        arr = np.load(path)
        return to_PTC(arr, layout_eff)

    if suf == ".npz":
        # Repo convention: pred samples stored as (P,T,3)
        if layout == "auto":
            layout_eff = "PTC"
        else:
            layout_eff = layout
        with np.load(path) as z:
            k = _npz_pick_key(z, key)
            arr = z[k]
        return to_PTC(arr, layout_eff)

    if suf in (".h5", ".hdf5"):
        # Typical convention: dataset stored as (P,T,3)
        if layout == "auto":
            layout_eff = "PTC"
        else:
            layout_eff = layout
        k = key or "train"
        with h5py.File(path, "r") as f:
            if k not in f:
                raise KeyError(f"H5 key '{k}' not found. Available keys: {list(f.keys())}")
            arr = np.array(f[k], dtype=np.float32)
        return to_PTC(arr, layout_eff)

    raise ValueError(f"Unsupported file type: {path} (expected .npy/.npz/.h5)")


# ----------------------------- STL outline plotting ----------------------------- #

def _require_stl():
    if stl_mesh is None:
        raise ImportError(
            "numpy-stl is required for STL visualization. Install with: pip install numpy-stl"
        )


def load_stl_triangles(path: Path) -> np.ndarray:
    """Return STL triangles as (N_tri,3,3) float32."""
    _require_stl()
    m = stl_mesh.Mesh.from_file(str(path))
    tris = np.asarray(m.vectors, dtype=np.float32)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected STL triangles shape: {tris.shape}")
    return tris


def rasterize_stl_projection(
    tris: np.ndarray,
    plane: Tuple[int, int] = (0, 1),
    pixel_size: float = 1e-4,
    margin: float = 0.0,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize STL projection into a 2D binary mask."""
    t = tris if stride is None else tris[:: int(stride)]
    t2d = t[:, :, plane]  # (N,3,2)

    xy_min = t2d.reshape(-1, 2).min(axis=0) - margin
    xy_max = t2d.reshape(-1, 2).max(axis=0) + margin

    width = float(xy_max[0] - xy_min[0])
    height = float(xy_max[1] - xy_min[1])

    nx = int(np.ceil(width / pixel_size))
    ny = int(np.ceil(height / pixel_size))
    nx = max(nx, 1)
    ny = max(ny, 1)

    mask = np.zeros((ny, nx), dtype=bool)

    for tri in t2d:
        col = (tri[:, 0] - xy_min[0]) / pixel_size
        row = (tri[:, 1] - xy_min[1]) / pixel_size
        rr, cc = draw.polygon(row, col, shape=mask.shape)
        mask[rr, cc] = True

    x_coords = xy_min[0] + (np.arange(nx) + 0.5) * pixel_size
    y_coords = xy_min[1] + (np.arange(ny) + 0.5) * pixel_size
    return mask, x_coords, y_coords


def extract_outline_from_mask(
    mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    level: float = 0.5,
) -> list[np.ndarray]:
    contours_pix = measure.find_contours(mask.astype(float), level=level)
    contours_world: list[np.ndarray] = []
    for c in contours_pix:
        ys = np.interp(c[:, 0], np.arange(len(y_coords)), y_coords)
        xs = np.interp(c[:, 1], np.arange(len(x_coords)), x_coords)
        contours_world.append(np.column_stack([xs, ys]))
    return contours_world


def plot_stl_outline_image(
    ax: plt.Axes,
    tris: np.ndarray,
    plane: Tuple[int, int] = (0, 1),
    pixel_size: float = 1e-4,
    margin: float = 0.0,
    stride: Optional[int] = None,
    color: str = "black",
    linewidth: float = 0.8,
    alpha: float = 0.8,
) -> None:
    mask, xs, ys = rasterize_stl_projection(
        tris, plane=plane, pixel_size=pixel_size, margin=margin, stride=stride
    )
    contours = extract_outline_from_mask(mask, xs, ys, level=0.5)
    for c in contours:
        ax.plot(c[:, 0], c[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    ax.set_aspect("equal", adjustable="datalim")


def plot_stl_wireframe_3d(
    ax: plt.Axes,
    tris: np.ndarray,
    stride: int = 1,
    color: str = "0.3",
    linewidth: float = 0.15,
    alpha: float = 0.35,
) -> None:
    """Plot STL as 3D wireframe edges using a single Line3DCollection."""
    t = tris[:: max(1, int(stride))]
    edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=0)  # (3N,2,3)
    lc = Line3DCollection(edges, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(lc)
    verts = tris.reshape(-1, 3)
    ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])


# ----------------------------- Main plotting ----------------------------- #

def create_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare GT vs diffusion trajectory samples.")

    # New unified args (preferred)
    p.add_argument("--gt", type=Path, default=None, help="GT trajectories: .npy/.npz/.h5")
    p.add_argument("--pred", type=Path, default=None, help="Pred trajectories: .npy/.npz/.h5")

    # Legacy args (still supported)
    p.add_argument("--gt-npy", type=Path, default=None, help="(deprecated) GT .npy")
    p.add_argument("--pred-npz", type=Path, default=None, help="(deprecated) pred .npz")

    # Dataset keys / layouts
    p.add_argument("--gt-key", type=str, default="train", help="Key inside GT .h5/.npz")
    p.add_argument("--pred-key", type=str, default="arr_0", help="Key inside pred .h5/.npz")
    p.add_argument("--gt-layout", type=str, default="auto", choices=("auto", "PTC", "TPC", "PCT", "CPT"))
    p.add_argument("--pred-layout", type=str, default="auto", choices=("auto", "PTC", "TPC", "PCT", "CPT"))

    # Plotting / selection
    p.add_argument("--n-train", type=int, default=256, help="#GT trajectories to plot")
    p.add_argument("--n-sample", type=int, default=256, help="#pred trajectories to plot")

    # Optional STL / geometry
    p.add_argument("--stl", type=Path, help="Optional STL mesh for outline/wireframe overlays")
    p.add_argument(
        "--stl-outline-pixel-size",
        type=float,
        default=1e-4,
        help="Raster pixel size (world units) for 2D STL silhouette extraction",
    )
    p.add_argument(
        "--stl-outline-margin",
        type=float,
        default=1e-4,
        help="Margin (world units) around STL bbox for outline extraction",
    )
    p.add_argument(
        "--stl-tri-stride",
        type=int,
        default=1,
        help="Triangle decimation for STL outline/wireframe (1=no decimation)",
    )

    p.add_argument(
        "--geometry-npz",
        type=Path,
        help="Geometry .npz used for training (origin/spacing/binary). Used to rescale predictions.",
    )
    p.add_argument(
        "--normalization-space",
        default="normalized",
        choices=("normalized", "voxel", "world"),
        help="Coordinate space of predicted trajectories before rescaling via --geometry-npz.",
    )

    # Output
    p.add_argument("--out", type=Path, help="Output path: directory OR .png file. If omitted, shows interactively.")

    return p


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve --gt/--pred with legacy fallbacks."""
    if args.gt is None:
        if args.gt_npy is None:
            raise SystemExit("Provide --gt (preferred) or --gt-npy (legacy)")
        args.gt = args.gt_npy
    if args.pred is None:
        if args.pred_npz is None:
            raise SystemExit("Provide --pred (preferred) or --pred-npz (legacy)")
        args.pred = args.pred_npz
    return args.gt, args.pred


def rescale_predictions_to_world(pred_ptc: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Rescale predicted coordinates to world using geometry metadata, if provided."""
    if args.geometry_npz is None:
        return pred_ptc

    geo = np.load(args.geometry_npz)
    origin = np.asarray(geo["origin"], dtype=np.float32)
    spacing = np.asarray(geo["spacing"], dtype=np.float32)
    dims = np.asarray(geo["binary"].shape, dtype=np.float32)
    denom = np.maximum(dims - 1.0, 1.0)

    pred = pred_ptc.astype(np.float32, copy=False)

    if args.normalization_space == "normalized":
        coords = (pred + 1.0) * 0.5 * denom
    elif args.normalization_space == "voxel":
        coords = pred
    else:
        coords = (pred - origin) / spacing

    pred_world = coords * spacing + origin

    log.info(
        "Rescaled predictions to world using geometry %s (%s space).",
        args.geometry_npz,
        args.normalization_space,
    )
    return pred_world.astype(np.float32, copy=False)


def resolve_output_paths(out: Optional[Path]) -> tuple[Optional[Path], Optional[Path]]:
    """Return (png_path, pdf_path) or (None,None) if out is None."""
    if out is None:
        return None, None

    out = Path(out)

    # If user provided a directory OR a path without suffix, save inside as compare_trajectories.png/pdf
    if out.suffix.lower() not in (".png", ".pdf"):
        out_dir = out
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "compare_trajectories.png"
        pdf = out_dir / "compare_trajectories.pdf"
        return png, pdf

    # User provided explicit file
    png = out if out.suffix.lower() == ".png" else out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    return png, pdf


def main(argv: Optional[list[str]] = None) -> int:
    parser = create_argparser()
    args = parser.parse_args(argv)

    gt_path, pred_path = resolve_inputs(args)

    # Load trajectories
    gt = load_trajectories(gt_path, key=args.gt_key, layout=args.gt_layout)  # (P,T,3)
    pred = load_trajectories(pred_path, key=args.pred_key, layout=args.pred_layout)  # (P,T,3)

    # Rescale predictions to world if geometry is provided
    pred = rescale_predictions_to_world(pred, args)

    # Optional STL triangles
    tris: Optional[np.ndarray] = None
    if args.stl is not None:
        tris = load_stl_triangles(args.stl)

    # Slice for plotting
    plot_gt = gt[: int(args.n_train)]
    plot_pred = pred[: int(args.n_sample)]

    fig = plt.figure(figsize=(20, 5))

    # Coordinate pairs for 2D projections
    xy_pairs = [(-3, -2), (-3, -1), (-2, -1)]  # (X,Y), (X,Z), (Y,Z) using negative indices
    labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

    for i, (xy, (xlabel, ylabel)) in enumerate(zip(xy_pairs, labels), start=1):
        ax = fig.add_subplot(1, 4, i)

        # STL silhouette behind trajectories
        if tris is not None:
            plot_stl_outline_image(
                ax,
                tris,
                plane=xy,
                pixel_size=float(args.stl_outline_pixel_size),
                margin=float(args.stl_outline_margin),
                stride=int(args.stl_tri_stride) if args.stl_tri_stride and args.stl_tri_stride > 1 else None,
                color="black",
                linewidth=0.8,
                alpha=0.65,
            )

        plot_xy_tracks(
            plot_gt,
            range(min(plot_gt.shape[0], int(args.n_train))),
            xy=xy,
            ax=ax,
            alpha=0.4,
            color="C2",
            label="Ground Truth",
            linestyle="-",
        )
        plot_xy_tracks(
            plot_pred,
            range(min(plot_pred.shape[0], int(args.n_sample))),
            xy=xy,
            ax=ax,
            alpha=0.4,
            color="C0",
            label="Diffusion",
            linestyle="-",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel}-{ylabel}")

        handles = [
            Line2D([0], [0], color="C2", linestyle="-", label="Ground Truth"),
            Line2D([0], [0], color="C0", linestyle="-", label="Guided-DDPM"),
        ]
        if tris is not None:
            handles = [Line2D([0], [0], color="black", linestyle="-", label="STL outline")] + handles
        ax.legend(handles=handles, loc="upper right")

    # 3D view
    ax3d = fig.add_subplot(1, 4, 4, projection="3d")

    plot_3d_tracks(
        plot_gt,
        range(min(plot_gt.shape[0], int(args.n_train))),
        ax=ax3d,
        alpha=0.4,
        color="C2",
        label="Ground Truth",
        linestyle="-",
    )
    plot_3d_tracks(
        plot_pred,
        range(min(plot_pred.shape[0], int(args.n_sample))),
        ax=ax3d,
        alpha=0.4,
        color="C0",
        label="Guided-DDPM",
        linestyle="-",
    )

    # if tris is not None:
    #     plot_stl_wireframe_3d(
    #         ax3d,
    #         tris,
    #         stride=max(1, int(args.stl_tri_stride)),
    #         color="0.3",
    #         linewidth=0.12,
    #         alpha=0.25,
    #     )

    ax3d.set_title("3D")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    fig.tight_layout()

    out_png, out_pdf = resolve_output_paths(args.out)
    if out_png is None:
        plt.show()
        return 0

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)

    log.info("Saved figure: %s", out_png)
    log.info("Saved figure: %s", out_pdf)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
